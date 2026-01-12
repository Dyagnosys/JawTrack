"""
JawTrack Stripe Payment Integration
Business model: SaaS with tiered pricing for jaw motion assessments
"""

import os
import stripe
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List
from enum import Enum
import json

# Stripe API Key (set via environment variable)
stripe.api_key = os.getenv("STRIPE_SECRET_KEY", "sk_test_your_key_here")

class PricingTier(Enum):
    FREE = "free"
    PROFESSIONAL = "professional"
    CLINICAL = "clinical"
    ENTERPRISE = "enterprise"

@dataclass
class PricingPlan:
    tier: PricingTier
    name: str
    price_monthly: float
    price_yearly: float
    assessments_per_month: int  # -1 for unlimited
    features: List[str]
    stripe_price_id_monthly: str
    stripe_price_id_yearly: str

# Pricing Configuration
PRICING_PLANS = {
    PricingTier.FREE: PricingPlan(
        tier=PricingTier.FREE,
        name="Free",
        price_monthly=0,
        price_yearly=0,
        assessments_per_month=3,
        features=[
            "3 assessments per month",
            "Basic jaw opening measurement",
            "Standard report",
            "Community support"
        ],
        stripe_price_id_monthly="",
        stripe_price_id_yearly=""
    ),
    PricingTier.PROFESSIONAL: PricingPlan(
        tier=PricingTier.PROFESSIONAL,
        name="Professional",
        price_monthly=29.99,
        price_yearly=299.99,
        assessments_per_month=50,
        features=[
            "50 assessments per month",
            "Full metric suite (velocity, asymmetry, trajectory)",
            "Auto-calibration",
            "Clinical interpretation report",
            "CSV data export",
            "Email support"
        ],
        stripe_price_id_monthly="price_professional_monthly",
        stripe_price_id_yearly="price_professional_yearly"
    ),
    PricingTier.CLINICAL: PricingPlan(
        tier=PricingTier.CLINICAL,
        name="Clinical",
        price_monthly=99.99,
        price_yearly=999.99,
        assessments_per_month=500,
        features=[
            "500 assessments per month",
            "All Professional features",
            "AI-powered clinical insights",
            "Patient database integration",
            "HIPAA compliance tools",
            "Priority support",
            "Custom branding"
        ],
        stripe_price_id_monthly="price_clinical_monthly",
        stripe_price_id_yearly="price_clinical_yearly"
    ),
    PricingTier.ENTERPRISE: PricingPlan(
        tier=PricingTier.ENTERPRISE,
        name="Enterprise",
        price_monthly=0,  # Custom pricing
        price_yearly=0,
        assessments_per_month=-1,  # Unlimited
        features=[
            "Unlimited assessments",
            "All Clinical features",
            "Multi-location support",
            "API access",
            "Custom integrations",
            "Dedicated account manager",
            "SLA guarantee",
            "On-premise deployment option"
        ],
        stripe_price_id_monthly="",
        stripe_price_id_yearly=""
    )
}

@dataclass
class Customer:
    id: str
    email: str
    name: str
    stripe_customer_id: Optional[str] = None
    tier: PricingTier = PricingTier.FREE
    subscription_id: Optional[str] = None
    assessments_used: int = 0
    assessments_reset_date: Optional[datetime] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.assessments_reset_date is None:
            self.assessments_reset_date = datetime.now() + timedelta(days=30)

class StripePaymentManager:
    """Manages Stripe payments and subscriptions for JawTrack"""
    
    def __init__(self):
        self.customers: Dict[str, Customer] = {}
        
    def create_customer(self, email: str, name: str) -> Customer:
        """Create a new customer in Stripe and local database"""
        try:
            # Create Stripe customer
            stripe_customer = stripe.Customer.create(
                email=email,
                name=name,
                metadata={"source": "jawtrack"}
            )
            
            customer = Customer(
                id=f"cust_{datetime.now().timestamp()}",
                email=email,
                name=name,
                stripe_customer_id=stripe_customer.id
            )
            
            self.customers[customer.id] = customer
            return customer
            
        except stripe.error.StripeError as e:
            raise Exception(f"Failed to create customer: {str(e)}")
    
    def create_checkout_session(
        self, 
        customer_id: str, 
        tier: PricingTier, 
        billing_period: str = "monthly"
    ) -> str:
        """Create a Stripe Checkout session for subscription"""
        
        customer = self.customers.get(customer_id)
        if not customer:
            raise ValueError("Customer not found")
            
        plan = PRICING_PLANS.get(tier)
        if not plan or tier == PricingTier.FREE:
            raise ValueError("Invalid pricing tier for checkout")
        
        price_id = (plan.stripe_price_id_monthly 
                   if billing_period == "monthly" 
                   else plan.stripe_price_id_yearly)
        
        try:
            session = stripe.checkout.Session.create(
                customer=customer.stripe_customer_id,
                payment_method_types=["card"],
                line_items=[{
                    "price": price_id,
                    "quantity": 1
                }],
                mode="subscription",
                success_url="https://jawtrack.app/success?session_id={CHECKOUT_SESSION_ID}",
                cancel_url="https://jawtrack.app/pricing",
                metadata={
                    "customer_id": customer_id,
                    "tier": tier.value
                }
            )
            
            return session.url
            
        except stripe.error.StripeError as e:
            raise Exception(f"Failed to create checkout session: {str(e)}")
    
    def handle_webhook(self, payload: bytes, sig_header: str) -> Dict:
        """Handle Stripe webhook events"""
        
        webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET", "whsec_test")
        
        try:
            event = stripe.Webhook.construct_event(
                payload, sig_header, webhook_secret
            )
        except ValueError:
            raise ValueError("Invalid payload")
        except stripe.error.SignatureVerificationError:
            raise ValueError("Invalid signature")
        
        # Handle specific events
        if event.type == "checkout.session.completed":
            session = event.data.object
            self._handle_subscription_created(session)
            
        elif event.type == "customer.subscription.updated":
            subscription = event.data.object
            self._handle_subscription_updated(subscription)
            
        elif event.type == "customer.subscription.deleted":
            subscription = event.data.object
            self._handle_subscription_cancelled(subscription)
            
        elif event.type == "invoice.payment_failed":
            invoice = event.data.object
            self._handle_payment_failed(invoice)
            
        return {"status": "success", "event_type": event.type}
    
    def _handle_subscription_created(self, session):
        """Handle new subscription"""
        customer_id = session.metadata.get("customer_id")
        tier = PricingTier(session.metadata.get("tier"))
        
        if customer_id in self.customers:
            customer = self.customers[customer_id]
            customer.tier = tier
            customer.subscription_id = session.subscription
            customer.assessments_used = 0
            customer.assessments_reset_date = datetime.now() + timedelta(days=30)
    
    def _handle_subscription_updated(self, subscription):
        """Handle subscription update"""
        for customer in self.customers.values():
            if customer.subscription_id == subscription.id:
                # Update tier based on price
                # Implementation depends on your price structure
                break
    
    def _handle_subscription_cancelled(self, subscription):
        """Handle subscription cancellation"""
        for customer in self.customers.values():
            if customer.subscription_id == subscription.id:
                customer.tier = PricingTier.FREE
                customer.subscription_id = None
                break
    
    def _handle_payment_failed(self, invoice):
        """Handle failed payment"""
        # Send notification, retry logic, etc.
        pass
    
    def check_assessment_quota(self, customer_id: str) -> Dict:
        """Check if customer can perform an assessment"""
        customer = self.customers.get(customer_id)
        if not customer:
            return {"allowed": False, "reason": "Customer not found"}
        
        plan = PRICING_PLANS[customer.tier]
        
        # Reset counter if needed
        if datetime.now() > customer.assessments_reset_date:
            customer.assessments_used = 0
            customer.assessments_reset_date = datetime.now() + timedelta(days=30)
        
        # Check quota
        if plan.assessments_per_month == -1:  # Unlimited
            return {"allowed": True, "remaining": -1}
        
        remaining = plan.assessments_per_month - customer.assessments_used
        
        return {
            "allowed": remaining > 0,
            "remaining": remaining,
            "used": customer.assessments_used,
            "limit": plan.assessments_per_month,
            "reset_date": customer.assessments_reset_date.isoformat()
        }
    
    def record_assessment(self, customer_id: str) -> bool:
        """Record an assessment usage"""
        quota = self.check_assessment_quota(customer_id)
        
        if not quota["allowed"]:
            return False
        
        customer = self.customers[customer_id]
        customer.assessments_used += 1
        return True
    
    def get_customer_portal_url(self, customer_id: str) -> str:
        """Get Stripe Customer Portal URL for subscription management"""
        customer = self.customers.get(customer_id)
        if not customer or not customer.stripe_customer_id:
            raise ValueError("Customer not found")
        
        try:
            session = stripe.billing_portal.Session.create(
                customer=customer.stripe_customer_id,
                return_url="https://jawtrack.app/dashboard"
            )
            return session.url
        except stripe.error.StripeError as e:
            raise Exception(f"Failed to create portal session: {str(e)}")


class UsageBasedBilling:
    """Alternative: Pay-per-assessment billing model"""
    
    PRICE_PER_ASSESSMENT = 1.99
    BULK_DISCOUNTS = {
        10: 0.10,   # 10% off for 10+
        50: 0.20,   # 20% off for 50+
        100: 0.30,  # 30% off for 100+
    }
    
    @classmethod
    def calculate_price(cls, num_assessments: int) -> Dict:
        """Calculate price with bulk discounts"""
        base_price = num_assessments * cls.PRICE_PER_ASSESSMENT
        
        discount = 0
        for threshold, disc in sorted(cls.BULK_DISCOUNTS.items(), reverse=True):
            if num_assessments >= threshold:
                discount = disc
                break
        
        final_price = base_price * (1 - discount)
        
        return {
            "num_assessments": num_assessments,
            "base_price": base_price,
            "discount_percent": discount * 100,
            "discount_amount": base_price - final_price,
            "final_price": round(final_price, 2)
        }
    
    @classmethod
    def create_payment_intent(cls, customer_id: str, num_assessments: int) -> str:
        """Create a one-time payment for assessment credits"""
        pricing = cls.calculate_price(num_assessments)
        
        try:
            intent = stripe.PaymentIntent.create(
                amount=int(pricing["final_price"] * 100),  # Cents
                currency="usd",
                metadata={
                    "customer_id": customer_id,
                    "num_assessments": num_assessments,
                    "type": "assessment_credits"
                }
            )
            return intent.client_secret
        except stripe.error.StripeError as e:
            raise Exception(f"Failed to create payment intent: {str(e)}")


# Pricing page data generator
def get_pricing_page_data() -> List[Dict]:
    """Generate pricing data for frontend display"""
    pricing_data = []
    
    for tier, plan in PRICING_PLANS.items():
        pricing_data.append({
            "tier": plan.tier.value,
            "name": plan.name,
            "price_monthly": plan.price_monthly,
            "price_yearly": plan.price_yearly,
            "yearly_savings": round((plan.price_monthly * 12) - plan.price_yearly, 2) if plan.price_yearly > 0 else 0,
            "assessments": "Unlimited" if plan.assessments_per_month == -1 else plan.assessments_per_month,
            "features": plan.features,
            "popular": tier == PricingTier.CLINICAL,
            "cta": "Contact Sales" if tier == PricingTier.ENTERPRISE else ("Get Started" if tier == PricingTier.FREE else "Subscribe")
        })
    
    return pricing_data
