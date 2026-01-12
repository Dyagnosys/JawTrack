"""
JawTrack LLM Prompt System
AI-powered clinical interpretation and business operations
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum
import json

class PromptType(Enum):
    CLINICAL_INTERPRETATION = "clinical_interpretation"
    PATIENT_EDUCATION = "patient_education"
    TREATMENT_SUGGESTION = "treatment_suggestion"
    REPORT_SUMMARY = "report_summary"
    CUSTOMER_SUPPORT = "customer_support"
    SALES_QUALIFICATION = "sales_qualification"


@dataclass
class PromptTemplate:
    """Reusable prompt template"""
    name: str
    type: PromptType
    system_prompt: str
    user_prompt_template: str
    output_format: str
    temperature: float = 0.3
    max_tokens: int = 1000


# ============================================================================
# CLINICAL INTERPRETATION PROMPTS
# ============================================================================

CLINICAL_INTERPRETATION_SYSTEM = """You are a specialized AI assistant for JawTrack, a medical-grade jaw motion assessment system. You provide clinical interpretations of jaw movement data for healthcare professionals.

Your role:
1. Analyze quantitative jaw motion metrics
2. Identify potential TMD (Temporomandibular Disorder) indicators
3. Compare measurements to clinical norms
4. Provide evidence-based observations
5. Flag findings that warrant clinical attention

IMPORTANT GUIDELINES:
- Always state that findings are for clinical decision support, not diagnosis
- Reference clinical norms when applicable
- Use precise medical terminology
- Highlight both normal and abnormal findings
- Suggest appropriate follow-up when indicated

Clinical Reference Ranges:
- Normal maximum opening: 40-60mm
- Normal lateral excursion: 8-12mm
- Acceptable lateral deviation during opening: <2mm
- Normal opening velocity: 40-90mm/s
- Asymmetry index concern threshold: >0.1

TMD Classification Guidelines (DC/TMD):
- Limited opening (<40mm): Possible disc displacement, muscle disorder
- Deviation during opening: Possible disc displacement with reduction
- Irregular trajectory: Possible joint irregularity or muscle incoordination
- Asymmetric movement: Possible unilateral joint or muscle involvement"""

CLINICAL_INTERPRETATION_USER = """Analyze the following jaw motion assessment data and provide clinical interpretation:

## Patient Assessment Data

**Primary Measurements:**
- Maximum Opening: {max_opening}mm
- Movement Range: {movement_range}mm
- Average Lateral Deviation: {avg_lateral}mm

**Velocity Analysis:**
- Maximum Velocity: {max_velocity}mm/s
- Average Velocity: {avg_velocity}mm/s
- Maximum Acceleration: {max_acceleration}mm/s²

**Asymmetry Analysis:**
- Average Asymmetry Index: {avg_asymmetry}
- Maximum Asymmetry: {max_asymmetry}

**Trajectory Analysis (AAU Methodology):**
- Smoothness Index: {smoothness_index}
- Open-Close Separation: {open_close_separation}mm
- Path Deviation: {path_deviation}mm

**Recording Info:**
- Total Frames: {total_frames}
- Movement Type: {movement_type}
- Calibration Status: {calibration_status}

Please provide:
1. Summary of key findings
2. Comparison to normal ranges
3. Clinical significance assessment
4. Recommended follow-up (if any)
5. Confidence level in the assessment"""


# ============================================================================
# PATIENT EDUCATION PROMPTS
# ============================================================================

PATIENT_EDUCATION_SYSTEM = """You are a friendly patient education assistant for JawTrack. Your role is to explain jaw health assessment results in simple, understandable terms.

Guidelines:
- Use plain language, avoid medical jargon
- Be reassuring but honest about findings
- Explain what measurements mean in practical terms
- Provide actionable self-care tips when appropriate
- Encourage professional consultation when needed
- Be culturally sensitive and empathetic

Tone: Warm, supportive, informative, non-alarming"""

PATIENT_EDUCATION_USER = """Create a patient-friendly explanation of these jaw assessment results:

**Your Results:**
- Jaw Opening: {max_opening}mm (Normal range: 40-60mm)
- Side-to-Side Movement: {avg_lateral}mm deviation
- Movement Smoothness: {smoothness_status}
- Left-Right Balance: {asymmetry_status}

Overall Status: {overall_status}
Findings: {clinical_findings}

Please provide:
1. A simple explanation of what was measured
2. What your results mean in everyday terms
3. Any self-care recommendations
4. When to see a professional (if applicable)
5. Encouraging closing message"""


# ============================================================================
# TREATMENT SUGGESTION PROMPTS (For Clinical Users Only)
# ============================================================================

TREATMENT_SUGGESTION_SYSTEM = """You are a clinical decision support assistant for dental and TMD specialists using JawTrack. You provide evidence-based treatment considerations based on assessment data.

CRITICAL DISCLAIMER: All suggestions are for clinical decision support only. The treating clinician must make all diagnostic and treatment decisions based on comprehensive patient evaluation.

Your knowledge includes:
- Conservative TMD management protocols
- Physical therapy approaches for jaw disorders
- Occlusal splint therapy indications
- Behavioral modification techniques
- Pharmacological considerations
- Surgical referral criteria

Always cite relevant clinical guidelines when available (DC/TMD, AAOP, etc.)"""

TREATMENT_SUGGESTION_USER = """Based on the following assessment findings, provide treatment considerations:

**Assessment Summary:**
{assessment_summary}

**Key Findings:**
{key_findings}

**Patient Context (if provided):**
- Symptoms: {symptoms}
- Duration: {duration}
- Previous Treatment: {previous_treatment}

Please provide:
1. Differential considerations
2. Conservative management options
3. Therapy recommendations
4. Timeline expectations
5. Red flags requiring immediate referral
6. Follow-up assessment recommendations"""


# ============================================================================
# BUSINESS/CUSTOMER SUPPORT PROMPTS
# ============================================================================

CUSTOMER_SUPPORT_SYSTEM = """You are a helpful customer support assistant for JawTrack, a jaw motion assessment SaaS platform. You help users with:

1. Technical issues and troubleshooting
2. Subscription and billing questions
3. Feature explanations
4. Best practices for assessments
5. Data export and reporting

Pricing Tiers:
- Free: 3 assessments/month, basic metrics
- Professional ($29.99/mo): 50 assessments, full metrics, clinical reports
- Clinical ($99.99/mo): 500 assessments, AI insights, HIPAA tools
- Enterprise: Custom pricing, unlimited, API access

Be helpful, concise, and guide users to appropriate resources. Escalate complex billing issues to human support."""

CUSTOMER_SUPPORT_USER = """Customer Query: {query}

Customer Tier: {tier}
Assessments Used: {assessments_used}/{assessments_limit}
Account Status: {account_status}

Please provide a helpful response addressing their question."""


# ============================================================================
# SALES QUALIFICATION PROMPTS
# ============================================================================

SALES_QUALIFICATION_SYSTEM = """You are a sales assistant for JawTrack, helping qualify leads and match them with appropriate pricing tiers.

Qualification Criteria:
- Practice size (solo, group, enterprise)
- Monthly assessment volume needs
- Required features (HIPAA, API, branding)
- Budget considerations
- Use case (screening, diagnosis, monitoring)

Your goals:
1. Understand the prospect's needs
2. Recommend appropriate tier
3. Highlight relevant features and benefits
4. Address common objections
5. Schedule demo or provide next steps

Be consultative, not pushy. Focus on solving their problems."""

SALES_QUALIFICATION_USER = """Prospect Information:
- Role: {role}
- Practice Type: {practice_type}
- Current Solution: {current_solution}
- Monthly Volume Estimate: {volume}
- Key Requirements: {requirements}
- Budget Range: {budget}

Initial Inquiry: {inquiry}

Please provide:
1. Recommended tier with justification
2. Key features to highlight
3. ROI talking points
4. Potential objections and responses
5. Suggested next steps"""


# ============================================================================
# PROMPT MANAGER CLASS
# ============================================================================

class LLMPromptManager:
    """Manages and generates prompts for JawTrack AI features"""
    
    TEMPLATES = {
        PromptType.CLINICAL_INTERPRETATION: PromptTemplate(
            name="Clinical Interpretation",
            type=PromptType.CLINICAL_INTERPRETATION,
            system_prompt=CLINICAL_INTERPRETATION_SYSTEM,
            user_prompt_template=CLINICAL_INTERPRETATION_USER,
            output_format="markdown",
            temperature=0.2,
            max_tokens=1500
        ),
        PromptType.PATIENT_EDUCATION: PromptTemplate(
            name="Patient Education",
            type=PromptType.PATIENT_EDUCATION,
            system_prompt=PATIENT_EDUCATION_SYSTEM,
            user_prompt_template=PATIENT_EDUCATION_USER,
            output_format="markdown",
            temperature=0.5,
            max_tokens=800
        ),
        PromptType.TREATMENT_SUGGESTION: PromptTemplate(
            name="Treatment Suggestions",
            type=PromptType.TREATMENT_SUGGESTION,
            system_prompt=TREATMENT_SUGGESTION_SYSTEM,
            user_prompt_template=TREATMENT_SUGGESTION_USER,
            output_format="markdown",
            temperature=0.3,
            max_tokens=1200
        ),
        PromptType.CUSTOMER_SUPPORT: PromptTemplate(
            name="Customer Support",
            type=PromptType.CUSTOMER_SUPPORT,
            system_prompt=CUSTOMER_SUPPORT_SYSTEM,
            user_prompt_template=CUSTOMER_SUPPORT_USER,
            output_format="text",
            temperature=0.4,
            max_tokens=500
        ),
        PromptType.SALES_QUALIFICATION: PromptTemplate(
            name="Sales Qualification",
            type=PromptType.SALES_QUALIFICATION,
            system_prompt=SALES_QUALIFICATION_SYSTEM,
            user_prompt_template=SALES_QUALIFICATION_USER,
            output_format="markdown",
            temperature=0.4,
            max_tokens=800
        )
    }
    
    @classmethod
    def get_template(cls, prompt_type: PromptType) -> PromptTemplate:
        """Get a prompt template by type"""
        return cls.TEMPLATES.get(prompt_type)
    
    @classmethod
    def generate_clinical_interpretation_prompt(cls, assessment_data: Dict) -> Dict:
        """Generate clinical interpretation prompt from assessment data"""
        template = cls.TEMPLATES[PromptType.CLINICAL_INTERPRETATION]
        
        # Map assessment data to template variables
        user_prompt = template.user_prompt_template.format(
            max_opening=assessment_data.get('max_opening', 'N/A'),
            movement_range=assessment_data.get('movement_range', 'N/A'),
            avg_lateral=assessment_data.get('avg_lateral', 'N/A'),
            max_velocity=assessment_data.get('max_velocity', 'N/A'),
            avg_velocity=assessment_data.get('avg_velocity', 'N/A'),
            max_acceleration=assessment_data.get('max_acceleration', 'N/A'),
            avg_asymmetry=assessment_data.get('avg_asymmetry', 'N/A'),
            max_asymmetry=assessment_data.get('max_asymmetry', 'N/A'),
            smoothness_index=assessment_data.get('smoothness_index', 'N/A'),
            open_close_separation=assessment_data.get('open_close_separation', 'N/A'),
            path_deviation=assessment_data.get('path_deviation', 'N/A'),
            total_frames=assessment_data.get('total_frames', 'N/A'),
            movement_type=assessment_data.get('movement_type', 'baseline'),
            calibration_status='Calibrated' if assessment_data.get('is_calibrated') else 'Uncalibrated'
        )
        
        return {
            "model": "gpt-4",  # or claude-3, etc.
            "messages": [
                {"role": "system", "content": template.system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": template.temperature,
            "max_tokens": template.max_tokens
        }
    
    @classmethod
    def generate_patient_education_prompt(cls, assessment_data: Dict, clinical_findings: str) -> Dict:
        """Generate patient-friendly explanation prompt"""
        template = cls.TEMPLATES[PromptType.PATIENT_EDUCATION]
        
        # Determine status labels
        max_opening = assessment_data.get('max_opening', 0)
        opening_status = "within normal range" if 40 <= max_opening <= 60 else "outside normal range"
        
        smoothness = assessment_data.get('smoothness_index', 0)
        smoothness_status = "smooth" if smoothness < 15 else "showing some irregularity"
        
        asymmetry = assessment_data.get('avg_asymmetry', 0)
        asymmetry_status = "well balanced" if asymmetry < 0.1 else "showing some imbalance"
        
        overall = "Good" if assessment_data.get('opening_normal') and assessment_data.get('lateral_normal') else "Needs Attention"
        
        user_prompt = template.user_prompt_template.format(
            max_opening=max_opening,
            avg_lateral=assessment_data.get('avg_lateral', 0),
            smoothness_status=smoothness_status,
            asymmetry_status=asymmetry_status,
            overall_status=overall,
            clinical_findings=clinical_findings
        )
        
        return {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": template.system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": template.temperature,
            "max_tokens": template.max_tokens
        }
    
    @classmethod
    def generate_support_prompt(cls, query: str, customer_info: Dict) -> Dict:
        """Generate customer support response prompt"""
        template = cls.TEMPLATES[PromptType.CUSTOMER_SUPPORT]
        
        user_prompt = template.user_prompt_template.format(
            query=query,
            tier=customer_info.get('tier', 'Free'),
            assessments_used=customer_info.get('assessments_used', 0),
            assessments_limit=customer_info.get('assessments_limit', 3),
            account_status=customer_info.get('status', 'Active')
        )
        
        return {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": template.system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": template.temperature,
            "max_tokens": template.max_tokens
        }


# ============================================================================
# API INTEGRATION HELPERS
# ============================================================================

class LLMAPIClient:
    """Client for LLM API calls (OpenAI, Anthropic, etc.)"""
    
    def __init__(self, provider: str = "openai", api_key: str = None):
        self.provider = provider
        self.api_key = api_key or os.getenv(f"{provider.upper()}_API_KEY")
        
    async def generate(self, prompt_config: Dict) -> str:
        """Generate response from LLM"""
        if self.provider == "openai":
            return await self._call_openai(prompt_config)
        elif self.provider == "anthropic":
            return await self._call_anthropic(prompt_config)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    async def _call_openai(self, config: Dict) -> str:
        """Call OpenAI API"""
        import openai
        
        client = openai.AsyncOpenAI(api_key=self.api_key)
        
        response = await client.chat.completions.create(
            model=config.get("model", "gpt-4"),
            messages=config["messages"],
            temperature=config.get("temperature", 0.3),
            max_tokens=config.get("max_tokens", 1000)
        )
        
        return response.choices[0].message.content
    
    async def _call_anthropic(self, config: Dict) -> str:
        """Call Anthropic API"""
        import anthropic
        
        client = anthropic.AsyncAnthropic(api_key=self.api_key)
        
        # Convert messages format
        system = next((m["content"] for m in config["messages"] if m["role"] == "system"), "")
        user_msg = next((m["content"] for m in config["messages"] if m["role"] == "user"), "")
        
        response = await client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=config.get("max_tokens", 1000),
            system=system,
            messages=[{"role": "user", "content": user_msg}]
        )
        
        return response.content[0].text


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

import os

def example_usage():
    """Example of using the LLM prompt system"""
    
    # Sample assessment data (from JawAssessment.get_analysis())
    assessment_data = {
        'max_opening': 42.5,
        'movement_range': 38.2,
        'avg_lateral': 1.8,
        'max_velocity': 85.3,
        'avg_velocity': 45.2,
        'max_acceleration': 120.5,
        'avg_asymmetry': 0.08,
        'max_asymmetry': 0.15,
        'smoothness_index': 12.5,
        'open_close_separation': 2.1,
        'path_deviation': 3.2,
        'total_frames': 450,
        'is_calibrated': True,
        'opening_normal': True,
        'lateral_normal': True
    }
    
    # Generate clinical interpretation prompt
    prompt_config = LLMPromptManager.generate_clinical_interpretation_prompt(assessment_data)
    
    print("=== Clinical Interpretation Prompt ===")
    print(json.dumps(prompt_config, indent=2))
    
    # Generate patient education prompt
    patient_prompt = LLMPromptManager.generate_patient_education_prompt(
        assessment_data, 
        "Mild limitation in jaw opening range"
    )
    
    print("\n=== Patient Education Prompt ===")
    print(json.dumps(patient_prompt, indent=2))
    
    return prompt_config


if __name__ == "__main__":
    example_usage()
