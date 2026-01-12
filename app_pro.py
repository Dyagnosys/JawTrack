"""
JawTrack Pro - Enhanced App with Business Features
Integrates payment, AI analysis, and assessment system
"""

import gradio as gr
import os
import json
from typing import Optional, Tuple
import asyncio

# Import main app functionality
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import (
    JawAssessment, 
    process_video, 
    CLINICAL_NORMS
)

from business.stripe_integration import (
    StripePaymentManager,
    PricingTier,
    PRICING_PLANS,
    get_pricing_page_data
)

from business.llm_prompts import (
    LLMPromptManager,
    LLMAPIClient,
    PromptType
)


# Initialize services
payment_manager = StripePaymentManager()


class JawTrackPro:
    """Enhanced JawTrack with business features"""
    
    def __init__(self):
        self.payment_manager = StripePaymentManager()
        self.llm_client = LLMAPIClient(
            provider="openai",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.current_customer_id: Optional[str] = None
    
    def process_assessment_with_ai(
        self, 
        video_path: str, 
        movement: str,
        include_ai_analysis: bool = False
    ) -> Tuple[Optional[str], str, any, str]:
        """Process assessment with optional AI clinical interpretation"""
        
        # Check quota if customer is set
        if self.current_customer_id:
            quota = self.payment_manager.check_assessment_quota(self.current_customer_id)
            if not quota["allowed"]:
                return (
                    None, 
                    f"Assessment quota exceeded. You've used {quota['used']}/{quota['limit']} assessments this month.\n\nUpgrade your plan for more assessments.",
                    None,
                    ""
                )
        
        # Process video
        assessment = JawAssessment()
        assessment.set_movement_type(movement)
        
        processed_path = process_video(video_path, assessment)
        report = assessment.generate_report()
        plot = assessment.plot_movements()
        
        # Record usage
        if self.current_customer_id:
            self.payment_manager.record_assessment(self.current_customer_id)
        
        # Generate AI analysis if requested
        ai_analysis = ""
        if include_ai_analysis and processed_path:
            analysis_data = assessment.get_analysis()
            if analysis_data:
                ai_analysis = self._generate_ai_analysis_sync(analysis_data)
        
        return processed_path, report, plot, ai_analysis
    
    def _generate_ai_analysis_sync(self, analysis_data: dict) -> str:
        """Synchronous wrapper for AI analysis"""
        try:
            prompt_config = LLMPromptManager.generate_clinical_interpretation_prompt(analysis_data)
            
            # For demo, return the prompt that would be sent
            # In production, call the actual LLM API
            return self._format_demo_ai_response(analysis_data)
            
        except Exception as e:
            return f"AI analysis unavailable: {str(e)}"
    
    def _format_demo_ai_response(self, data: dict) -> str:
        """Generate demo AI response (replace with actual LLM call in production)"""
        
        findings = []
        recommendations = []
        
        # Analyze opening
        max_opening = data.get('max_opening', 0)
        if max_opening < CLINICAL_NORMS['normal_opening_min']:
            findings.append(f"Limited maximum opening ({max_opening:.1f}mm) - below normal range of 40-60mm")
            recommendations.append("Consider evaluation for disc displacement or muscle restriction")
        elif max_opening > CLINICAL_NORMS['normal_opening_max']:
            findings.append(f"Hypermobility detected ({max_opening:.1f}mm) - above normal range")
            recommendations.append("Monitor for joint instability")
        else:
            findings.append(f"Maximum opening within normal limits ({max_opening:.1f}mm)")
        
        # Analyze asymmetry
        asymmetry = data.get('avg_asymmetry', 0)
        if asymmetry > 0.1:
            findings.append(f"Asymmetric movement pattern detected (index: {asymmetry:.3f})")
            recommendations.append("Evaluate for unilateral TMJ or muscle involvement")
        
        # Analyze trajectory
        smoothness = data.get('smoothness_index', 0)
        if smoothness > 15:
            findings.append(f"Irregular movement trajectory (smoothness index: {smoothness:.2f})")
            recommendations.append("Consider imaging to rule out disc irregularity")
        
        response = """## AI Clinical Interpretation

### Key Findings
"""
        for finding in findings:
            response += f"- {finding}\n"
        
        response += """
### Clinical Considerations
"""
        for rec in recommendations:
            response += f"- {rec}\n"
        
        response += """
### Confidence Level
Moderate - Based on quantitative motion analysis. Clinical correlation recommended.

---
*This analysis is for clinical decision support only. All diagnostic and treatment decisions should be made by qualified healthcare professionals based on comprehensive patient evaluation.*
"""
        
        return response


# Create enhanced Gradio interface
def create_pro_interface():
    """Create the JawTrack Pro Gradio interface"""
    
    jawtrack_pro = JawTrackPro()
    
    def process_with_ai(video, movement, use_ai):
        return jawtrack_pro.process_assessment_with_ai(video, movement, use_ai)
    
    # Pricing data for display
    pricing_data = get_pricing_page_data()
    
    with gr.Blocks(title="JawTrack Pro", theme=gr.themes.Soft()) as demo:
        
        gr.Markdown("""
        # JawTrack Pro
        ### AI-Powered Jaw Motion Assessment System
        
        Professional-grade jaw movement analysis with automatic calibration, 
        clinical metrics, and AI-powered interpretation.
        """)
        
        with gr.Tabs():
            # Assessment Tab
            with gr.Tab("Assessment"):
                with gr.Row():
                    with gr.Column(scale=1):
                        video_input = gr.Video(label="Upload Video or Record")
                        movement_type = gr.Radio(
                            choices=["baseline", "maximum_opening", "lateral_left", 
                                    "lateral_right", "combined"],
                            label="Movement Type",
                            value="baseline"
                        )
                        use_ai = gr.Checkbox(
                            label="Include AI Clinical Interpretation",
                            value=False,
                            info="Available on Clinical and Enterprise plans"
                        )
                        submit_btn = gr.Button("Analyze", variant="primary")
                    
                    with gr.Column(scale=2):
                        video_output = gr.Video(label="Processed Recording")
                        with gr.Tabs():
                            with gr.Tab("Report"):
                                report_output = gr.Textbox(
                                    label="Assessment Report", 
                                    lines=15,
                                    show_copy_button=True
                                )
                            with gr.Tab("AI Analysis"):
                                ai_output = gr.Markdown(label="AI Clinical Interpretation")
                            with gr.Tab("Charts"):
                                plot_output = gr.Plot(label="Movement Patterns")
                
                submit_btn.click(
                    fn=process_with_ai,
                    inputs=[video_input, movement_type, use_ai],
                    outputs=[video_output, report_output, plot_output, ai_output]
                )
            
            # Pricing Tab
            with gr.Tab("Pricing"):
                gr.Markdown("## Choose Your Plan")
                
                with gr.Row():
                    for plan in pricing_data:
                        with gr.Column():
                            popular_badge = "⭐ MOST POPULAR" if plan.get('popular') else ""
                            gr.Markdown(f"""
                            ### {plan['name']} {popular_badge}
                            
                            **${plan['price_monthly']}/month**
                            
                            *${plan['price_yearly']}/year (Save ${plan['yearly_savings']})*
                            
                            **{plan['assessments']} assessments/month**
                            
                            ---
                            
                            {"".join([f"✓ {f}<br>" for f in plan['features']])}
                            """)
            
            # Help Tab
            with gr.Tab("Help"):
                gr.Markdown("""
                ## Getting Started
                
                1. **Record or upload** a video of jaw movements
                2. **Select** the movement type being performed
                3. **Click Analyze** to process the assessment
                4. **Review** the detailed report and visualizations
                
                ## Best Practices
                
                - Ensure good lighting on the face
                - Position camera at eye level, facing directly
                - Record at least 3-5 complete open-close cycles
                - Keep head relatively still during recording
                
                ## Measurement Guide
                
                | Metric | Normal Range | Clinical Significance |
                |--------|-------------|----------------------|
                | Max Opening | 40-60mm | <40mm may indicate restriction |
                | Lateral Deviation | <2mm | >2mm suggests asymmetric function |
                | Asymmetry Index | <0.1 | >0.1 indicates unilateral involvement |
                | Smoothness Index | <15 | Higher values indicate irregular motion |
                
                ## Support
                
                - **Email**: support@jawtrack.app
                - **Documentation**: docs.jawtrack.app
                """)
        
        gr.Markdown("""
        ---
        *JawTrack Pro - Clinical decision support tool. Not intended for diagnosis.*
        """)
    
    return demo


# Main entry point
if __name__ == "__main__":
    demo = create_pro_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
