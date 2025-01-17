import gradio as gr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import cv2
import mediapipe as mp
import os
import tempfile

@dataclass
class AssessmentMeasurement:
    timestamp: float
    jaw_opening: float
    lateral_deviation: float
    frame_number: int
    movement_type: str
    quality_score: float

class JawAssessment:
    def __init__(self):
        self.measurements: List[AssessmentMeasurement] = []
        self.current_movement: str = "baseline"
        self.calibration_factor: float = 1.0
        self.assessment_date = datetime.now()
        
    def set_calibration(self, pixel_distance: float, real_distance: float = 20.0):
        """Set calibration using known distance marker"""
        self.calibration_factor = real_distance / pixel_distance
        
    def add_measurement(self, jaw_opening: float, lateral_dev: float, 
                       frame_num: int, quality: float = 1.0):
        """Add a new measurement to the assessment"""
        measurement = AssessmentMeasurement(
            timestamp=datetime.now().timestamp(),
            jaw_opening=jaw_opening * self.calibration_factor,
            lateral_deviation=lateral_dev * self.calibration_factor,
            frame_number=frame_num,
            movement_type=self.current_movement,
            quality_score=quality
        )
        self.measurements.append(measurement)
        
    def set_movement_type(self, movement: str):
        """Set current movement being assessed"""
        self.current_movement = movement
        
    def get_analysis(self) -> Dict:
        """Analyze collected measurements"""
        if not self.measurements:
            return {}
            
        df = pd.DataFrame([asdict(m) for m in self.measurements])
        
        analysis = {
            'max_opening': df['jaw_opening'].max(),
            'avg_lateral': df['lateral_deviation'].mean(),
            'movement_range': df['jaw_opening'].max() - df['jaw_opening'].min(),
            'quality_average': df['quality_score'].mean(),
            'movement_counts': df['movement_type'].value_counts().to_dict(),
            'timestamp': self.assessment_date.isoformat()
        }
        
        return analysis
        
    def plot_movements(self) -> plt.Figure:
        """Generate movement pattern plot"""
        if not self.measurements:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No measurements available', 
                   ha='center', va='center')
            return fig
            
        df = pd.DataFrame([asdict(m) for m in self.measurements])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df['frame_number'], df['jaw_opening'], 
                label='Jaw Opening', color='blue')
        ax.plot(df['frame_number'], df['lateral_deviation'], 
                label='Lateral Deviation', color='red')
        
        ax.set_title('Jaw Movement Patterns')
        ax.set_xlabel('Frame Number')
        ax.set_ylabel('Distance (mm)')
        ax.grid(True)
        ax.legend()
        
        return fig
        
    def generate_report(self) -> str:
        """Generate assessment report"""
        analysis = self.get_analysis()
        
        if not analysis:
            return "No measurements available for report generation."
        
        report = f"""
# Jaw Motion Assessment Report

Date: {self.assessment_date.strftime('%Y-%m-%d %H:%M:%S')}

## Measurements
- Maximum Opening: {analysis.get('max_opening', 0):.1f} mm
- Average Lateral Deviation: {analysis.get('avg_lateral', 0):.1f} mm
- Movement Range: {analysis.get('movement_range', 0):.1f} mm
- Quality Score: {analysis.get('quality_average', 0):.1f}/10

## Movement Analysis
"""
        
        for movement, count in analysis.get('movement_counts', {}).items():
            report += f"- {movement}: {count} frames\n"
            
        return report

def process_video(video_path: str, assessment: JawAssessment) -> Optional[str]:
    """Process video and update assessment with measurements"""
    try:
        if not video_path:
            return None
            
        # Initialize MediaPipe Face Mesh
        mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
            
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Create output video
        output_path = tempfile.mktemp(suffix='.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mp_face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                
                # Get key points
                upper_lip = np.array([landmarks[13].x, landmarks[13].y, landmarks[13].z])
                lower_lip = np.array([landmarks[14].x, landmarks[14].y, landmarks[14].z])
                left_jaw = np.array([landmarks[389].x, landmarks[389].y, landmarks[389].z])
                right_jaw = np.array([landmarks[356].x, landmarks[356].y, landmarks[356].z])
                
                # Calculate measurements
                jaw_opening = np.linalg.norm(upper_lip - lower_lip) * height
                lateral_dev = np.linalg.norm(left_jaw - right_jaw) * width
                
                # Add to assessment
                assessment.add_measurement(jaw_opening, lateral_dev, frame_count)
                
                # Draw landmarks
                h, w = frame.shape[:2]
                for point in [upper_lip, lower_lip, left_jaw, right_jaw]:
                    px = tuple(np.multiply(point[:2], [w, h]).astype(int))
                    cv2.circle(frame, px, 2, (0, 255, 0), -1)
                    
                # Add measurements to frame
                cv2.putText(frame, f"Opening: {jaw_opening:.1f}px", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Lateral: {lateral_dev:.1f}px", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            out.write(frame)
            frame_count += 1
            
        # Cleanup
        cap.release()
        out.release()
        mp_face_mesh.close()
        
        return output_path
        
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        return None

def process_assessment(video_path: str, movement: str) -> Tuple[Optional[str], str, plt.Figure]:
    """Main assessment processing function"""
    assessment = JawAssessment()
    assessment.set_movement_type(movement)
    
    processed_path = process_video(video_path, assessment)
    report = assessment.generate_report()
    plot = assessment.plot_movements()
    
    return processed_path, report, plot

# Create Gradio interface
demo = gr.Interface(
    fn=process_assessment,
    inputs=[
        gr.Video(label="Record Assessment"),
        gr.Radio(
            choices=["baseline", "maximum_opening", "lateral_left", 
                    "lateral_right", "combined"],
            label="Movement Type",
            value="baseline"
        )
    ],
    outputs=[
        gr.Video(label="Processed Recording"),
        gr.Textbox(label="Analysis Report", lines=10),
        gr.Plot(label="Movement Patterns")
    ],
    title="Jaw Motion Assessment",
    description="Upload a video recording to analyze jaw movements."
)

if __name__ == "__main__":
    demo.launch()