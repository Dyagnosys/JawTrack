import gradio as gr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Tuple
import cv2
import os
import tempfile
from scipy import signal
from scipy.ndimage import gaussian_filter1d

# MediaPipe imports - using tasks API for newer versions (0.10.30+)
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# Check if legacy solutions API is available
USE_LEGACY_API = False
try:
    if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_mesh'):
        _test = mp.solutions.face_mesh.FaceMesh
        USE_LEGACY_API = True
        print("Using legacy MediaPipe solutions API")
except Exception:
    pass

if not USE_LEGACY_API:
    print("Using new MediaPipe Tasks API")

# MediaPipe Face Mesh Landmark Indices
LANDMARKS = {
    'upper_lip': 13,
    'lower_lip': 14,
    'chin': 152,
    'nose_tip': 1,
    'left_eye_outer': 263,
    'right_eye_outer': 33,
    'left_eye_inner': 362,
    'right_eye_inner': 133,
    'left_jaw': 389,
    'right_jaw': 356,
    'left_mouth_corner': 61,
    'right_mouth_corner': 291,
}

# Clinical Reference Values
CLINICAL_NORMS = {
    'normal_opening_min': 40.0,  # mm
    'normal_opening_max': 60.0,  # mm
    'normal_lateral_max': 2.0,   # mm deviation
    'inter_pupillary_distance': 63.0,  # mm average
    'smoothness_threshold': 0.15,  # trajectory smoothness threshold
}

@dataclass
class AssessmentMeasurement:
    timestamp: float
    jaw_opening: float
    lateral_deviation: float
    frame_number: int
    movement_type: str
    quality_score: float
    # Extended landmarks
    chin_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    nose_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Velocity metrics
    velocity: float = 0.0
    acceleration: float = 0.0
    # Asymmetry metrics
    left_opening: float = 0.0
    right_opening: float = 0.0
    asymmetry_index: float = 0.0

@dataclass 
class TrajectoryAnalysis:
    """Trajectory smoothness and deviation analysis per AAU research"""
    smoothness_index: float = 0.0  # Lower = smoother
    open_close_separation: float = 0.0  # Deviation between opening and closing paths
    path_deviation: float = 0.0  # Deviation from ideal straight path
    condylar_symmetry: float = 0.0  # Left-right symmetry score
    jerk_metric: float = 0.0  # Rate of change of acceleration
    
class AutoCalibration:
    """Automatic calibration using inter-pupillary distance"""
    
    def __init__(self, reference_ipd_mm: float = 63.0):
        self.reference_ipd_mm = reference_ipd_mm
        self.calibration_samples: List[float] = []
        self.calibration_factor: float = 1.0
        self.is_calibrated: bool = False
        
    def add_ipd_sample(self, left_eye: np.ndarray, right_eye: np.ndarray, 
                       frame_width: int, frame_height: int):
        """Add inter-pupillary distance sample for calibration"""
        # Calculate pixel distance between eyes
        left_px = np.array([left_eye[0] * frame_width, left_eye[1] * frame_height])
        right_px = np.array([right_eye[0] * frame_width, right_eye[1] * frame_height])
        ipd_pixels = np.linalg.norm(left_px - right_px)
        
        if ipd_pixels > 20:  # Valid sample threshold
            self.calibration_samples.append(ipd_pixels)
            
    def compute_calibration(self) -> float:
        """Compute calibration factor from collected samples"""
        if len(self.calibration_samples) < 5:
            return 1.0
            
        # Use median for robustness against outliers
        median_ipd_pixels = np.median(self.calibration_samples)
        self.calibration_factor = self.reference_ipd_mm / median_ipd_pixels
        self.is_calibrated = True
        return self.calibration_factor
        
    def to_mm(self, pixel_value: float) -> float:
        """Convert pixel measurement to millimeters"""
        return pixel_value * self.calibration_factor

class TrajectoryAnalyzer:
    """Analyze movement trajectories per AAU research methodology"""
    
    def __init__(self):
        self.opening_trajectory: List[Tuple[float, float]] = []  # (x, y) positions
        self.closing_trajectory: List[Tuple[float, float]] = []
        self.current_phase: str = "unknown"  # opening, closing, static
        self.prev_opening: float = 0.0
        
    def add_point(self, chin_x: float, chin_y: float, jaw_opening: float):
        """Add trajectory point and detect phase"""
        # Detect movement phase
        delta = jaw_opening - self.prev_opening
        
        if abs(delta) < 0.5:  # Static threshold
            self.current_phase = "static"
        elif delta > 0:
            self.current_phase = "opening"
            self.opening_trajectory.append((chin_x, chin_y))
        else:
            self.current_phase = "closing"
            self.closing_trajectory.append((chin_x, chin_y))
            
        self.prev_opening = jaw_opening
        
    def compute_smoothness(self, positions: List[float]) -> float:
        """Calculate trajectory smoothness using spectral arc length"""
        if len(positions) < 10:
            return 0.0
            
        # Compute velocity profile
        velocity = np.diff(positions)
        
        # Spectral analysis for smoothness
        if len(velocity) > 0:
            # Normalized jerk (rate of change of acceleration)
            acceleration = np.diff(velocity)
            if len(acceleration) > 0:
                jerk = np.diff(acceleration)
                # Dimensionless jerk metric - lower is smoother
                duration = len(positions)
                amplitude = np.max(positions) - np.min(positions)
                if amplitude > 0 and duration > 0:
                    smoothness = np.sqrt(0.5 * np.sum(jerk**2) * (duration**5 / amplitude**2))
                    return min(smoothness / 1000, 10.0)  # Normalize to 0-10 scale
        return 0.0
        
    def compute_open_close_separation(self) -> float:
        """Calculate deviation between opening and closing paths"""
        if len(self.opening_trajectory) < 5 or len(self.closing_trajectory) < 5:
            return 0.0
            
        # Resample trajectories to same length
        n_points = min(len(self.opening_trajectory), len(self.closing_trajectory))
        
        opening_x = [p[0] for p in self.opening_trajectory[:n_points]]
        closing_x = [p[0] for p in self.closing_trajectory[:n_points]]
        
        # Calculate mean separation
        separation = np.mean(np.abs(np.array(opening_x) - np.array(closing_x[::-1])))
        return separation
        
    def compute_path_deviation(self, trajectory: List[Tuple[float, float]]) -> float:
        """Calculate deviation from ideal straight vertical path"""
        if len(trajectory) < 3:
            return 0.0
            
        x_coords = [p[0] for p in trajectory]
        # Ideal path would have constant x (straight down)
        x_deviation = np.std(x_coords)
        return x_deviation
        
    def get_analysis(self) -> TrajectoryAnalysis:
        """Get complete trajectory analysis"""
        all_y = [p[1] for p in self.opening_trajectory + self.closing_trajectory]
        
        return TrajectoryAnalysis(
            smoothness_index=self.compute_smoothness(all_y),
            open_close_separation=self.compute_open_close_separation(),
            path_deviation=self.compute_path_deviation(
                self.opening_trajectory + self.closing_trajectory
            ),
            condylar_symmetry=0.0,  # Updated separately with jaw measurements
            jerk_metric=self.compute_smoothness(all_y)
        )

class JawAssessment:
    def __init__(self):
        self.measurements: List[AssessmentMeasurement] = []
        self.current_movement: str = "baseline"
        self.assessment_date = datetime.now()
        self.auto_calibration = AutoCalibration()
        self.trajectory_analyzer = TrajectoryAnalyzer()
        self.fps: float = 30.0
        
        # Velocity tracking
        self.prev_jaw_opening: float = 0.0
        self.prev_velocity: float = 0.0
        self.prev_timestamp: float = 0.0
        
    def set_fps(self, fps: float):
        """Set video FPS for velocity calculations"""
        self.fps = fps if fps > 0 else 30.0
        
    def compute_velocity_metrics(self, jaw_opening: float, timestamp: float) -> Tuple[float, float]:
        """Compute velocity and acceleration"""
        dt = timestamp - self.prev_timestamp if self.prev_timestamp > 0 else 1.0 / self.fps
        
        if dt > 0:
            velocity = (jaw_opening - self.prev_jaw_opening) / dt
            acceleration = (velocity - self.prev_velocity) / dt
        else:
            velocity = 0.0
            acceleration = 0.0
            
        self.prev_jaw_opening = jaw_opening
        self.prev_velocity = velocity
        self.prev_timestamp = timestamp
        
        return velocity, acceleration
        
    def compute_asymmetry_index(self, left_opening: float, right_opening: float) -> float:
        """Compute asymmetry index (0 = symmetric, 1 = fully asymmetric)"""
        total = left_opening + right_opening
        if total > 0:
            asymmetry = abs(left_opening - right_opening) / total
            return asymmetry
        return 0.0
        
    def add_measurement(self, jaw_opening: float, lateral_dev: float, 
                       frame_num: int, quality: float = 1.0,
                       chin_pos: Tuple[float, float, float] = (0, 0, 0),
                       nose_pos: Tuple[float, float, float] = (0, 0, 0),
                       left_opening: float = 0.0,
                       right_opening: float = 0.0):
        """Add a new measurement with extended metrics"""
        
        # Convert to mm using auto-calibration
        jaw_opening_mm = self.auto_calibration.to_mm(jaw_opening)
        lateral_dev_mm = self.auto_calibration.to_mm(lateral_dev)
        left_opening_mm = self.auto_calibration.to_mm(left_opening)
        right_opening_mm = self.auto_calibration.to_mm(right_opening)
        
        # Compute velocity metrics
        timestamp = frame_num / self.fps
        velocity, acceleration = self.compute_velocity_metrics(jaw_opening_mm, timestamp)
        
        # Compute asymmetry
        asymmetry = self.compute_asymmetry_index(left_opening_mm, right_opening_mm)
        
        # Update trajectory analyzer
        self.trajectory_analyzer.add_point(chin_pos[0], chin_pos[1], jaw_opening_mm)
        
        measurement = AssessmentMeasurement(
            timestamp=timestamp,
            jaw_opening=jaw_opening_mm,
            lateral_deviation=lateral_dev_mm,
            frame_number=frame_num,
            movement_type=self.current_movement,
            quality_score=quality,
            chin_position=chin_pos,
            nose_position=nose_pos,
            velocity=velocity,
            acceleration=acceleration,
            left_opening=left_opening_mm,
            right_opening=right_opening_mm,
            asymmetry_index=asymmetry
        )
        self.measurements.append(measurement)
        
    def set_movement_type(self, movement: str):
        """Set current movement being assessed"""
        self.current_movement = movement
        
    def get_analysis(self) -> Dict:
        """Analyze collected measurements with extended metrics"""
        if not self.measurements:
            return {}
            
        df = pd.DataFrame([asdict(m) for m in self.measurements])
        
        # Get trajectory analysis
        traj_analysis = self.trajectory_analyzer.get_analysis()
        
        # Basic metrics
        max_opening = df['jaw_opening'].max()
        min_opening = df['jaw_opening'].min()
        avg_lateral = df['lateral_deviation'].mean()
        
        # Velocity metrics
        max_velocity = df['velocity'].abs().max()
        avg_velocity = df['velocity'].abs().mean()
        max_acceleration = df['acceleration'].abs().max()
        
        # Asymmetry metrics
        avg_asymmetry = df['asymmetry_index'].mean()
        max_asymmetry = df['asymmetry_index'].max()
        
        # Clinical assessment
        opening_normal = CLINICAL_NORMS['normal_opening_min'] <= max_opening <= CLINICAL_NORMS['normal_opening_max']
        lateral_normal = avg_lateral <= CLINICAL_NORMS['normal_lateral_max']
        smoothness_normal = traj_analysis.smoothness_index <= CLINICAL_NORMS['smoothness_threshold'] * 100
        
        analysis = {
            # Basic measurements
            'max_opening': max_opening,
            'min_opening': min_opening,
            'avg_lateral': avg_lateral,
            'movement_range': max_opening - min_opening,
            'quality_average': df['quality_score'].mean(),
            
            # Velocity metrics
            'max_velocity': max_velocity,
            'avg_velocity': avg_velocity,
            'max_acceleration': max_acceleration,
            
            # Asymmetry metrics
            'avg_asymmetry': avg_asymmetry,
            'max_asymmetry': max_asymmetry,
            
            # Trajectory metrics (AAU research)
            'smoothness_index': traj_analysis.smoothness_index,
            'open_close_separation': traj_analysis.open_close_separation,
            'path_deviation': traj_analysis.path_deviation,
            
            # Calibration info
            'calibration_factor': self.auto_calibration.calibration_factor,
            'is_calibrated': self.auto_calibration.is_calibrated,
            
            # Clinical assessment
            'opening_normal': opening_normal,
            'lateral_normal': lateral_normal,
            'smoothness_normal': smoothness_normal,
            
            # Metadata
            'movement_counts': df['movement_type'].value_counts().to_dict(),
            'timestamp': self.assessment_date.isoformat(),
            'total_frames': len(df)
        }
        
        return analysis
        
    def plot_movements(self) -> plt.Figure:
        """Generate comprehensive movement pattern plots"""
        if not self.measurements:
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.text(0.5, 0.5, 'No measurements available', 
                   ha='center', va='center')
            return fig
            
        df = pd.DataFrame([asdict(m) for m in self.measurements])
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Jaw Opening and Lateral Deviation
        ax1 = axes[0, 0]
        ax1.plot(df['frame_number'], df['jaw_opening'], 
                label='Jaw Opening', color='blue', linewidth=2)
        ax1.axhline(y=CLINICAL_NORMS['normal_opening_min'], color='green', 
                   linestyle='--', alpha=0.7, label='Normal Range')
        ax1.axhline(y=CLINICAL_NORMS['normal_opening_max'], color='green', 
                   linestyle='--', alpha=0.7)
        ax1.fill_between(df['frame_number'], CLINICAL_NORMS['normal_opening_min'],
                        CLINICAL_NORMS['normal_opening_max'], alpha=0.1, color='green')
        ax1.set_title('Jaw Opening Over Time')
        ax1.set_xlabel('Frame Number')
        ax1.set_ylabel('Distance (mm)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Velocity Profile
        ax2 = axes[0, 1]
        ax2.plot(df['frame_number'], df['velocity'], 
                label='Velocity', color='orange', linewidth=2)
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax2.set_title('Movement Velocity')
        ax2.set_xlabel('Frame Number')
        ax2.set_ylabel('Velocity (mm/s)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Plot 3: Asymmetry Index
        ax3 = axes[1, 0]
        ax3.plot(df['frame_number'], df['asymmetry_index'], 
                label='Asymmetry Index', color='red', linewidth=2)
        ax3.axhline(y=0.1, color='orange', linestyle='--', 
                   alpha=0.7, label='Threshold')
        ax3.set_title('Movement Asymmetry')
        ax3.set_xlabel('Frame Number')
        ax3.set_ylabel('Asymmetry Index (0-1)')
        ax3.set_ylim(0, 1)
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Plot 4: Lateral Deviation Path
        ax4 = axes[1, 1]
        ax4.plot(df['lateral_deviation'], df['jaw_opening'], 
                 color='purple', linewidth=2, alpha=0.7)
        ax4.scatter(df['lateral_deviation'].iloc[0], df['jaw_opening'].iloc[0],
                   color='green', s=100, zorder=5, label='Start')
        ax4.scatter(df['lateral_deviation'].iloc[-1], df['jaw_opening'].iloc[-1],
                   color='red', s=100, zorder=5, label='End')
        ax4.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax4.set_title('Open-Close Deviation Path')
        ax4.set_xlabel('Lateral Deviation (mm)')
        ax4.set_ylabel('Jaw Opening (mm)')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()
        return fig
        
    def generate_report(self) -> str:
        """Generate comprehensive clinical assessment report"""
        analysis = self.get_analysis()
        
        if not analysis:
            return "No measurements available for report generation."
        
        # Clinical status indicators
        def status_icon(is_normal: bool) -> str:
            return "[NORMAL]" if is_normal else "[ATTENTION]"
            
        report = f"""
# Jaw Motion Assessment Report

Date: {self.assessment_date.strftime('%Y-%m-%d %H:%M:%S')}
Calibration: {'Auto-calibrated (IPD)' if analysis.get('is_calibrated') else 'Uncalibrated'}
Calibration Factor: {analysis.get('calibration_factor', 1.0):.4f} mm/px

---

## Primary Measurements

| Metric | Value | Status |
|--------|-------|--------|
| Maximum Opening | {analysis.get('max_opening', 0):.1f} mm | {status_icon(analysis.get('opening_normal', False))} |
| Movement Range | {analysis.get('movement_range', 0):.1f} mm | - |
| Avg Lateral Deviation | {analysis.get('avg_lateral', 0):.2f} mm | {status_icon(analysis.get('lateral_normal', False))} |

### Reference: Normal opening range is {CLINICAL_NORMS['normal_opening_min']}-{CLINICAL_NORMS['normal_opening_max']} mm

---

## Velocity Analysis

| Metric | Value |
|--------|-------|
| Max Velocity | {analysis.get('max_velocity', 0):.1f} mm/s |
| Avg Velocity | {analysis.get('avg_velocity', 0):.1f} mm/s |
| Max Acceleration | {analysis.get('max_acceleration', 0):.1f} mm/s² |

---

## Asymmetry Analysis

| Metric | Value | Status |
|--------|-------|--------|
| Average Asymmetry Index | {analysis.get('avg_asymmetry', 0):.3f} | {'[NORMAL]' if analysis.get('avg_asymmetry', 0) < 0.1 else '[ATTENTION]'} |
| Maximum Asymmetry | {analysis.get('max_asymmetry', 0):.3f} | - |

### Note: Asymmetry Index > 0.1 may indicate unilateral TMD involvement

---

## Trajectory Smoothness Analysis (AAU Methodology)

| Metric | Value | Status |
|--------|-------|--------|
| Smoothness Index | {analysis.get('smoothness_index', 0):.2f} | {status_icon(analysis.get('smoothness_normal', False))} |
| Open-Close Separation | {analysis.get('open_close_separation', 0):.2f} mm | - |
| Path Deviation | {analysis.get('path_deviation', 0):.2f} mm | - |

### Note: Higher smoothness index indicates jerky/irregular movement

---

## Recording Statistics

- Total Frames Analyzed: {analysis.get('total_frames', 0)}
- Quality Score: {analysis.get('quality_average', 0):.1f}/10
"""
        
        # Movement type breakdown
        report += "\n## Movement Type Distribution\n\n"
        for movement, count in analysis.get('movement_counts', {}).items():
            report += f"- {movement}: {count} frames\n"
            
        # Clinical interpretation
        report += "\n---\n\n## Clinical Interpretation\n\n"
        
        issues = []
        if not analysis.get('opening_normal', True):
            if analysis.get('max_opening', 0) < CLINICAL_NORMS['normal_opening_min']:
                issues.append("Limited mouth opening (possible TMD restriction)")
            else:
                issues.append("Hypermobility detected")
                
        if not analysis.get('lateral_normal', True):
            issues.append("Significant lateral deviation during opening")
            
        if analysis.get('avg_asymmetry', 0) > 0.1:
            issues.append("Asymmetric jaw movement pattern")
            
        if not analysis.get('smoothness_normal', True):
            issues.append("Irregular/jerky movement trajectory")
            
        if issues:
            report += "**Findings requiring attention:**\n\n"
            for issue in issues:
                report += f"- {issue}\n"
        else:
            report += "**All parameters within normal limits.**\n"
            
        return report

def get_landmark_point(landmarks, idx: int) -> np.ndarray:
    """Extract landmark as numpy array"""
    lm = landmarks[idx]
    return np.array([lm.x, lm.y, lm.z])

def get_landmark_point_from_normalized(landmark) -> np.ndarray:
    """Extract landmark from NormalizedLandmark (new API)"""
    return np.array([landmark.x, landmark.y, landmark.z])

class FaceMeshProcessor:
    """Wrapper for MediaPipe Face Mesh that handles both old and new API"""
    
    def __init__(self):
        self.detector = None
        self.use_legacy = USE_LEGACY_API
        self._init_detector()
        
    def _init_detector(self):
        if self.use_legacy:
            # Use legacy solutions API
            self.detector = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        else:
            # Use new tasks API
            base_options = mp_python.BaseOptions(
                model_asset_path=self._get_model_path()
            )
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.detector = vision.FaceLandmarker.create_from_options(options)
            
    def _get_model_path(self) -> str:
        """Get the face landmarker model path"""
        import urllib.request
        model_path = os.path.join(tempfile.gettempdir(), 'face_landmarker.task')
        if not os.path.exists(model_path):
            url = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
            urllib.request.urlretrieve(url, model_path)
        return model_path
    
    def process(self, rgb_frame: np.ndarray) -> Optional[List]:
        """Process frame and return landmarks"""
        if self.use_legacy:
            results = self.detector.process(rgb_frame)
            if results.multi_face_landmarks:
                return results.multi_face_landmarks[0].landmark
            return None
        else:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = self.detector.detect(mp_image)
            if results.face_landmarks:
                return results.face_landmarks[0]
            return None
    
    def close(self):
        if self.use_legacy and hasattr(self.detector, 'close'):
            self.detector.close()

def process_video(video_path: str, assessment: JawAssessment) -> Optional[str]:
    """Process video with extended landmark tracking and auto-calibration"""
    try:
        if not video_path:
            return None
            
        # Initialize MediaPipe Face Mesh
        face_mesh = FaceMeshProcessor()
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
            
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        assessment.set_fps(fps if fps > 0 else 30.0)
        
        # Create output video
        output_path = tempfile.mktemp(suffix='.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, int(fps) if fps > 0 else 30, (width, height))
        
        frame_count = 0
        calibration_frames = 30  # Collect calibration data for first 30 frames
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            landmarks = face_mesh.process(rgb_frame)
            
            if landmarks is not None:
                
                # Get extended landmark points
                upper_lip = get_landmark_point(landmarks, LANDMARKS['upper_lip'])
                lower_lip = get_landmark_point(landmarks, LANDMARKS['lower_lip'])
                chin = get_landmark_point(landmarks, LANDMARKS['chin'])
                nose_tip = get_landmark_point(landmarks, LANDMARKS['nose_tip'])
                left_eye_outer = get_landmark_point(landmarks, LANDMARKS['left_eye_outer'])
                right_eye_outer = get_landmark_point(landmarks, LANDMARKS['right_eye_outer'])
                left_jaw = get_landmark_point(landmarks, LANDMARKS['left_jaw'])
                right_jaw = get_landmark_point(landmarks, LANDMARKS['right_jaw'])
                left_mouth = get_landmark_point(landmarks, LANDMARKS['left_mouth_corner'])
                right_mouth = get_landmark_point(landmarks, LANDMARKS['right_mouth_corner'])
                
                # Auto-calibration using inter-pupillary distance
                if frame_count < calibration_frames:
                    assessment.auto_calibration.add_ipd_sample(
                        left_eye_outer, right_eye_outer, width, height
                    )
                elif frame_count == calibration_frames:
                    assessment.auto_calibration.compute_calibration()
                
                # Calculate primary measurements (in pixels)
                # Jaw opening: distance from upper lip to chin (more accurate than lip-to-lip)
                jaw_opening_px = np.linalg.norm(
                    np.array([upper_lip[0] * width, upper_lip[1] * height]) - 
                    np.array([chin[0] * width, chin[1] * height])
                )
                
                # Alternative: lip-to-lip opening
                lip_opening_px = np.linalg.norm(
                    np.array([upper_lip[0] * width, upper_lip[1] * height]) - 
                    np.array([lower_lip[0] * width, lower_lip[1] * height])
                )
                
                # Calculate lateral deviation (chin position relative to nose midline)
                nose_x = nose_tip[0] * width
                chin_x = chin[0] * width
                lateral_dev_px = abs(chin_x - nose_x)
                
                # Calculate left/right asymmetry
                # Distance from upper lip to left/right mouth corners
                left_opening_px = np.linalg.norm(
                    np.array([upper_lip[0] * width, upper_lip[1] * height]) - 
                    np.array([left_mouth[0] * width, left_mouth[1] * height])
                )
                right_opening_px = np.linalg.norm(
                    np.array([upper_lip[0] * width, upper_lip[1] * height]) - 
                    np.array([right_mouth[0] * width, right_mouth[1] * height])
                )
                
                # Store chin and nose positions for trajectory analysis
                chin_pos = (chin[0] * width, chin[1] * height, chin[2])
                nose_pos = (nose_tip[0] * width, nose_tip[1] * height, nose_tip[2])
                
                # Quality score - simplified for new API compatibility
                # New Tasks API doesn't have visibility scores, use default quality
                quality = 8.0  # Default good quality if face detected
                
                # Add measurement with all extended data
                assessment.add_measurement(
                    jaw_opening=lip_opening_px,
                    lateral_dev=lateral_dev_px,
                    frame_num=frame_count,
                    quality=quality,
                    chin_pos=chin_pos,
                    nose_pos=nose_pos,
                    left_opening=left_opening_px,
                    right_opening=right_opening_px
                )
                
                # Draw all tracked landmarks
                h, w = frame.shape[:2]
                landmark_points = {
                    'Upper Lip': (upper_lip, (0, 255, 0)),
                    'Lower Lip': (lower_lip, (0, 255, 0)),
                    'Chin': (chin, (255, 0, 0)),
                    'Nose': (nose_tip, (255, 255, 0)),
                    'L Eye': (left_eye_outer, (0, 255, 255)),
                    'R Eye': (right_eye_outer, (0, 255, 255)),
                }
                
                for name, (point, color) in landmark_points.items():
                    px = (int(point[0] * w), int(point[1] * h))
                    cv2.circle(frame, px, 3, color, -1)
                
                # Draw reference line from nose to chin
                nose_px = (int(nose_tip[0] * w), int(nose_tip[1] * h))
                chin_px = (int(chin[0] * w), int(chin[1] * h))
                cv2.line(frame, nose_px, chin_px, (255, 0, 255), 1)
                
                # Get calibrated measurements for display
                cal = assessment.auto_calibration
                opening_mm = cal.to_mm(lip_opening_px)
                lateral_mm = cal.to_mm(lateral_dev_px)
                
                # Calculate current asymmetry
                asym = assessment.compute_asymmetry_index(
                    cal.to_mm(left_opening_px), cal.to_mm(right_opening_px)
                )
                
                # Display measurements on frame
                y_offset = 30
                metrics = [
                    f"Opening: {opening_mm:.1f}mm",
                    f"Lateral Dev: {lateral_mm:.1f}mm",
                    f"Asymmetry: {asym:.2f}",
                    f"Cal: {'OK' if cal.is_calibrated else 'Pending...'}"
                ]
                
                for i, text in enumerate(metrics):
                    color = (0, 255, 0) if cal.is_calibrated else (0, 255, 255)
                    cv2.putText(frame, text, (10, y_offset + i * 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            out.write(frame)
            frame_count += 1
            
        # Cleanup
        cap.release()
        out.release()
        face_mesh.close()
        
        return output_path
        
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        import traceback
        traceback.print_exc()
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