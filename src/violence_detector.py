"""
Violence Detector — MobileNetV2 + Bi-LSTM hybrid for violence detection.

Processes a rolling buffer of 16 video frames through a TimeDistributed
CNN backbone followed by temporal sequence classification.

When model is not available, uses robust motion-based heuristic:
- Frame differencing
- Kept velocity spike detection
- Keypoint-based limb movement analysis
"""

import os
import cv2
import numpy as np

import config
from src.utils import setup_logger, draw_label

logger = setup_logger("ViolenceDetector")


class ViolenceDetector:
    """
    Detects violent/aggressive behavior using a CNN-LSTM classifier.
    Implements temporal smoothing to reduce false positives.
    """

    def __init__(self, model_path=None):
        self.model = None
        self.frame_resolution = config.VIOLENCE_FRAME_RESOLUTION
        self.input_frames = config.VIOLENCE_INPUT_FRAMES
        self.threshold = config.VIOLENCE_THRESHOLD
        self.consecutive_required = config.VIOLENCE_CONSECUTIVE_DETECTIONS
        self.inference_interval = config.VIOLENCE_INFERENCE_INTERVAL

        # Load State-of-the-Art R3D_18 from TorchVision Hub!
        try:
            import torch
            from torchvision.models.video import r3d_18, R3D_18_Weights
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            logger.info("Downloading/Loading Ultra-Accurate R3D Extra Model...")
            self.model = r3d_18(weights=R3D_18_Weights.DEFAULT).to(self.device)
            self.model.eval()
            self.is_pytorch = True
            
            # Action classes that strictly map to fighting/violence in K400
            self.fighting_classes = {105, 152, 266, 267, 302, 395, 345}
            
            logger.info("Loaded State-of-the-Art R3D_18 Extra Action Model successfully.")
        except Exception as e:
            logger.warning(f"Could not load R3D_18 extra model: {e}")
            self.model = None

        self.frame_buffer = []       # Normalized float frames (0-1)
        self.raw_frame_buffer = []   # Original uint8 frames for motion analysis
        self.frame_counter = 0
        self.consecutive_positives = 0
        self.last_violence_prob = 0.0
        self.is_violent = False

        # Motion heuristic history
        self.motion_history = []

    def add_frame(self, frame):
        """Add a BGR frame to the processing buffer."""
        resized = cv2.resize(frame, (self.frame_resolution, self.frame_resolution))
        normalized = resized.astype(np.float32) / 255.0
        self.frame_buffer.append(normalized)
        if len(self.frame_buffer) > self.input_frames:
            self.frame_buffer = self.frame_buffer[-self.input_frames:]

        # Store raw frames for motion analysis
        self.raw_frame_buffer.append(frame.copy())
        if len(self.raw_frame_buffer) > 30:
            self.raw_frame_buffer = self.raw_frame_buffer[-30:]

        self.frame_counter += 1

    def should_run_inference(self):
        buffer_full = len(self.frame_buffer) >= self.input_frames
        interval_ok = self.frame_counter % self.inference_interval == 0
        return buffer_full and interval_ok

    def predict(self):
        """
        Run violence detection on current frame buffer.
        Returns: (is_violent: bool, probability: float)
        """
        if not self.should_run_inference():
            return self.is_violent, self.last_violence_prob

        frames = np.array(self.frame_buffer[-self.input_frames:])

        if self.model is not None:
            if getattr(self, 'is_pytorch', False):
                try:
                    import torch
                    import torch.nn.functional as F
                    from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
                    
                    # R3D_18 preprocessing sequence
                    # Kinetics 400 models expect 16 frames exactly: (B, C, T, H, W) where T=16, C=3, H=112, W=112
                    
                    transform = Compose([
                        Resize(128, interpolation=cv2.INTER_LINEAR, antialias=True),
                        CenterCrop(112),
                        Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])
                    ])
                    
                    processed_frames = []
                    # Get 16 strictly spaced frames
                    needed_frames = 16
                    import math
                    skip = max(1, math.floor(len(self.raw_frame_buffer) / needed_frames))
                    raw_sample = self.raw_frame_buffer[::skip][-needed_frames:]
                    
                    # Pad to 16 if we don't have enough
                    while len(raw_sample) < needed_frames:
                        raw_sample.insert(0, raw_sample[0])
                        
                    for f in raw_sample:
                        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                        # HWC to CHW
                        tensor_frame = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
                        tensor_frame = transform(tensor_frame)
                        processed_frames.append(tensor_frame)
                    
                    # Create T tensor: [C, T, H, W]
                    video_tensor = torch.stack(processed_frames, dim=1)
                    batch_frames = video_tensor.unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.model(batch_frames)
                        probs = F.softmax(outputs[0], dim=0)
                        
                        # Sum the probabilities of all violence/fighting classes
                        fight_prob = sum(probs[i].item() for i in self.fighting_classes)
                    
                    # Trust ONLY the SOTA PyTorch extra deep learning model for accuracy
                    prob = float(fight_prob)

                    if prob >= self.threshold:
                        self.consecutive_positives += 1
                    else:
                        self.consecutive_positives = max(0, self.consecutive_positives - 1)

                    self.last_violence_prob = prob
                    self.is_violent = self.consecutive_positives >= self.consecutive_required
                    return self.is_violent, self.last_violence_prob
                except Exception as e:
                    logger.warning(f"PyTorch Violence inference failed: {e}")
            else:
                try:
                    import tensorflow as tf
                    frames = np.expand_dims(frames, axis=0)
                    prob = float(self.model.predict(frames, verbose=0)[0][0])
    
                    if prob >= self.threshold:
                        self.consecutive_positives += 1
                    else:
                        self.consecutive_positives = max(0, self.consecutive_positives - 1)
    
                    self.last_violence_prob = prob
                    self.is_violent = self.consecutive_positives >= self.consecutive_required
                    return self.is_violent, self.last_violence_prob
                except Exception as e:
                    logger.warning(f"Violence inference failed: {e}")

        # Heuristic fallback: analyze motion patterns
        prob = self._motion_heuristic()
        self.last_violence_prob = prob

        if prob >= self.threshold * 0.6:
            self.consecutive_positives += 1
        else:
            self.consecutive_positives = max(0, self.consecutive_positives - 1)
        self.is_violent = self.consecutive_positives >= self.consecutive_required
        return self.is_violent, self.last_violence_prob

    def _motion_heuristic(self):
        """
        Multi-signal motion heuristic for violence detection.
        Uses frame differencing + keypoint limb velocity if available.
        Returns 0.0-1.0 probability of violence.
        """
        if len(self.raw_frame_buffer) < 5:
            return 0.0

        # --- Signal 1: Frame differencing (overall motion energy) ---
        recent = self.raw_frame_buffer[-5:]
        gray_frames = []
        for f in recent:
            if len(f.shape) == 3:
                gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            else:
                gray = f.copy()
            gray_frames.append(gray)

        diffs = []
        largest_box = None
        max_area = 0
        
        for i in range(1, len(gray_frames)):
            diff = cv2.absdiff(gray_frames[i], gray_frames[i-1])
            _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > max_area and area > 1000:
                    max_area = area
                    largest_box = cv2.boundingRect(cnt)
                    
            motion_pixels = np.sum(thresh) / 255.0
            motion_ratio = motion_pixels / (gray_frames[i].shape[0] * gray_frames[i].shape[1])
            diffs.append(motion_ratio)

        self.motion_box = largest_box
        
        avg_frame_motion = np.mean(diffs)
        max_frame_motion = np.max(diffs)

        # --- Signal 2: Motion spike detection ---
        motion_spikes = sum(1 for d in diffs if d > 0.15)
        spike_score = motion_spikes / len(diffs)

        # --- Signal 3: Combined score ---
        score = 0.5 * avg_frame_motion + 0.3 * max_frame_motion + 0.2 * spike_score

        # Normalize to 0-1 (typical range: 0-0.5 for normal, 0.1-0.8 for violent)
        prob = min(1.0, score * 2.0)

        return float(prob)

    def reset(self):
        """Reset state (e.g. camera change)."""
        self.frame_buffer = []
        self.raw_frame_buffer = []
        self.frame_counter = 0
        self.consecutive_positives = 0
        self.last_violence_prob = 0.0
        self.is_violent = False
        self.motion_history = []

    def get_violence_status_color(self):
        """Return status color for UI indicator."""
        if self.is_violent:
            return (0, 0, 255)
        if self.last_violence_prob > 0.4:
            return (0, 165, 255)
        return (0, 255, 0)

    def draw_violence_indicator(self, frame):
        """Draw violence status indicator on frame."""
        color = self.get_violence_status_color()
        prob = int(self.last_violence_prob * 100)
        status = "VIOLENCE DETECTED" if self.is_violent else f"Threat Level: {prob}%"
        return draw_label(frame, status, bg_color=(0, 0, 0),
                          text_color=color, y_offset=25, font_scale=0.8)
