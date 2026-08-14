"""
Train the Violence Detection CNN-LSTM model with GPU acceleration and Memory Optimization.

Architecture: TimeDistributed MobileNetV2 -> Bi-LSTM -> Sigmoid
"""

import os
import sys
import numpy as np
import random
import math
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.utils import Sequence

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import cv2

# --- GPU Configuration ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✓ Found {len(gpus)} GPU(s). Memory growth enabled.")
    except RuntimeError as e:
        print(e)
else:
    print("⚠ No GPU found. Training will be slow on CPU.")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


class ViolenceDataGenerator(Sequence):
    """Memory-efficient data generator for video sequences."""
    def __init__(self, video_paths, labels, batch_size, input_frames, 
                 resize_dim=(224, 224), shuffle=True):
        self.video_paths = list(video_paths)
        self.labels = list(labels)
        self.batch_size = batch_size
        self.input_frames = input_frames
        self.resize_dim = resize_dim
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.video_paths) / self.batch_size)

    def __getitem__(self, idx):
        batch_paths = self.video_paths[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_labels = self.labels[idx * self.batch_size : (idx + 1) * self.batch_size]
        
        X, y = [], []
        for path, label in zip(batch_paths, batch_labels):
            frames = self._load_video(path)
            if len(frames) >= self.input_frames:
                # Extract the middle window
                mid = len(frames) // 2
                start = max(0, mid - self.input_frames // 2)
                window = frames[start:start + self.input_frames]
                if len(window) == self.input_frames:
                    X.append(window)
                    y.append(label)
        
        if not X:
            return self.__getitem__(random.randint(0, self.__len__() - 1))
            
        return np.array(X), np.array(y)

    def _load_video(self, path):
        cap = cv2.VideoCapture(str(path))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.resize(frame, self.resize_dim)
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)
        cap.release()
        return frames

    def on_epoch_end(self):
        if self.shuffle:
            combined = list(zip(self.video_paths, self.labels))
            random.shuffle(combined)
            if combined:
                self.video_paths, self.labels = zip(*combined)


def build_model(input_frames=config.VIOLENCE_INPUT_FRAMES):
    """Build MobileNetV2 + Bi-LSTM violence detection architecture."""
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        Input, TimeDistributed, LSTM, Bidirectional,
        Dropout, Dense, GlobalAveragePooling2D
    )
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.optimizers import Adam

    frames = Input(shape=(input_frames, 224, 224, 3))

    # Pre-trained MobileNetV2 backbone
    base = MobileNetV2(weights='imagenet', include_top=False,
                       input_shape=(224, 224, 3))
    base.trainable = False  # Freeze for transfer learning

    time_distributed = TimeDistributed(base)(frames)
    # Pool to (1280,)
    pooled = TimeDistributed(GlobalAveragePooling2D())(time_distributed)

    # Bi-LSTM temporal layer
    lstm = Bidirectional(LSTM(64, return_sequences=True))(pooled)
    lstm = Dropout(0.5)(lstm)
    lstm = Bidirectional(LSTM(32))(lstm)

    output = Dense(32, activation='relu')(lstm)
    output = Dropout(0.3)(output)
    output = Dense(1, activation='sigmoid')(output)

    model = Model(inputs=frames, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    return model


def get_dataset_files(dataset_path, max_videos=2000):
    """Get list of video paths and labels."""
    video_paths, labels = [], []
    for label, sub in enumerate(["violent", "non_violent"]):
        sub_dir = os.path.join(dataset_path, sub)
        if not os.path.exists(sub_dir):
            continue

        files = list(Path(sub_dir).glob("*.avi")) + list(Path(sub_dir).glob("*.mp4"))
        random.shuffle(files)
        files = files[:max_videos]
        
        video_paths.extend(files)
        labels.extend([1.0 if sub == "violent" else 0.0] * len(files))
        
    return video_paths, labels


def train(dataset_path=None, epochs=30, batch_size=8, max_videos=2000):
    if dataset_path is None:
        dataset_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "datasets", "violence"
        )

    print(f"Searching dataset in {dataset_path}...")
    video_paths, labels = get_dataset_files(dataset_path, max_videos=max_videos)

    if not video_paths:
        print(f"No videos found in {dataset_path}")
        return

    print(f"Total videos: {len(video_paths)} (Violent: {sum(labels)}, Non-Violent: {len(labels) - sum(labels)})")

    # Split
    split_idx = int(0.8 * len(video_paths))
    combined = list(zip(video_paths, labels))
    random.shuffle(combined)
    train_data = combined[:split_idx]
    val_data = combined[split_idx:]

    train_paths, train_labels = zip(*train_data)
    val_paths, val_labels = zip(*val_data)

    train_gen = ViolenceDataGenerator(train_paths, train_labels, batch_size, config.VIOLENCE_INPUT_FRAMES)
    val_gen = ViolenceDataGenerator(val_paths, val_labels, batch_size, config.VIOLENCE_INPUT_FRAMES)

    model = build_model()

    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True, monitor='val_accuracy'),
        ModelCheckpoint(config.VIOLENCE_MODEL_PATH, save_best_only=True, monitor='val_accuracy'),
        ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-6),
    ]

    print(f"Starting training: {len(train_paths)} training, {len(val_paths)} validation")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
    )

    model.save(config.VIOLENCE_MODEL_PATH)
    print(f"Model saved to {config.VIOLENCE_MODEL_PATH}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--max_videos", type=int, default=3000)
    args = parser.parse_args()
    train(dataset_path=args.data, epochs=args.epochs, batch_size=args.batch, max_videos=args.max_videos)
