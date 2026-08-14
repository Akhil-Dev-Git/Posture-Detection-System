"""
Train the Action Recognition LSTM model.

Usage:
    python training/train_action_model.py --epochs 50
Requires pre-collected sequences in datasets/pose_sequences/
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

ACTIONS = ["standing", "walking", "sitting_down", "standing_up",
           "raising_hand", "jumping", "waving", "pointing"]
WINDOW = config.ACTION_WINDOW_SIZE  # 30


def load_sequences(data_dir):
    """Load pose sequence .npy files and split into windows."""
    X, y = [], []
    for action_idx, action_name in enumerate(ACTIONS):
        action_dir = os.path.join(data_dir, action_name)
        if not os.path.exists(action_dir):
            print(f"Warning: {action_dir} not found, skipping")
            continue
        for fp in Path(action_dir).glob("*.npy"):
            seq = np.load(fp)
            for start in range(0, len(seq) - WINDOW + 1, WINDOW // 2):
                window = seq[start:start + WINDOW]
                if len(window) == WINDOW:
                    X.append(window)
                    y.append(action_idx)
    return np.array(X), np.array(y)


def build_model(input_dim=132, output_classes=len(ACTIONS)):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dropout, Dense
    from tensorflow.keras.optimizers import Adam

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(WINDOW, input_dim)),
        Dropout(0.3),
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        LSTM(64),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(output_classes, activation='softmax'),
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    return model


def train(data_dir=None, epochs=100, batch_size=32):
    data_dir = data_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "datasets", "pose_sequences"
    )
    print(f"Loading sequences from {data_dir}...")
    X, y = load_sequences(data_dir)

    if len(X) == 0:
        print("No training data found. Run training/collect_pose_data.py first.")
        return

    print(f"Loaded {len(X)} sequences across {len(ACTIONS)} classes")

    # Split
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = build_model()

    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True, monitor='val_accuracy'),
        ModelCheckpoint(
            config.MODEL_DIR + "/action_recognition_lstm.h5", save_best_only=True,
            monitor='val_accuracy'
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
    )

    model.save(config.ACTION_MODEL_PATH)
    print(f"Model saved to {config.ACTION_MODEL_PATH}")

    # Evaluate
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation accuracy: {val_acc:.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()
    train(epochs=args.epochs, batch_size=args.batch)
