from pathlib import Path
import copy
import json

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import train_test_split

from .model import RoverPolicy


# ============================================================
# PERFORMANCE SETTINGS
# ============================================================

# Let PyTorch use several CPU threads.
# For your 16 GB Ryzen laptop, 4-8 is reasonable.
torch.set_num_threads(8)

# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "telemetry.csv"
ALT_DATA_PATH = BASE_DIR / "telemetry.csv"

MODEL_PATH = BASE_DIR / "rover_model.pth"
SCALER_PATH = BASE_DIR / "scaler.json"


# ============================================================
# FEATURES
# ============================================================

LIDAR_COLUMNS = [
    f"lidar_{i}"
    for i in range(16)
]

FEATURE_COLUMNS = LIDAR_COLUMNS + [
    "goal_angle",
    "goal_distance",
    "speed",
    "heading",
    "payload_weight",
    "friction",
]

TARGET_COLUMNS = [
    "steering",
    "target_speed",
]


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    target_path = (
        DATA_PATH
        if DATA_PATH.exists()
        else ALT_DATA_PATH
    )

    if not target_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at "
            f"{DATA_PATH} or {ALT_DATA_PATH}. "
            f"Export telemetry from INSINIX simulator first."
        )

    print(
        f"Loading telemetry dataset from: "
        f"{target_path}"
    )

    df = pd.read_csv(target_path)

    required = (
        FEATURE_COLUMNS
        + TARGET_COLUMNS
    )

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Telemetry CSV missing required columns: "
            f"{missing}"
        )

    # Remove invalid rows
    df = df.dropna(
        subset=required
    ).copy()

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna(
        subset=required
    )

    # Clamp targets
    df["steering"] = df[
        "steering"
    ].clip(-1.0, 1.0)

    df["target_speed"] = df[
        "target_speed"
    ].clip(0.0, 1.0)

    X = df[
        FEATURE_COLUMNS
    ].astype(
        np.float32
    ).values

    y = df[
        TARGET_COLUMNS
    ].astype(
        np.float32
    ).values

    print(
        f"Dataset loaded: "
        f"{len(X):,} valid telemetry samples."
    )

    print(
        f"Features: {X.shape}"
    )

    print(
        f"Targets: {y.shape}"
    )

    return X, y


# ============================================================
# SCALER
# ============================================================

def fit_scaler(X):

    mean = X.mean(
        axis=0
    )

    std = X.std(
        axis=0
    )

    # Prevent division by zero
    std[std < 1e-6] = 1.0

    return mean, std


# ============================================================
# TRAIN
# ============================================================

def train_model():

    print("\n==========================================")
    print("       INSINIX ML TRAINING")
    print("==========================================")

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    X, y = load_dataset()

    # --------------------------------------------------------
    # Train / validation split
    # --------------------------------------------------------

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        shuffle=True
    )

    print(
        f"\nTraining samples:   {len(X_train):,}"
    )

    print(
        f"Validation samples: {len(X_val):,}"
    )

    # --------------------------------------------------------
    # Fit scaler ONLY on training data
    # --------------------------------------------------------

    mean, std = fit_scaler(
        X_train
    )

    X_train_norm = (
        X_train - mean
    ) / std

    X_val_norm = (
        X_val - mean
    ) / std

    # --------------------------------------------------------
    # Convert everything to tensors ONCE
    # --------------------------------------------------------

    X_train_tensor = torch.from_numpy(
        X_train_norm
    )

    y_train_tensor = torch.from_numpy(
        y_train
    )

    X_val_tensor = torch.from_numpy(
        X_val_norm
    )

    y_val_tensor = torch.from_numpy(
        y_val
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = RoverPolicy(
        input_size=22
    )

    loss_fn = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-5
    )

    # --------------------------------------------------------
    # Training settings
    # --------------------------------------------------------

    epochs = 40

    best_val_loss = float("inf")
    best_model_state = None

    print(
        f"\nStarting training "
        f"({epochs} epochs)...\n"
    )

    training_start = torch.cuda.Event(
        enable_timing=True
    ) if torch.cuda.is_available() else None

    start_time = __import__(
        "time"
    ).perf_counter()

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(
        1,
        epochs + 1
    ):

        model.train()

        # --------------------------------------------
        # Forward pass
        # --------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        predictions = model(
            X_train_tensor
        )

        train_loss = loss_fn(
            predictions,
            y_train_tensor
        )

        # --------------------------------------------
        # Backprop
        # --------------------------------------------

        train_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

        # --------------------------------------------
        # Validation
        # --------------------------------------------

        model.eval()

        with torch.no_grad():

            val_predictions = model(
                X_val_tensor
            )

            val_loss = loss_fn(
                val_predictions,
                y_val_tensor
            )

        train_loss_value = (
            train_loss.item()
        )

        val_loss_value = (
            val_loss.item()
        )

        # --------------------------------------------
        # Save BEST model
        # --------------------------------------------

        if val_loss_value < best_val_loss:

            best_val_loss = (
                val_loss_value
            )

            best_model_state = copy.deepcopy(
                model.state_dict()
            )

        # --------------------------------------------
        # Progress
        # --------------------------------------------

        if (
            epoch == 1
            or epoch % 5 == 0
            or epoch == epochs
        ):

            print(
                f"Epoch {epoch:03d}/{epochs} | "
                f"Train Loss: {train_loss_value:.6f} | "
                f"Val Loss: {val_loss_value:.6f}"
            )

    elapsed = (
        __import__("time").perf_counter()
        - start_time
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    if best_model_state is not None:

        torch.save(
            best_model_state,
            MODEL_PATH
        )

        print(
            f"\nSaved best model to:"
        )

        print(
            MODEL_PATH
        )

    # ========================================================
    # SAVE SCALER
    # ========================================================

    scaler_data = {
        "features": FEATURE_COLUMNS,
        "mean": mean.tolist(),
        "std": std.tolist(),
    }

    SCALER_PATH.write_text(
        json.dumps(
            scaler_data,
            indent=2
        )
    )

    print(
        f"Saved scaler to:"
    )

    print(
        SCALER_PATH
    )

    # ========================================================
    # FINAL
    # ========================================================

    print("\n==========================================")
    print("       TRAINING COMPLETED")
    print("==========================================")

    print(
        f"Best Validation MSE: "
        f"{best_val_loss:.6f}"
    )

    print(
        f"Training Time: "
        f"{elapsed:.2f} seconds"
    )

    print(
        "==========================================\n"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    train_model()