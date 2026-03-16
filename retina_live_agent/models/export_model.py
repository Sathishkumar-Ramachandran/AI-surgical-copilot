"""
Export trained Keras model to SavedModel format expected by OCTDetector.

Run after training:
    python models/export_model.py --weights path/to/weights.h5
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build_cnn(num_classes: int = 4, input_shape=(224, 224, 3)):
    """Mirror of the training architecture from the notebook."""
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models
    except ImportError:
        print("TensorFlow not installed.")
        sys.exit(1)

    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape, padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Block 2
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Block 3
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Block 4
        layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),

        # Classifier
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=None, help="Path to .h5 weights file")
    parser.add_argument("--out",     default="models/oct_cnn_model", help="Output path")
    args = parser.parse_args()

    model = build_cnn()

    if args.weights:
        model.load_weights(args.weights)
        print(f"Weights loaded from {args.weights}")

    out = Path(args.out)
    model.save(str(out))
    print(f"Model saved to {out.resolve()}")


if __name__ == "__main__":
    main()
