import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    cohen_kappa_score,
)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

CLASS_NAMES = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR",
]

MODEL_PATH = os.path.join(
    "model",
    "retina_model_weighted.keras"
)

TEST_DIR = os.path.join(
    "dataset",
    "test"
)


print("=" * 60)
print("RETINAGUARD AI - MODEL EVALUATION")
print("=" * 60)

print("\nLoading model:")
print(MODEL_PATH)

model = keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


# ---------------------------------------------------------
# Load test dataset
# ---------------------------------------------------------

test_ds = keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False,
)

print("\nDetected classes:")
print(test_ds.class_names)


# ---------------------------------------------------------
# Predictions
# ---------------------------------------------------------

print("\nRunning predictions on test dataset...")

y_true = []
y_pred = []

for images, labels in test_ds:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes)


y_true = np.array(y_true)
y_pred = np.array(y_pred)


# ---------------------------------------------------------
# Accuracy
# ---------------------------------------------------------

accuracy = accuracy_score(
    y_true,
    y_pred
)

print("\n" + "=" * 60)
print("OVERALL RESULTS")
print("=" * 60)

print(
    f"\nTest Accuracy: {accuracy * 100:.2f}%"
)


# ---------------------------------------------------------
# Classification Report
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

report = classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0,
)

print(report)


# ---------------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------------

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print()

print("Predicted →")
print()

print(
    f"{'Actual':<22}"
    + "".join(
        f"{name[:10]:>12}"
        for name in CLASS_NAMES
    )
)

for i, row in enumerate(cm):

    print(
        f"{CLASS_NAMES[i]:<22}"
        + "".join(
            f"{value:>12}"
            for value in row
        )
    )


# ---------------------------------------------------------
# Quadratic Weighted Kappa
# ---------------------------------------------------------

qwk = cohen_kappa_score(
    y_true,
    y_pred,
    weights="quadratic"
)

print("\n" + "=" * 60)
print("SEVERITY AGREEMENT")
print("=" * 60)

print(
    f"\nQuadratic Weighted Kappa: {qwk:.4f}"
)


# ---------------------------------------------------------
# Number of test images
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("TEST DATASET")
print("=" * 60)

print(
    f"\nTotal test images: {len(y_true)}"
)

print("\nEvaluation completed successfully.")
