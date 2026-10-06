import argparse
import os
from collections import Counter

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input


IMG_SIZE = (224, 224)
NUM_CLASSES = 5

CLASS_NAMES = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR",
]


def build_datasets(data_dir, batch_size):

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    train_ds = keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=IMG_SIZE,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=True,
        seed=42,
    )

    val_ds = keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=IMG_SIZE,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False,
    )

    print("\nDetected classes:")
    print(train_ds.class_names)

    augment = keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.08),
        layers.RandomBrightness(0.1),
        layers.RandomContrast(0.1),
    ])

    def prep(ds, training=False):

        ds = ds.map(
            lambda x, y: (preprocess_input(x), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        if training:
            ds = ds.map(
                lambda x, y: (augment(x, training=True), y),
                num_parallel_calls=tf.data.AUTOTUNE,
            )

        return ds.prefetch(tf.data.AUTOTUNE)

    return prep(train_ds, True), prep(val_ds, False)


def calculate_class_weights(data_dir):

    train_dir = os.path.join(data_dir, "train")

    counts = []

    for class_index in range(NUM_CLASSES):

        folder = os.path.join(
            train_dir,
            f"{class_index}_{CLASS_NAMES[class_index].replace(' ', '_')}"
        )

        # Handle actual folder name used by your dataset
        possible_folders = [
            folder,
            os.path.join(train_dir, "0_No_DR"),
            os.path.join(train_dir, "1_Mild"),
            os.path.join(train_dir, "2_Moderate"),
            os.path.join(train_dir, "3_Severe"),
            os.path.join(train_dir, "4_Proliferative_DR"),
        ]

        if class_index == 0:
            folder = possible_folders[1]
        elif class_index == 1:
            folder = possible_folders[2]
        elif class_index == 2:
            folder = possible_folders[3]
        elif class_index == 3:
            folder = possible_folders[4]
        else:
            folder = possible_folders[5]

        count = 0

        if os.path.exists(folder):
            for filename in os.listdir(folder):
                if filename.lower().endswith(
                    (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
                ):
                    count += 1

        counts.append(count)

    total = sum(counts)

    class_weights = {}

    for class_index, count in enumerate(counts):

        if count > 0:
            class_weights[class_index] = total / (
                NUM_CLASSES * count
            )
        else:
            class_weights[class_index] = 1.0

    print("\nTraining class distribution:")

    for i, count in enumerate(counts):
        print(
            f"{i} - {CLASS_NAMES[i]}: "
            f"{count} images"
        )

    print("\nClass weights:")

    for i in range(NUM_CLASSES):
        print(
            f"{i} - {CLASS_NAMES[i]}: "
            f"{class_weights[i]:.4f}"
        )

    return class_weights


def build_model():

    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMG_SIZE, 3),
        pooling="avg",
    )

    base_model.trainable = False

    inputs = keras.Input(
        shape=(*IMG_SIZE, 3)
    )

    x = base_model(
        inputs,
        training=False
    )

    x = layers.Dropout(0.3)(x)

    x = layers.Dense(
        128,
        activation="relu"
    )(x)

    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )(x)

    model = keras.Model(
        inputs,
        outputs
    )

    return model, base_model


def create_callbacks(
    checkpoint_dir,
    phase_name
):

    os.makedirs(
        checkpoint_dir,
        exist_ok=True
    )

    best_model_path = os.path.join(
        checkpoint_dir,
        f"{phase_name}_weighted_best.keras"
    )

    latest_model_path = os.path.join(
        checkpoint_dir,
        f"{phase_name}_weighted_latest.keras"
    )

    return [

        keras.callbacks.ModelCheckpoint(
            filepath=best_model_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),

        keras.callbacks.ModelCheckpoint(
            filepath=latest_model_path,
            monitor="val_accuracy",
            save_best_only=False,
            save_weights_only=False,
            verbose=0,
        ),

        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),

        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            verbose=1,
        ),
    ]


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        type=str,
        default="./dataset"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=15
    )

    parser.add_argument(
        "--fine_tune_epochs",
        type=int,
        default=5
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32
    )

    parser.add_argument(
        "--output",
        type=str,
        default="./model/retina_model_weighted.keras"
    )

    args = parser.parse_args()

    os.makedirs(
        os.path.dirname(args.output),
        exist_ok=True
    )

    checkpoint_dir = os.path.join(
        os.path.dirname(args.output),
        "checkpoints"
    )

    os.makedirs(
        checkpoint_dir,
        exist_ok=True
    )

    print("\n==============================================")
    print("RETINAGUARD AI - CLASS WEIGHTED TRAINING")
    print("==============================================")

    train_ds, val_ds = build_datasets(
        args.data_dir,
        args.batch_size
    )

    # Calculate weights for imbalanced classes
    class_weights = calculate_class_weights(
        args.data_dir
    )

    model, base_model = build_model()

    # ---------------------------------------------------------
    # PHASE 1
    # ---------------------------------------------------------

    print("\n==============================================")
    print("PHASE 1 - CLASS WEIGHTED TRAINING")
    print("==============================================")

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=1e-3
        ),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
        ],
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=create_callbacks(
            checkpoint_dir,
            "phase1"
        ),
    )

    # ---------------------------------------------------------
    # PHASE 2
    # ---------------------------------------------------------

    print("\n==============================================")
    print("PHASE 2 - FINE TUNING")
    print("==============================================")

    base_model.trainable = True

    for layer in base_model.layers[:-20]:
        layer.trainable = False

    # Keep BatchNormalization frozen
    for layer in base_model.layers:
        if isinstance(
            layer,
            layers.BatchNormalization
        ):
            layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=1e-5
        ),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
        ],
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.fine_tune_epochs,
        class_weight=class_weights,
        callbacks=create_callbacks(
            checkpoint_dir,
            "phase2"
        ),
    )

    # ---------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------

    model.save(args.output)

    print("\n==============================================")
    print("WEIGHTED TRAINING COMPLETED")
    print("==============================================")

    print("\nModel saved to:")
    print(os.path.abspath(args.output))

    print("\nCheckpoints saved to:")
    print(os.path.abspath(checkpoint_dir))


if __name__ == "__main__":
    main()