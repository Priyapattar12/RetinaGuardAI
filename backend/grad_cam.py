import tensorflow as tf
import numpy as np
import cv2


def make_gradcam_heatmap(img_array, model, pred_index=None):
    """
    Generate Grad-CAM heatmap.

    Parameters
    ----------
    img_array : numpy.ndarray
        Preprocessed image batch, shape (1, 224, 224, 3).

    model : tf.keras.Model
        Complete RetinaGuard EfficientNetB0 classification model.

    pred_index : int
        Predicted class index.
    """

    try:
        # ---------------------------------------------------------
        # 1. Get nested EfficientNetB0 backbone
        # ---------------------------------------------------------
        backbone = model.get_layer("efficientnetb0")

        # ---------------------------------------------------------
        # 2. Get final convolutional feature layer
        # ---------------------------------------------------------
        target_layer = backbone.get_layer("top_activation")

        print(
            "[Grad-CAM] Backbone:",
            backbone.name
        )

        print(
            "[Grad-CAM] Target layer:",
            target_layer.name
        )

        print(
            "[Grad-CAM] Target shape:",
            target_layer.output.shape
        )

        # ---------------------------------------------------------
        # 3. Create feature model
        # ---------------------------------------------------------
        feature_model = tf.keras.Model(
            inputs=backbone.input,
            outputs=[
                target_layer.output,
                backbone.output
            ]
        )

        # ---------------------------------------------------------
        # 4. Forward pass
        # ---------------------------------------------------------
        with tf.GradientTape() as tape:

            conv_outputs, backbone_output = feature_model(
                img_array,
                training=False
            )

            # Use the same classification head as the
            # original trained model.

            x = backbone_output

            x = model.get_layer("dropout")(
                x,
                training=False
            )

            x = model.get_layer("dense")(x)

            x = model.get_layer("dropout_1")(
                x,
                training=False
            )

            predictions = model.get_layer("dense_1")(x)

            # -----------------------------------------------------
            # Determine prediction class
            # -----------------------------------------------------
            if pred_index is None:
                pred_index = tf.argmax(
                    predictions[0]
                )

            pred_index = tf.cast(
                pred_index,
                tf.int32
            )

            class_score = predictions[:, pred_index]

        # ---------------------------------------------------------
        # 5. Calculate gradients
        # ---------------------------------------------------------
        grads = tape.gradient(
            class_score,
            conv_outputs
        )

        if grads is None:
            print(
                "[Grad-CAM] Gradient calculation returned None."
            )
            return None

        # ---------------------------------------------------------
        # 6. Global average pooling
        # ---------------------------------------------------------
        pooled_grads = tf.reduce_mean(
            grads,
            axis=(1, 2)
        )

        # Remove batch dimension
        conv_outputs = conv_outputs[0]
        pooled_grads = pooled_grads[0]

        # ---------------------------------------------------------
        # 7. Weight feature maps
        # ---------------------------------------------------------
        heatmap = tf.reduce_sum(
            conv_outputs * pooled_grads,
            axis=-1
        )

        # ---------------------------------------------------------
        # 8. ReLU
        # ---------------------------------------------------------
        heatmap = tf.maximum(
            heatmap,
            0
        )

        # ---------------------------------------------------------
        # 9. Normalize
        # ---------------------------------------------------------
        max_value = tf.reduce_max(
            heatmap
        )

        if float(max_value) > 0:
            heatmap = heatmap / max_value

        heatmap = heatmap.numpy()

        # ---------------------------------------------------------
        # 10. Resize to 224x224
        # ---------------------------------------------------------
        heatmap = cv2.resize(
            heatmap,
            (224, 224)
        )

        print(
            "[Grad-CAM] Heatmap generated successfully."
        )

        return heatmap

    except Exception as e:

        print(
            f"[Grad-CAM] Error generating heatmap: {e}"
        )

        return None


def overlay_heatmap(original_image, heatmap, alpha=0.40):
    """
    Overlay Grad-CAM heatmap on an RGB image.
    """

    try:

        if heatmap is None:
            return original_image

        # ---------------------------------------------------------
        # Ensure original image is uint8
        # ---------------------------------------------------------
        if original_image.dtype != np.uint8:

            image = original_image.astype(
                np.float32
            )

            image = image - image.min()

            if image.max() > 0:
                image = (
                    image /
                    image.max()
                ) * 255.0

            original_image = np.uint8(
                image
            )

        # ---------------------------------------------------------
        # Resize heatmap to original image dimensions
        # ---------------------------------------------------------
        if heatmap.shape[:2] != original_image.shape[:2]:

            heatmap = cv2.resize(
                heatmap,
                (
                    original_image.shape[1],
                    original_image.shape[0]
                )
            )

        # ---------------------------------------------------------
        # Convert heatmap to 0-255
        # ---------------------------------------------------------
        heatmap_uint8 = np.uint8(
            np.clip(
                heatmap,
                0,
                1
            ) * 255
        )

        # ---------------------------------------------------------
        # Apply color map
        # ---------------------------------------------------------
        heatmap_color = cv2.applyColorMap(
            heatmap_uint8,
            cv2.COLORMAP_JET
        )

        # OpenCV BGR -> RGB
        heatmap_color = cv2.cvtColor(
            heatmap_color,
            cv2.COLOR_BGR2RGB
        )

        # ---------------------------------------------------------
        # Overlay
        # ---------------------------------------------------------
        overlay = cv2.addWeighted(
            original_image,
            1.0 - alpha,
            heatmap_color,
            alpha,
            0
        )

        return overlay

    except Exception as e:

        print(
            f"[Grad-CAM] Error creating overlay: {e}"
        )

        return original_image