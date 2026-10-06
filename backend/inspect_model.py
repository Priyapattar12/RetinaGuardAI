import os
import tensorflow as tf

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "model",
    "retina_model_weighted.keras"
)

print("Loading model:")
print(os.path.abspath(MODEL_PATH))

model = tf.keras.models.load_model(MODEL_PATH)

print("\n===== TOP LEVEL MODEL =====")
model.summary()

print("\n===== TOP LEVEL LAYERS =====")

for i, layer in enumerate(model.layers):
    print(
        i,
        layer.name,
        type(layer).__name__,
        getattr(layer.output, "shape", None)
    )

    if hasattr(layer, "layers") and len(layer.layers) > 0:

        print("   --- Nested layers ---")

        for j, sub_layer in enumerate(layer.layers):
            try:
                shape = sub_layer.output.shape
            except Exception:
                shape = "unknown"

            print(
                "   ",
                j,
                sub_layer.name,
                type(sub_layer).__name__,
                shape
            )