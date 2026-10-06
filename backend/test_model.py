import os
import numpy as np
import tensorflow as tf
from tensorflow import keras


MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "model",
    "retina_model_weighted.keras"
)

IMG_SIZE = (224, 224)


print("Loading model:")
print(os.path.abspath(MODEL_PATH))

model = keras.models.load_model(MODEL_PATH)

print("\nModel loaded successfully!")

print("\nModel input:")
print(model.input_shape)

print("\nModel output:")
print(model.output_shape)

# Dummy image just to verify inference
dummy_image = np.zeros(
    (1, 224, 224, 3),
    dtype=np.float32
)

prediction = model.predict(
    dummy_image,
    verbose=0
)

print("\nPrediction shape:")
print(prediction.shape)

print("\nPrediction probabilities:")
print(prediction[0])

print("\nPredicted class:")
print(np.argmax(prediction[0]))

print("\nModel test completed successfully.")
