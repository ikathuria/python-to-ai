import pickle
import numpy as np
import onnx
import tf2onnx
import tensorflow as tf
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import os

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "app", "static", "models")
OUTPUT_DIR = MODELS_DIR  # Save ONNX models in same directory


def convert_sklearn_model(name, input_dim):
    print(f"Converting {name}...")
    try:
        model_path = os.path.join(MODELS_DIR, f"{name}.sav")
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        # Define input type
        initial_type = [("float_input", FloatTensorType([None, input_dim]))]

        # Convert
        onnx_model = convert_sklearn(model, initial_types=initial_type)

        # Save
        output_path = os.path.join(OUTPUT_DIR, f"{name}.onnx")
        with open(output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        print(f"Success: {output_path}")
    except Exception as e:
        print(f"Failed to convert {name}: {e}")


def convert_keras_model(name):
    print(f"Converting {name}...")
    try:
        model_path = os.path.join(MODELS_DIR, name)
        # Load Keras model
        model = tf.keras.models.load_model(model_path)

        # Convert to ONNX
        # opset 13 is widely supported by onnxruntime-web
        spec = (tf.TensorSpec((None, 100, 100, 1), tf.float32, name="input"),)
        output_path = os.path.join(OUTPUT_DIR, "hand_gesture.onnx")

        model_proto, _ = tf2onnx.convert.from_keras(
            model, input_signature=spec, opset=13, output_path=output_path
        )
        print(f"Success: {output_path}")
    except Exception as e:
        print(f"Failed to convert {name}: {e}")


def main():
    if not os.path.exists(MODELS_DIR):
        print(f"Error: Models directory not found at {MODELS_DIR}")
        return

    # 1. Census Income Models (15 features)
    # naive_bayes.sav, knn.sav, log_reg.sav
    census_models = ["naive_bayes", "knn", "log_reg"]
    for m in census_models:
        convert_sklearn_model(m, 15)

    # 2. Iris Clustering (2 features: Petal Length, Petal Width)
    # kmeans.sav
    convert_sklearn_model("kmeans", 2)

    # 3. Hand Gesture (Keras .h5)
    # 10Apr_model_12.h5
    convert_keras_model("10Apr_model_12.h5")


if __name__ == "__main__":
    main()
