import os
import base64
import io
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Railway/Fly.io cung cấp port động
port = int(os.environ.get("PORT", 10000))

# Đường dẫn tới model
model_path = os.path.join(os.path.dirname(__file__), "model", "mobilenetv5.tflite")

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Lấy input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Giả sử model chỉ phân 2 class: anemia vs normal
labels = ["anemia", "normal"]

def preprocess_image(base64_str):
    # Decode base64 → PIL Image
    image_data = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    
    # Resize về input model (lấy từ input_details)
    input_shape = input_details[0]['shape']  # e.g., [1, 224, 224, 3]
    image = image.resize((input_shape[2], input_shape[1]))
    
    # Chuyển sang numpy, normalize nếu cần (0-1)
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # [1, H, W, C]
    return img_array

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        if "image" not in data:
            return jsonify({"error": "No image provided"}), 400
        
        base64_img = data["image"]
        img_array = preprocess_image(base64_img)

        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()

        # Lấy output
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]
        
        # Nếu model 2 class -> softmax
        if output_data.ndim == 1 and len(output_data) == len(labels):
            confidence = float(np.max(output_data))
            result = labels[int(np.argmax(output_data))]
        else:
            # Nếu model trả 1 số float (binary)
            confidence = float(output_data[0])
            result = "anemia" if confidence > 0.5 else "normal"

        return jsonify({"result": result, "confidence": confidence})
    
    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # host="0.0.0.0" để server public truy cập được
    app.run(host="0.0.0.0", port=port)
