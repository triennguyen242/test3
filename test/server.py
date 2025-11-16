import os
from flask import Flask, request, jsonify

app = Flask(__name__)

# Railway/Fly.io cung cấp port động
port = int(os.environ.get("PORT", 10000))

# Đường dẫn tới model
model_path = os.path.join(os.path.dirname(__file__), "model", "mobilenetv5.tflite")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    # TODO: xử lý ảnh base64 với tflite
    return jsonify({"result": "anemia", "confidence": 0.95})

if __name__ == "__main__":
    # host="0.0.0.0" để server public truy cập được
    app.run(host="0.0.0.0", port=port)
