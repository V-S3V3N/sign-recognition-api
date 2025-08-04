from flask import Flask, request, jsonify
import os
import uuid
from inference_pipeline import predict_sign

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    return "IT IS WORKING", 200

@app.route("/predict", methods=["POST"])
def predict():
    if "video" not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    video_file = request.files["video"]
    filename = f"{uuid.uuid4()}.mp4"
    video_path = os.path.join(UPLOAD_FOLDER, filename)
    video_file.save(video_path)

    try:
        result = predict_sign(video_path)
        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(video_path)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
