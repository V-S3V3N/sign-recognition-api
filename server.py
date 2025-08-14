from flask import Flask, request, jsonify
import os
import uuid
from inference_pipeline import segment_and_predict_signs
from inference_pipeline_msl import msl_segment_and_predict_signs
from utils.sentence_generator import llm_generate_sentence

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    return "IT IS WORKING", 200

def save_uploaded_video(file):
    filename = f"{uuid.uuid4()}.mp4"
    video_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(video_path)
    return video_path

def process_video(video_path, segment_predict_func):
    results = segment_predict_func(video_path)
    if not results:
        return {"predictions": [], "sentence": "", "success": True}
    valid_results = [p for p in results if len(p) >= 4] #check and see the predictions there got > 4 fields
    words_only = [r[2] for r in valid_results]
    sentence = llm_generate_sentence(words_only)
    return {
        "predictions": [
            {"start": p[0], "end": p[1], "label": p[2], "confidence": p[3]}
            for p in valid_results
        ],
        "sentence": sentence,
        "success": True,
    }

@app.route("/predict", methods=["POST"])
def predict():
    if "video" not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    video_file = request.files["video"]
    video_path = save_uploaded_video(video_file)

    try:
        # when video coming in we segment them first, then only send to predict sign
        response = process_video(video_path, segment_and_predict_signs)
        return response
    except Exception as e:
        import traceback
        print("❌ Error in /predict:", e)
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    finally:
        os.remove(video_path)

@app.route("/msl_predict", methods=["POST"])
def predict_msl():
    if "video" not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    video_file = request.files["video"]
    video_path = save_uploaded_video(video_file)

    try:
        # when video coming in we segment them first, then only send to predict sign
        response = process_video(video_path, msl_segment_and_predict_signs)
        return response
    except Exception as e:
        import traceback
        print("❌ Error in /msl_predict:", e)
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    finally:
        os.remove(video_path)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
