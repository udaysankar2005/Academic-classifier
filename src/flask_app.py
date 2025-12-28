import os
import json
from flask import Flask, render_template, request, jsonify
from transformers import pipeline
import logging


MODEL_DIR = os.environ.get("MODEL_DIR", "models/abstract_classifier")

app = Flask(__name__, template_folder="../templates", static_folder="../static")


logger = logging.getLogger("werkzeug")
logger.setLevel(logging.INFO)

classifier = None
def get_classifier():
    global classifier
    if classifier is None:
        # prefer GPU if available; pipeline expects device index (0) or -1 for CPU
        device = 0 if (os.environ.get("USE_CUDA","") == "1") else -1
        classifier = pipeline(
            "text-classification",
            model=MODEL_DIR,
            tokenizer=MODEL_DIR,
            truncation=True,
            max_length=256,
            device=device
        )
        print("Loaded pipeline. Device:", "cuda" if device == 0 else "cpu")
    return classifier


label_map_path = os.path.join(MODEL_DIR, "label_map.json")
inv_label_map = {}
if os.path.exists(label_map_path):
    try:
        with open(label_map_path, "r", encoding="utf-8") as f:
            label2id = json.load(f)
        inv_label_map = {int(v): k for k, v in label2id.items()}
        print("Loaded label_map.json with", len(inv_label_map), "entries.")
    except Exception as e:
        print("Warning: failed to load/invert label_map.json:", e)

friendly_names = {
    "AI": "Artificial Intelligence",
    "Business": "Business Research",
    "Healthcare": "Healthcare Research",
    "Environmental Science": "Environment Research",
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    payload = request.get_json(force=True) or {}
    text = payload.get('text', '').strip()
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    clf = get_classifier()
    try:
        out = clf(text, truncation=True, max_length=256)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    raw_label = out[0].get('label')
    score = float(out[0].get('score', 0.0))

    # convert LABEL_N -> original label if label_map exists
    original = raw_label
    if isinstance(raw_label, str) and raw_label.startswith('LABEL_'):
        try:
            idx = int(raw_label.replace('LABEL_', ''))
            original = inv_label_map.get(idx, raw_label)
        except Exception:
            original = raw_label

    friendly = friendly_names.get(original, original)
    return jsonify({'label': friendly, 'original_label': original, 'confidence': round(score * 100, 2)})

if __name__ == '__main__':
    
    app.run(host='0.0.0.0', port=5000, debug=True)
