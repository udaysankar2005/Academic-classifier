import os
import json
from transformers import pipeline

MODEL_DIR = os.path.join("..", "models", "abstract_classifier")

if not os.path.exists(MODEL_DIR):
    MODEL_DIR = os.path.join("models", "abstract_classifier")

print("Using model dir:", MODEL_DIR)

classifier = pipeline(
    "text-classification",
    model=MODEL_DIR,
    tokenizer=MODEL_DIR,
    truncation=True,
    max_length=256,
    device=-1
)

# load label map
label_map_path = os.path.join(MODEL_DIR, "label_map.json")
if os.path.exists(label_map_path):
    with open(label_map_path, "r", encoding="utf-8") as f:
        label2id = json.load(f)
    id2label = {v: k for k, v in label2id.items()}
else:
    id2label = None
    print("Warning: label_map.json not found in model folder. Raw LABEL_N will be returned.")

friendly = {
    "AI": "Artificial Intelligence",
    "Business": "Business Research",
    "Healthcare": "Healthcare Research",
    "Environmental Science": "Environment Research"
}

def pretty_label(raw_label):
    if raw_label and raw_label.startswith("LABEL_") and id2label:
        idx = int(raw_label.replace("LABEL_", ""))
        original = id2label.get(idx, raw_label)
    else:
        original = raw_label
    return friendly.get(original, original)

if __name__ == "__main__":
    print("Enter abstract text (empty line to quit):")
    while True:
        try:
            text = []
            while True:
                line = input()
                if line == "":
                    break
                text.append(line)
        except EOFError:
            # user pressed Ctrl+D
            break
        text = "\n".join(text).strip()
        if not text:
            print("No text entered; exiting.")
            break
        out = classifier(text)[0]
        raw = out.get("label")
        score = out.get("score", 0.0)
        print("Predicted:", pretty_label(raw))
        print("Confidence:", f"{score * 100:.2f}%\n")
