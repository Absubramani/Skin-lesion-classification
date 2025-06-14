import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models.vision_transformer import vit_b_16
from flask import Flask, render_template, request, jsonify
from PIL import Image

app = Flask(__name__)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = vit_b_16(pretrained=False)
model.heads.head = nn.Linear(model.heads.head.in_features, 2)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Route for the main page
@app.route('/')
def index():
    return render_template('index.html')  # You must save your HTML file as templates/index.html

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": True, "message": "No image uploaded."})

    file = request.files['image']
    try:
        image = Image.open(file).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            predicted_class = int(torch.argmax(outputs, dim=1).item())

        decision = "Benign" if predicted_class == 0 else "Malignant"
        confidence = f"{probabilities[predicted_class]*100:.2f}%"
        probs = f"Benign: {probabilities[0]*100:.2f}%, Malignant: {probabilities[1]*100:.2f}%"

        return jsonify({
            "error": False,
            "decision": decision,
            "correctness": confidence,
            "probabilities": probs
        })

    except Exception as e:
        return jsonify({"error": True, "message": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
