from flask import Flask, render_template, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
from model import CNNtoRNN  # Import your model
import pickle

app = Flask(__name__)

# Load the saved model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNtoRNN(embed_size=256, hidden_size=256, vocab_size=5000, num_layers=1)
checkpoint = torch.load("flickr8k_model.pth", map_location=device)
model.load_state_dict(checkpoint['state_dict'])
model.eval()  # Set the model to evaluation mode

# Load the vocabulary
with open('vocab.pkl', 'rb') as f:
    vocabulary = pickle.load(f)

# Preprocessing function for incoming image
transform = transforms.Compose(
    [
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    img_file = request.files['image']
    img = Image.open(img_file.stream).convert("RGB")

    # Preprocess the image
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Generate the caption
    caption = model.caption_image(img_tensor, vocabulary)  # You need to pass vocabulary loaded

    return jsonify({'caption': " ".join(caption)})

if __name__ == "__main__":
    app.run(debug=True)
