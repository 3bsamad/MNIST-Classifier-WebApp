import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from streamlit_drawable_canvas import st_canvas
from mnist_classifier import MNISTClassifier

# Set device to MPS if available, otherwise CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load the trained model and move to device
model = MNISTClassifier()
model.load_state_dict(torch.load('models/mnist_model_50.pth', map_location=device))
model.to(device)  # Move model to the appropriate device
model.eval()

# Define transformations to match MNIST data pre-processing
transform = transforms.Compose([
    transforms.Grayscale(),       # Ensure the image is in grayscale
    transforms.Resize((28, 28)),  # Resize to 28x28 pixels
    transforms.ToTensor(),        # Convert to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize as per MNIST
])

st.title("MNIST Digit Classifier")
st.write("Draw a digit below and let the model predict!")

# Create a canvas component
canvas_result = st_canvas(
    fill_color="black",  # Background color for the canvas
    stroke_width=10,     # Thickness of the pen strokes
    stroke_color="white",# Color for the pen strokes
    background_color="black",
    height=150,          # Canvas height in pixels
    width=150,           # Canvas width in pixels
    drawing_mode="freedraw",
    key="canvas"
)

# Process and predict when there's a drawing
if canvas_result.image_data is not None:
    # Convert the image data from canvas to a PIL image
    img = Image.fromarray(np.uint8(canvas_result.image_data))

    # Apply transformations and prepare the image for the model
    img = transform(img).unsqueeze(0).to(device)  # Add batch dimension and move to device

    # Get the model prediction
    with torch.no_grad():
        output = model(img)
        _, prediction = torch.max(output, 1)

    st.write(f"Prediction: {prediction.item()}")