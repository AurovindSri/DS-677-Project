import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from io import BytesIO
import uvicorn


# Define the CNN model (same architecture as before)
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Load the .pth file into the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model
model = CNNModel().to(device)

# Specify the path to your downloaded .pth model file
model_path = "model-viT.pth"  # Replace with the correct path

# Load the model weights into the model
model = torch.load(model_path, map_location=device)

# Set the model to evaluation mode
model.eval()

# Define the FastAPI app
app = FastAPI()

# Image transformation
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize to match CIFAR-10 image size
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the uploaded image
        image_data = await file.read()
        image = Image.open(BytesIO(image_data)).convert("RGB")

        # Transform the image
        image = transform(image).unsqueeze(0).to(device)

        # Perform prediction
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            class_idx = predicted.item()

        # CIFAR-10 class names
        class_names = [
            "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"
        ]
        predicted_class = class_names[class_idx]
        probability = torch.softmax(outputs, dim=1)[0][class_idx].item()

        return {"class_name": predicted_class, "probability": probability}

    except Exception as e:
        return {"error": str(e)}

# Command to run the app:
# uvicorn app:app --reload


# Command to run the app:
# uvicorn app:app --reload
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8085, log_level="info")
