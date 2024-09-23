import cv2
import torch
import argparse
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision import models

# Command-line argument parsing
parser = argparse.ArgumentParser(description="Run fall detection on video input using LSTM and save output with predictions.")
parser.add_argument("--input", type=str, required=True, help="Path to input video file")
parser.add_argument("--output", type=str, required=True, help="Path to output video file")
parser.add_argument("--sequence_length", type=int, default=16, help="Number of frames per sequence")
parser.add_argument("--model", type=str, default="trained_fall_lstm_model.pth", help="Path to the trained PyTorch model")
args = parser.parse_args()

# Define video transforms (for each frame)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet normalization
])

# Define the model: CNN + LSTM
class CNN_LSTM_Model(nn.Module):
    def __init__(self, num_classes=2, sequence_length=16):
        super(CNN_LSTM_Model, self).__init__()
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.fc = nn.Identity()  # Remove the final classification layer
        self.lstm = nn.LSTM(input_size=512, hidden_size=256, num_layers=1, batch_first=True)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        batch_size, sequence_length, C, H, W = x.size()
        cnn_features = []
        for i in range(sequence_length):
            frame = x[:, i, :, :, :]  # Extract each frame in the sequence
            feature = self.resnet(frame)
            cnn_features.append(feature)
        
        cnn_features = torch.stack(cnn_features, dim=1)
        lstm_out, _ = self.lstm(cnn_features)
        lstm_out = lstm_out[:, -1, :]
        out = self.fc(lstm_out)
        return out

# Load the trained model
model = CNN_LSTM_Model(num_classes=2, sequence_length=args.sequence_length)
model.load_state_dict(torch.load(args.model))
model.eval()  # Set model to evaluation mode

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Open video input
cap = cv2.VideoCapture(args.input)
if not cap.isOpened():
    print(f"Error: Could not open video {args.input}")
    sys.exit()

# Get the video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Open video writer to save the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for output video (you can change it as needed)
out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

frames = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Apply the transform to the frame
    frame_tensor = transform(frame_rgb)

    frames.append(frame_tensor)

    # If we have enough frames, perform inference
    if len(frames) == args.sequence_length:
        frames_tensor = torch.stack(frames).unsqueeze(0).to(device)  # Shape: [1, sequence_length, 3, 224, 224]
        
        with torch.no_grad():
            outputs = model(frames_tensor)
            _, predicted = torch.max(outputs, 1)
            confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted].item() * 100
            class_desc = "Falling" if predicted.item() == 0 else "Not Falling"
        
        # Convert frame back to BGR for OpenCV
        frame_display = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        # Overlay the prediction text on the frame
        label = f"{class_desc} ({confidence:.2f}%)"
        cv2.putText(frame_display, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Write the frame with the overlay to the output video
        out.write(frame_display)

        # Clear the frames buffer for the next sequence
        frames = []

cap.release()
out.release()
print(f"Video saved to {args.output}")
