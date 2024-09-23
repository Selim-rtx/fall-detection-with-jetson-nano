import os
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import Dataset, DataLoader
import random

# Command-line argument parsing
import argparse
parser = argparse.ArgumentParser(description="Train a fall detection model using PyTorch and LSTM.")
parser.add_argument("--dataset", type=str, required=True, help="Path to dataset with videos")
parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
parser.add_argument("--sequence_length", type=int, default=16, help="Number of frames per sequence")
parser.add_argument("--output_model", type=str, default="trained_fall_lstm_model.pth", help="Path to save the trained model")
args = parser.parse_args()

# Define video transforms (for each frame)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet normalization
])

# Custom Video Dataset class
class VideoDataset(Dataset):
    def __init__(self, root_dir, sequence_length, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.sequence_length = sequence_length
        self.classes = ['falling', 'not_falling']
        self.videos = []

        # Load video paths and their labels (0 for falling, 1 for not_falling)
        for class_index, class_name in enumerate(self.classes):
            class_path = os.path.join(self.root_dir, class_name)
            for video_name in os.listdir(class_path):
                if video_name.endswith(('.mp4', '.avi', '.mkv')):
                    self.videos.append((os.path.join(class_path, video_name), class_index))

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_path, label = self.videos[idx]

        # Load video with OpenCV
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0

        while frame_count < self.sequence_length:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert frame to RGB (OpenCV loads as BGR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Apply transformation to the frame
            if self.transform:
                frame = self.transform(frame)

            frames.append(frame)
            frame_count += 1

        cap.release()

        # If the video has fewer frames than the sequence length, repeat frames to match sequence_length
        while len(frames) < self.sequence_length:
            frames.append(frames[-1])  # Repeat last frame

        # Stack frames (each frame is [3, 224, 224]), now it's a tensor of shape [sequence_length, 3, 224, 224]
        frames_tensor = torch.stack(frames)

        return frames_tensor, label

# Define the model: CNN + LSTM
class CNN_LSTM_Model(nn.Module):
    def __init__(self, num_classes=2, sequence_length=16):
        super(CNN_LSTM_Model, self).__init__()
        # Use ResNet18 for feature extraction
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Identity()  # Remove the final classification layer
        
        # LSTM for sequence modeling
        self.lstm = nn.LSTM(input_size=512, hidden_size=256, num_layers=1, batch_first=True)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        batch_size, sequence_length, C, H, W = x.size()
        
        # Extract features for each frame in the sequence using ResNet
        cnn_features = []
        for i in range(sequence_length):
            frame = x[:, i, :, :, :]  # Extract each frame in the sequence
            feature = self.resnet(frame)  # Feature vector of shape [batch_size, 512]
            cnn_features.append(feature)
        
        # Stack features along the sequence dimension
        cnn_features = torch.stack(cnn_features, dim=1)  # Shape: [batch_size, sequence_length, 512]

        # Pass the sequence of features through the LSTM
        lstm_out, _ = self.lstm(cnn_features)  # Shape: [batch_size, sequence_length, 256]
        
        # Take the output of the last time step
        lstm_out = lstm_out[:, -1, :]  # Shape: [batch_size, 256]

        # Pass through the final fully connected layer
        out = self.fc(lstm_out)  # Shape: [batch_size, num_classes]

        return out

# Load dataset
train_dataset = VideoDataset(root_dir=args.dataset, sequence_length=args.sequence_length, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

# Load model
model = CNN_LSTM_Model(num_classes=2, sequence_length=args.sequence_length)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Set up loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct = 0
        total = 0
        
        for frames, labels in train_loader:
            frames = frames.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(frames)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            running_loss += loss.item() * frames.size(0)

        # Calculate epoch loss and accuracy
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100 * correct / total

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

    print("Training complete!")

# Save model function
def save_model(model, output_path):
    torch.save(model.state_dict(), output_path)
    print(f"Model saved to {output_path}")

# Train the model
train_model(model, train_loader, criterion, optimizer, num_epochs=args.epochs, device=device)

# Save the trained model
save_model(model, args.output_model)
