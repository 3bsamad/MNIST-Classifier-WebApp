import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

def show_predictions(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    images, labels = next(iter(test_loader))  # Get a batch of test images
    
    # Move images and labels to device
    images, labels = images.to(device), labels.to(device)
    
    with torch.no_grad():  # No need for gradients during inference
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
    
    # Plot the images in the batch with their predicted and true labels
    fig, axes = plt.subplots(1, 6, figsize=(12, 3))
    for i in range(6):  # Display first 6 images
        ax = axes[i]
        ax.imshow(images[i].view(28, 28).cpu().numpy(), cmap='gray')  # Move to CPU for display
        ax.set_title(f"Pred: {predictions[i].item()}\nTrue: {labels[i].item()}")
        ax.axis('off')

    plt.show()

class MNISTClassifier(nn.Module):
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 28, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(28, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * 64, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


if __name__ == "__main__":


    # Define transformations for the dataset (convert to tensor and normalize)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # Download and load the training data
    train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transform)

    # Define data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Instantiate the model
    model = MNISTClassifier().to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Set model to training mode
    model.train()

    # Number of epochs
    epochs = 50

    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:

            images, labels = images.to(device), labels.to(device)
        
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), 'deep_learning/models/mnist_model_20.pth')


    # # Set model to evaluation mode
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient computation for inference
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy on the test set: {accuracy:.2f}%')

    # Function to display images with predictions


    # Call the function to display predictions
    # show_predictions(model, test_loader)