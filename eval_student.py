import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import datasets, transforms
from transformers import AdamW
from torchvision.models import resnet18
from fvcore.nn import FlopCountAnalysis
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
import time
from torch.optim.lr_scheduler import CosineAnnealingLR

class Config:
    in_channels = 3
    num_classes = 10
    batch_size = 64
    lr = 1e-3
    epochs = 20
    num_students = 3
    num_big_classes = num_students
    hidden_dim = 256
    #temperature = 3.0
    temperature = 5.0
    alpha = 0.7
    teacher_model_path = "resnet18_cifar10_tailored_epoch20.pth"
    student_model_path = "student_{}.pth"

config = Config()

class ResNet18CIFAR10(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18CIFAR10, self).__init__()
        self.model = resnet18(pretrained=False)  # Load base ResNet-18
        self.model.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False
        )  # Replace first conv layer
        self.model.maxpool = nn.Identity()  # Remove MaxPooling layer
        self.model.fc = nn.Linear(512, num_classes)  # Adjust fully connected layer

    def forward(self, x):
        return self.model(x)

# Data Loaders with Data Augmentation
def get_data_loaders():
    # Training transformations (with augmentations)
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))  # Mean and std of CIFAR-10
    ])

    # Test transformations (without augmentations)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))  # Same mean and std as training set
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform_train, download=True)
    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform_test, download=True)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader

class StudentModel(nn.Module):
    def __init__(self):
        super(StudentModel, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(8 * 8 * 64, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_classes)
        )

    def forward(self, x):
        return self.network(x)
    

def evaluate_with_metrics(model, loader, device, description="Model"):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct = 0, 0
    total_samples = 0

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)

            if i == 0:
                start_time = time.time()

                flops_input = inputs[:1].to(device)
                flops_analysis = FlopCountAnalysis(model, flops_input)
                flops_per_image = flops_analysis.total() / batch_size

                end_time = time.time()

                latency = (end_time - start_time) / batch_size

            outputs = model(inputs)

            loss = criterion(outputs, targets)
            total_loss += loss.item()
            correct += (outputs.argmax(1) == targets).sum().item()
            total_samples += batch_size

    accuracy = correct / total_samples
    print(f"{description} Results:")
    print(f"Loss: {total_loss / len(loader):.4f}, Accuracy: {accuracy:.4f}")
    print(f"Latency per Image: {latency:.6f} secs")
    print(f"FLOPs per Image: {flops_per_image / 1e6:.2f} MFLOPs")

    return total_loss, accuracy, latency, flops_per_image

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_data_loaders()
    
    student_model_folder = "student_models"
    student_model_paths = [
        "student_c1_l0.0625.pth",
        "student_c1_l0.125.pth",
        "student_c1_l0.25.pth",
        "student_c1_l0.5.pth",
        "student_c1_l1.pth",
        "student_c1_l2.pth",
        "student_c1_l4.pth",
        "student_c1_l8.pth"
    ]
    
    # For each student model, load the weights and evaluate
    students = []
    for i, student_model_path in enumerate(student_model_paths):
        student = StudentModel().to(device)
        student.load_state_dict(torch.load(f"{student_model_folder}/{student_model_path}"))
        students.append(student)
        print(f"Student {i + 1} loaded from {student_model_path}")
        evaluate_with_metrics(student, test_loader, device, description=f"Student {i + 1}")


if __name__ == "__main__":
    main()
