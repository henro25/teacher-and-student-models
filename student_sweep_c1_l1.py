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
    epochs = 1
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


class GatingNetwork(nn.Module):
    def __init__(self, num_students, input_dim):
        super(GatingNetwork, self).__init__()
        # Convolutional layers to extract spatial features
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # Output: 32 x 32 x 32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                          # Output: 16 x 16 x 32

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # Output: 16 x 16 x 64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                          # Output: 8 x 8 x 64
        )
        # Fully connected layers to produce routing probabilities
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 8 * 64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_students)
        )
        self.temperature = 5.0  # High initial temperature for exploration

    def forward(self, x):
        # x is expected to have shape [batch_size, 3, 32, 32]
        x = x.view(-1, 3, 32, 32)
        features = self.conv_layers(x)
        logits = self.fc_layers(features)
        # Apply temperature scaling to logits before softmax
        return F.softmax(logits / self.temperature, dim=1)


class MoE(nn.Module):
    def __init__(self, students, gating_net):
        super(MoE, self).__init__()
        self.students = nn.ModuleList(students)
        self.gating_net = gating_net

    def forward(self, x, return_router_assignments=False):
        batch_size = x.size(0)
        gating_probs = self.gating_net(x.view(batch_size, -1))  # Router probabilities
        best_experts = gating_probs.argmax(dim=1)  # Selected experts for each input

        outputs = torch.zeros(batch_size, self.students[0].network[-1].out_features).to(x.device)
        for i, expert_idx in enumerate(best_experts):
            outputs[i] = self.students[expert_idx](x[i].unsqueeze(0)).squeeze(0)

        if return_router_assignments:
            return outputs, best_experts
        return outputs

def distill_teacher_to_student(teacher, student, loader, optimizer, criterion, device):
    teacher.eval()
    student.train()
    total_loss = 0
    correct = 0
    total_samples = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        with torch.no_grad():
            teacher_outputs = teacher(inputs)
            teacher_soft = F.softmax(teacher_outputs / config.temperature, dim=1)

        student_outputs = student(inputs)
        student_soft = F.log_softmax(student_outputs / config.temperature, dim=1)

        distill_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (config.temperature ** 2)
        hard_loss = F.cross_entropy(student_outputs, targets)
        loss = config.alpha * distill_loss + (1 - config.alpha) * hard_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (student_outputs.argmax(1) == targets).sum().item()
        total_samples += targets.size(0)

    accuracy = correct / total_samples
    print(f"Distill Loss: {total_loss / len(loader):.4f}, Accuracy: {accuracy:.4f}")

    return total_loss / len(loader)

def create_big_class_map_from_teacher(teacher, data_loader, num_big_classes, device):
    """
    Create a big class map by clustering the teacher's logits.
    """
    teacher.eval()
    all_logits, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            logits = teacher(inputs).cpu().numpy()
            all_logits.append(logits)
            all_labels.append(labels.cpu().numpy())

    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Calculate class embeddings by averaging teacher logits for each original class
    class_embeddings = {c: all_logits[all_labels == c].mean(axis=0) for c in np.unique(all_labels)}
    class_features = np.array([class_embeddings[c] for c in sorted(class_embeddings)])

    # Cluster the class embeddings into big classes
    kmeans = KMeans(n_clusters=num_big_classes, random_state=42).fit(class_features)
    big_class_map = {c: cluster for c, cluster in zip(sorted(class_embeddings), kmeans.labels_)}

    return big_class_map

def balance_big_class_map(big_class_map):
    total_counts = sum(big_class_map.values())
    for class_idx in big_class_map:
        big_class_map[class_idx] /= total_counts  # Normalize probabilities
    return big_class_map

def distill_teacher_to_router_with_clusters(
    teacher, router, loader, optimizer, device, big_class_map, epochs=5, use_combined_loss=True
):
    teacher.eval()
    router.train()

    # Create reverse mapping from class to cluster
    class_to_cluster = {c: cluster for c, cluster in big_class_map.items()}

    # Define the cosine annealing scheduler
    scheduler_router = CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        total_loss = 0
        correct_assignments = 0
        total_samples = 0
        alpha = epoch / epochs if use_combined_loss else 1.0  # Linearly increase soft loss weight

        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Compute cluster probabilities from teacher logits
            with torch.no_grad():
                teacher_logits = teacher(inputs)
                teacher_probs = F.softmax(teacher_logits / config.temperature, dim=1)
                cluster_probs = torch.zeros(inputs.size(0), config.num_big_classes, device=device)
                for cls, cluster in class_to_cluster.items():
                    cluster_probs[:, cluster] += teacher_probs[:, cls]

                # Compute hard cluster assignments for optional hard loss
                hard_labels = cluster_probs.argmax(dim=1)

            # Router outputs (probability of each cluster)
            router_logits = router(inputs)
            router_probs = F.softmax(router_logits, dim=1)

            # Compute soft label loss (distillation)
            soft_loss = F.kl_div(
                F.log_softmax(router_logits / config.temperature, dim=1),
                cluster_probs,
                reduction='batchmean'
            ) * (config.temperature ** 2)

            # Compute hard label loss (if enabled)
            hard_loss = F.cross_entropy(router_logits, hard_labels) if use_combined_loss else 0.0

            # Combine losses
            total_loss_batch = alpha * soft_loss + (1 - alpha) * hard_loss

            # # Diversity loss (optional)
            # router_usage = router_probs.mean(dim=0)  # Average over the batch
            # diversity_loss = torch.sum((router_usage - 1 / config.num_students) ** 2)

            # # Add diversity loss to total loss
            # total_loss_batch += 0.01 * diversity_loss

            # Backprop and optimization step
            optimizer.zero_grad()
            total_loss_batch.backward()
            optimizer.step()

            # Update metrics
            total_loss += total_loss_batch.item()
            predicted_clusters = router_probs.argmax(dim=1)
            ground_truth_clusters = torch.tensor(
                [class_to_cluster[label.item()] for label in targets], device=device
            )
            correct_assignments += (predicted_clusters == ground_truth_clusters).sum().item()
            total_samples += targets.size(0)

        # Step the scheduler after each epoch
        scheduler_router.step()

        # Log metrics
        router_accuracy = correct_assignments / total_samples
        print(f"Epoch {epoch + 1}/{epochs} - Router Loss: {total_loss / len(loader):.4f}, Accuracy: {router_accuracy:.4f}")

    return router

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

def visualize_specialization(moe_model, loader, device, num_classes, num_students):
    moe_model.eval()
    class_prob_tracker = defaultdict(lambda: torch.zeros(num_students, device=device))

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            router_probs = moe_model.gating_net(inputs.view(inputs.size(0), -1))

            for class_id in range(num_classes):
                class_indices = (targets == class_id)
                if class_indices.sum() > 0:
                    class_prob_tracker[class_id] += router_probs[class_indices].mean(dim=0)

    # Convert class probabilities to a NumPy array
    data = torch.stack([class_prob_tracker[c] for c in range(num_classes)]).cpu().numpy()

    # Visualization
    bar_width = 0.75
    x_indices = np.arange(num_classes)
    student_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    fig, ax = plt.subplots(figsize=(12, 8))
    bottom_values = np.zeros(num_classes)

    for student_id in range(num_students):
        student_probs = data[:, student_id]
        ax.bar(
            x_indices,
            student_probs,
            bar_width,
            bottom=bottom_values,
            color=student_colors[student_id % len(student_colors)],
            label=f"Student {student_id}",
            alpha=0.9
        )
        bottom_values += student_probs

    ax.set_xlabel("Classes")
    ax.set_ylabel("Router Probability")
    ax.set_title("Class Specialization Across Students")
    ax.set_xticks(x_indices)
    ax.set_xticklabels([f"Class {i}" for i in range(num_classes)])
    ax.legend(title="Students", loc="upper right")

    plt.tight_layout()
    plt.show()


def train_moe(moe_model, train_loader, optimizer, criterion, device, epochs=10):
    moe_model.train()

    # Define the cosine annealing scheduler
    scheduler_moe = CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total_samples = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            # Forward pass
            outputs, router_assignments = moe_model(inputs, return_router_assignments=True)
            loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Update metrics
            total_loss += loss.item()
            correct += (outputs.argmax(1) == targets).sum().item()
            total_samples += targets.size(0)

        # Step the scheduler after each epoch
        scheduler_moe.step()

        # Logging epoch metrics
        accuracy = correct / total_samples
        print(f"Epoch {epoch + 1}/{epochs} - MoE Loss: {total_loss / len(train_loader):.4f}, Accuracy: {accuracy:.4f}")

    return moe_model

def visualize_router_assignments(router, loader, num_classes, num_students, device):
    router.eval()
    assignment_counts = torch.zeros(num_classes, num_students, device=device)

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            router_probs = router(inputs.view(inputs.size(0), -1))

            for class_id in range(num_classes):
                class_indices = (targets == class_id)
                if class_indices.sum() > 0:
                    assignment_counts[class_id] += router_probs[class_indices].sum(dim=0)

    # Normalize counts for visualization
    assignment_counts = assignment_counts.cpu().numpy()
    assignment_counts /= assignment_counts.sum(axis=1, keepdims=True)

    # Plot
    bar_width = 0.75
    x_indices = np.arange(num_classes)
    student_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red

    plt.figure(figsize=(14, 8))  # Increase figure size for better visibility
    bottom_values = np.zeros(num_classes)

    for student_idx in range(num_students):
        student_probs = assignment_counts[:, student_idx]
        plt.bar(
            x_indices,
            student_probs,
            bar_width,
            bottom=bottom_values,
            color=student_colors[student_idx % len(student_colors)],
            label=f"Student {student_idx}",
            alpha=0.9
        )
        bottom_values += student_probs  # Stack bars vertically

    plt.xlabel("Classes")
    plt.ylabel("Router Assignment Probabilities")
    plt.title("Router Assignment Probabilities per Class")
    plt.xticks(x_indices, [f"Class {i}" for i in range(num_classes)], rotation=45)
    plt.ylim(0, 1)  # Ensure the y-axis always ranges from 0 to 1
    plt.legend(title="Students", loc="upper right", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add a grid for better readability
    plt.tight_layout()  # Adjust layout to fit everything properly
    plt.show()

def train_teacher(teacher, train_loader, test_loader, device, epochs=10, lr=1e-3):
    teacher.train()
    optimizer = AdamW(teacher.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss, correct, total_samples = 0, 0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            outputs = teacher(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == targets).sum().item()
            total_samples += targets.size(0)

        accuracy = correct / total_samples
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss / len(train_loader):.4f}, Accuracy: {accuracy:.4f}")

    print("\nEvaluating Teacher Model:")
    evaluate_with_metrics(teacher, test_loader, device, description="Teacher")
    return teacher

def fine_tune_router_with_hard_labels(router, train_loader, device, big_class_map, epochs=5, lr=1e-3):
    """
    Fine-tune the router with hard cluster labels, using cosine annealing.
    
    Args:
        router (nn.Module): The router network to fine-tune.
        train_loader (DataLoader): DataLoader for the training dataset.
        device (torch.device): Device to run training on.
        big_class_map (dict): Mapping from class labels to cluster labels.
        epochs (int): Number of fine-tuning epochs.
        lr (float): Initial learning rate.
    """
    # Reverse mapping from class labels to cluster labels
    class_to_cluster = {c: cluster for c, cluster in big_class_map.items()}
    
    # Set up optimizer and loss function
    optimizer = AdamW(router.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)  # Add Cosine Annealing
    criterion = nn.CrossEntropyLoss()
    
    # Fine-tune the router
    for epoch in range(epochs):
        router.train()
        total_loss = 0
        correct_assignments = 0
        total_samples = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Map class labels to cluster labels
            cluster_labels = torch.tensor(
                [class_to_cluster[label.item()] for label in targets],
                device=device,
                dtype=torch.long  # Ensure the target tensor is of type long
            )
            
            # Forward pass through the router
            optimizer.zero_grad()
            router_logits = router(inputs)
            
            # Compute loss
            loss = criterion(router_logits, cluster_labels)
            loss.backward()
            
            # Backpropagation and optimization step
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            predicted_clusters = router_logits.argmax(dim=1)
            correct_assignments += (predicted_clusters == cluster_labels).sum().item()
            total_samples += targets.size(0)
        
        # Step the cosine annealing scheduler
        scheduler.step()
        
        # Log metrics for the epoch
        accuracy = correct_assignments / total_samples
        print(f"Epoch {epoch + 1}/{epochs} - Fine-tuning Loss: {total_loss / len(train_loader):.4f}, "
              f"Accuracy: {accuracy:.4f}")

def evaluate_router_with_cluster_labels(router, loader, device, big_class_map):
    """
    Evaluate the router's accuracy based on cluster labels.
    
    Args:
        router (nn.Module): The trained router.
        loader (DataLoader): DataLoader for the dataset to evaluate.
        device (torch.device): Device to run evaluation on.
        big_class_map (dict): Mapping from class labels to cluster labels.
    """
    router.eval()
    class_to_cluster = {c: cluster for c, cluster in big_class_map.items()}
    
    total_samples = 0
    correct_predictions = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Map class labels to cluster labels
            cluster_labels = torch.tensor(
                [class_to_cluster[label.item()] for label in targets],
                device=device,
                dtype=torch.long
            )
            
            # Forward pass through the router
            router_logits = router(inputs)
            predicted_clusters = router_logits.argmax(dim=1)
            
            # Compare predictions to ground truth cluster labels
            correct_predictions += (predicted_clusters == cluster_labels).sum().item()
            total_samples += targets.size(0)

    accuracy = correct_predictions / total_samples
    print(f"Router Accuracy Based on Cluster Labels: {accuracy:.4f}")
    return accuracy

def evaluate_router_cluster_accuracies(router, loader, device, big_class_map):
    """
    Evaluate the router's accuracy for each cluster individually.
    
    Args:
        router (nn.Module): The trained router.
        loader (DataLoader): DataLoader for the validation dataset.
        device (torch.device): Device to run evaluation on.
        big_class_map (dict): Mapping from class labels to cluster labels.
    
    Returns:
        dict: Cluster-wise accuracy metrics.
    """
    router.eval()
    class_to_cluster = {c: cluster for c, cluster in big_class_map.items()}
    
    # Initialize tracking metrics for each cluster
    cluster_correct = defaultdict(int)
    cluster_total = defaultdict(int)

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Map class labels to cluster labels
            cluster_labels = torch.tensor(
                [class_to_cluster[label.item()] for label in targets],
                device=device,
                dtype=torch.long
            )
            
            # Forward pass through the router
            router_logits = router(inputs)
            predicted_clusters = router_logits.argmax(dim=1)
            
            # Update metrics for each cluster
            for i in range(len(cluster_labels)):
                cluster = cluster_labels[i].item()
                cluster_total[cluster] += 1
                if predicted_clusters[i] == cluster_labels[i]:
                    cluster_correct[cluster] += 1

    # Calculate accuracy for each cluster
    cluster_accuracies = {}
    for cluster, correct_count in cluster_correct.items():
        accuracy = correct_count / cluster_total[cluster] if cluster_total[cluster] > 0 else 0.0
        cluster_accuracies[cluster] = accuracy
        print(f"Cluster {cluster}: Accuracy = {accuracy:.4f} ({correct_count}/{cluster_total[cluster]})")
    
    return cluster_accuracies


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_data_loaders()

    # Step 1: Initialize and train the teacher model
    # teacher = TeacherModel().to(device)
    # print("Training the Teacher Model:")
    # teacher = train_teacher(teacher, train_loader, test_loader, device, epochs=config.epochs, lr=config.lr)

    # teacher = TeacherModel().to(device)
    # teacher.load_state_dict(torch.load(config.teacher_model_path, map_location=device))
    # teacher.eval()

    teacher = ResNet18CIFAR10().to(device)
    teacher.load_state_dict(torch.load("resnet18_cifar10_tailored_epoch20.pth"))
    teacher.to(device)

    print(f"\nEvaluating Teacher:")
    evaluate_with_metrics(teacher, test_loader, device, description="Teacher")

    # Step 2: Create the big class map from the teacher's logits
    print("\nCreating Big Class Map from Teacher:")
    big_class_map = create_big_class_map_from_teacher(teacher, train_loader, config.num_big_classes, device)
    print("Big Class Map:", big_class_map)

    # # Optional: Balance the big class map
    # print("\nBalancing Big Class Map:")
    # big_class_map = balance_big_class_map(big_class_map)
    # print("Balanced Big Class Map:", big_class_map)

    # Step 3: Initialize the router
    router = GatingNetwork(num_students=config.num_students, input_dim=3 * 32 * 32).to(device)
    optimizer_router = AdamW(router.parameters(), lr=config.lr)
    # optimizer_router = AdamW(router.parameters(), lr=config.lr, weight_decay=0.01)  # Add weight decay

    # Step 4: Initialize the students
    print("\nInitializing Students:")
    students = [StudentModel().to(device) for _ in range(config.num_students)]
    student_optimizers = [AdamW(student.parameters(), lr=config.lr) for student in students]
    # student_optimizers = [AdamW(student.parameters(), lr=config.lr, weight_decay=0.01) for student in students]
  

    # Step 4: Fine-tune the router directly using cluster labels
    print("\nFine-tuning Router with Hard Cluster Labels and Cosine Annealing:")
    fine_tune_router_with_hard_labels(
        router=router,
        train_loader=train_loader,
        device=device,
        big_class_map=big_class_map,
        epochs=20,  # Adjust epochs as needed
        lr=config.lr  # Use initial learning rate
    )


    # Evaluate the router's accuracy based on cluster labels
    print("\nEvaluating Router Based on Cluster Labels:")
    evaluate_router_with_cluster_labels(
        router=router,
        loader=test_loader,
        device=device,
        big_class_map=big_class_map
    )

    print("\nEvaluating Router Cluster-wise Validation Accuracies:")
    cluster_accuracies = evaluate_router_cluster_accuracies(
        router=router,
        loader=test_loader,
        device=device,
        big_class_map=big_class_map
    )



    print("\nVisualizing Router Assignments:")
    visualize_router_assignments(router, test_loader, config.num_classes, config.num_students, device)
    
    # save the router model
    router_save_path = "router.pth"
    torch.save(router.state_dict(), router_save_path)

    # # Step 6: Distill teacher knowledge into students
    print("\nDistilling Teacher Knowledge into a Single Student:")
    single_student = StudentModel().to(device)
    optimizer_student = AdamW(single_student.parameters(), lr=config.lr)

    for epoch in range(config.epochs):
        distill_teacher_to_student(teacher, single_student, train_loader, optimizer_student, nn.CrossEntropyLoss(), device)

    # Save the single student model
    single_student_save_path = config.student_model_path.format(1)
    torch.save(single_student.state_dict(), single_student_save_path)
    print(f"Single Student saved to {single_student_save_path}")

    # print("\nDuplicating the Single Student:")
    ##Load the pre-trained single student model
    # single_student_path = "student_1.pth"  # Ensure this is the correct path to your saved student model

    # students = []
    # for i in range(config.num_students):
    #     student = StudentModel().to(device)  # Initialize a new student model
    #     student.load_state_dict(torch.load(single_student_path))  # Load the weights from the single student
    #     students.append(student)  # Add the student to the list

    # Evaluate the single student on the validation dataset
    print("\nEvaluating the Single Student:")
    evaluate_with_metrics(students[0], test_loader, device, description="Single Student")
    
    # Save the single student model
    single_student_save_path = "student_c1_l1.pth"
    torch.save(single_student.state_dict(), single_student_save_path)

    print("\nDuplicating the Single Student:")
    students = []
    for i in range(config.num_students):
        student = StudentModel().to(device)
        student.load_state_dict(torch.load(single_student_save_path))
        students.append(student)

    print(f"{config.num_students} Students initialized by duplicating the Single Student model.")

    # print("\nDistilling Teacher Knowledge into Students:")
    # for i, student in enumerate(students):
    #     optimizer_student = AdamW(student.parameters(), lr=config.lr)
    #     #for epoch in range(config.epochs // 2):
    #     for epoch in range(30):
    #         distill_teacher_to_student(teacher, student, train_loader, optimizer_student, nn.CrossEntropyLoss(), device)

    #     # Save the distilled student model
    #     student_save_path = config.student_model_path.format(i + 1)
    #     torch.save(student.state_dict(), student_save_path)
    #     print(f"Student {i + 1} saved to {student_save_path}")

    #     # Evaluate the student on the validation dataset
    #     print(f"\nEvaluating Student {i + 1}:")
    #     evaluate_with_metrics(student, test_loader, device, description=f"Student {i + 1}")

    # Step 7: Jointly train the MoE
    print("\nJoint Training of Mixture of Experts (MoE):")
    moe_model = MoE(students, router).to(device)
    optimizer_moe = AdamW(moe_model.parameters(), lr=config.lr)
    # optimizer_moe = AdamW(moe_model.parameters(), lr=config.lr, weight_decay=0.01)
    train_moe(moe_model, train_loader, optimizer_moe, nn.CrossEntropyLoss(), device, epochs=config.epochs)

    # Step 8: Evaluate the MoE model
    print("\nEvaluating Mixture of Experts (MoE):")
    evaluate_with_metrics(moe_model, test_loader, device, description="MoE")

    # Step 9: Visualize specialization
    print("\nVisualizing Specialization:")
    visualize_specialization(moe_model, test_loader, device, config.num_classes, config.num_students)


if __name__ == "__main__":
    main()
