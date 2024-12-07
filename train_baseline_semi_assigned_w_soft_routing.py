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

class Config:
    in_channels = 3
    num_classes = 10
    batch_size = 64
    lr = 1e-3
    epochs = 20
    num_students = 3
    num_big_classes = num_students
    hidden_dim = 256
    temperature = 3.0
    alpha = 0.7
    teacher_model_path = "teacher.pth"
    student_model_path = "student_{}.pth"

config = Config()

def get_data_loaders():
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader


class TeacherModel(nn.Module):
    def __init__(self, num_classes=config.num_classes):
        super(TeacherModel, self).__init__()
        self.network = resnet18(pretrained=True)
        self.network.fc = nn.Linear(self.network.fc.in_features, num_classes)

    def forward(self, x):
        return self.network(x)

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
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_students)
        )
        self.temperature = 10.0  # Start with higher temperature

    def forward(self, x):
        logits = self.network(x)
        return F.softmax(logits / self.temperature, dim=1)

    def update_temperature(self, factor=0.95):
        self.temperature *= factor



class MoE(nn.Module):
    def __init__(self, students, gating_net):
        super(MoE, self).__init__()
        self.students = nn.ModuleList(students)
        self.gating_net = gating_net

    def forward(self, x, return_router_assignments=False):
        batch_size = x.size(0)
        # Get routing probabilities from the gating network
        gating_probs = self.gating_net(x.view(batch_size, -1))  # Shape: [batch_size, num_students]

        if self.training:
            # Soft Routing: Weighted sum of all experts
            expert_outputs = torch.stack([student(x) for student in self.students], dim=1)  # [batch_size, num_students, num_classes]
            gating_probs = gating_probs.unsqueeze(2)  # [batch_size, num_students, 1]
            outputs = (expert_outputs * gating_probs).sum(dim=1)  # [batch_size, num_classes]
            if return_router_assignments:
                best_experts = gating_probs.squeeze(2).argmax(dim=1)  # For monitoring purposes
                return outputs, best_experts
            return outputs
        else:
            # Hard Routing: Select top-1 expert
            best_experts = gating_probs.argmax(dim=1)  # [batch_size]
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

    print(f"Distill Loss: {total_loss / len(loader):.4f}")
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


def distill_teacher_to_router_with_epochs(teacher, router, loader, optimizer, criterion, device, big_class_map, epochs=5):
    teacher.eval()
    router.train()

    # Reverse the big class map for efficient lookup
    class_to_big_class = {c: bc for c, bc in big_class_map.items()}

    for epoch in range(epochs):
        total_loss = 0
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)

            with torch.no_grad():
                teacher_outputs = F.softmax(teacher(inputs) / config.temperature, dim=1)
                big_class_targets = torch.zeros(inputs.size(0), config.num_big_classes, device=device)
                for class_idx, big_class_idx in class_to_big_class.items():
                    big_class_targets[:, big_class_idx] += teacher_outputs[:, class_idx]

            # Router outputs
            router_outputs = router(inputs.view(inputs.size(0), -1))
            router_probs = F.softmax(router_outputs, dim=1)

            # Supervised assignment loss
            assignment_targets = torch.tensor(
                [class_to_big_class[label.item()] for label in targets],
                device=device,
                dtype=torch.long
            )
            supervised_loss = criterion(router_outputs, assignment_targets)

            # Diversity loss
            student_usage = router_probs.mean(dim=0)
            diversity_loss = torch.sum((student_usage - 1 / config.num_students) ** 2)

            # Distillation loss
            distill_loss = F.kl_div(
                F.log_softmax(router_outputs / config.temperature, dim=1),
                big_class_targets,
                reduction='batchmean'
            ) * (config.temperature ** 2)

            # Total loss
            total_loss_batch = distill_loss + 0.5 * supervised_loss + 0.01 * diversity_loss

            optimizer.zero_grad()
            total_loss_batch.backward()
            optimizer.step()

            total_loss += total_loss_batch.item()

        print(f"Epoch {epoch + 1}/{epochs} - Router Distillation Loss: {total_loss / len(loader):.4f}")

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
    """
    Train the MoE model (router and students) together.
    """
    moe_model.train()
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

            # Calculate accuracy
            total_loss += loss.item()
            correct += (outputs.argmax(1) == targets).sum().item()
            total_samples += targets.size(0)

        # Logging epoch metrics
        accuracy = correct / total_samples
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss / len(train_loader):.4f}, Accuracy: {accuracy:.4f}")

    return moe_model

def iterative_router_distillation(teacher, router, students, loader, optimizer_router, optimizer_students, device, big_class_map, num_rounds=10):
    teacher.eval()
    router.train()
    total_loss = 0

    class_to_big_class = {c: bc for c, bc in big_class_map.items()}

    for round_idx in range(num_rounds):
        print(f"Distillation Round {round_idx + 1}/{num_rounds}")
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)

            with torch.no_grad():
                teacher_outputs = F.softmax(teacher(inputs) / config.temperature, dim=1)
                big_class_targets = torch.zeros(inputs.size(0), config.num_big_classes, device=device)
                for class_idx, big_class_idx in class_to_big_class.items():
                    big_class_targets[:, big_class_idx] += teacher_outputs[:, class_idx]

            # Router training
            router_outputs = router(inputs.view(inputs.size(0), -1))
            distill_loss = F.kl_div(F.log_softmax(router_outputs / config.temperature, dim=1), big_class_targets, reduction='batchmean') * (config.temperature ** 2)

            assignment_targets = torch.tensor([class_to_big_class[label.item()] for label in targets], device=device, dtype=torch.long)

            supervised_loss = F.cross_entropy(router_outputs, assignment_targets)

            router_loss = distill_loss + 0.1 * supervised_loss
            optimizer_router.zero_grad()
            router_loss.backward()
            optimizer_router.step()

        # Optional: Fine-tune students after each round
        if students is not None:
            for student, optimizer_student in zip(students, optimizer_students):
                student.train()
                for inputs, targets in loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    optimizer_student.zero_grad()
                    outputs = student(inputs)
                    loss = F.cross_entropy(outputs, targets)
                    loss.backward()
                    optimizer_student.step()

        # Update Router Temperature
        router.update_temperature(factor=0.9)

    return router


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


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_data_loaders()

    # Step 1: Initialize and train the teacher model
    teacher = TeacherModel().to(device)
    print("Training the Teacher Model:")
    teacher = train_teacher(teacher, train_loader, test_loader, device, epochs=config.epochs, lr=config.lr)
    # teacher.load_state_dict(torch.load("teacher.pth", map_location=device, weights_only=True))
    # teacher.eval()  # Ensure the teacher is in evaluation mode

    # Step 2: Create the big class map from the teacher's logits
    print("\nCreating Big Class Map from Teacher:")
    big_class_map = create_big_class_map_from_teacher(teacher, train_loader, config.num_big_classes, device)
    print("Big Class Map:", big_class_map)

    # Step 3: Initialize the router
    router = GatingNetwork(num_students=config.num_students, input_dim=3 * 32 * 32).to(device)
    optimizer_router = AdamW(router.parameters(), lr=config.lr)

    # Step 4: Initialize the students
    print("\nInitializing Students:")
    students = [StudentModel().to(device) for _ in range(config.num_students)]
    student_optimizers = [AdamW(student.parameters(), lr=config.lr) for student in students]

    # Step 5: Perform iterative router distillation
    print("\nDistilling Teacher Knowledge into the Router:")
    criterion = nn.CrossEntropyLoss()
    router = distill_teacher_to_router_with_epochs(
        teacher=teacher,
        router=router,
        loader=train_loader,
        optimizer=optimizer_router,
        criterion=criterion,
        device=device,
        big_class_map=big_class_map,
        epochs=10
    )

    print("\nVisualizing Router Assignments:")
    visualize_router_assignments(router, test_loader, config.num_classes, config.num_students, device)

    # Step 6: Distill teacher knowledge into students
    print("\nDistilling Teacher Knowledge into Students:")
    for i, student in enumerate(students):
        optimizer_student = AdamW(student.parameters(), lr=config.lr)
        for epoch in range(config.epochs // 2):
            distill_teacher_to_student(teacher, student, train_loader, optimizer_student, nn.CrossEntropyLoss(), device)
        # student_path = "student_models/baseline_cnn_students/student_" + str(i) + ".pth"
        # student.load_state_dict(torch.load(student_path, map_location=device))
    print("Distilled student models loaded successfully.")

    # Step 7: Jointly train the MoE
    print("\nJoint Training of Mixture of Experts (MoE):")
    moe_model = MoE(students, router).to(device)
    optimizer_moe = AdamW(moe_model.parameters(), lr=config.lr)
    train_moe(moe_model, train_loader, optimizer_moe, nn.CrossEntropyLoss(), device, epochs=config.epochs)

    # Step 8: Evaluate the MoE model
    print("\nEvaluating Mixture of Experts (MoE):")
    evaluate_with_metrics(moe_model, test_loader, device, description="MoE")

    # Step 9: Visualize specialization
    print("\nVisualizing Specialization:")
    visualize_specialization(moe_model, test_loader, device, config.num_classes, config.num_students)


if __name__ == "__main__":
    main()
