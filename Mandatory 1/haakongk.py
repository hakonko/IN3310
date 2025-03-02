from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

import torch.optim as optim
from ResNet import ResNet

from sklearn.metrics import accuracy_score, average_precision_score


def create_data():
    dataset = Path('/mnt/e/ml_projects/IN3310/2025/tut_data/mandatory1_data/')

    classes = [str(subdir.parts[-1]) for subdir in dataset.iterdir() if subdir.is_dir()]

    base_path = Path('/mnt/e/ml_projects/IN3310/2025/tut_data/oblig1/')

    # the directories we want to create
    dirs = ['train', 'val', 'test']

    for dir_name in dirs:
        # creating a path string
        dir_path = base_path / dir_name
        dir_path.mkdir(parents=True, exist_ok=True) # creating directory

        # creating subdirectories of class names
        for class_name in classes:
            class_path = base_path / dir_name / class_name
            class_path.mkdir(parents=True, exist_ok=True)

        img_paths = [] # container for image paths
        class_indices = [] # container for class indices

        for class_index, class_name in enumerate(classes):
            class_path = dataset / class_name
            for img_file in class_path.iterdir():
                if img_file.is_file():
                    img_paths.append(img_file)
                    class_indices.append(class_index)

        train_imgs, temp_imgs, train_indices, temp_indices = train_test_split(
        img_paths, class_indices, test_size=0.3, stratify=class_indices, random_state=42
        )

        val_imgs, test_imgs, val_indices, test_indices = train_test_split(
        temp_imgs, temp_indices, test_size=0.6, stratify=temp_indices, random_state=42
        )

        copy_images(train_imgs, train_indices, 'train')
        copy_images(val_imgs, val_indices, 'val')
        copy_images(test_imgs, test_indices, 'test')

        return train_imgs, train_indices, val_imgs, val_indices, test_imgs, test_indices


def copy_images(img_paths, class_indices, split_name):
    
    for img_path, class_index in zip(img_paths, class_indices):
        target_dir = base_path / split_name / classes[class_index]
        target_file = target_dir / img_path.name

        # copying the file if it's not already there
        if not target_file.exists():
            shutil.copy(img_path, target_file)

def verify_no_duplicates(train_imgs, val_imgs, test_imgs):
    train_set = set(train_imgs)
    val_set = set(val_imgs)
    test_set = set(test_imgs)

    # using intersection to check for data overlaps
    assert len(train_set.intersection(val_set)) == 0, 'Overlap between Train and Val'
    assert len(train_set.intersection(test_set)) == 0, 'Overlap between Train and Test'
    assert len(val_set.intersection(test_set)) == 0, 'Overlap between Val and Test'

def train_model(model, train_loader, val_loader, criterion, optimizer, lr, num_epochs=10):
    model = model.to(device)
    optimizer = optimizer(model.parameters(), lr=lr)

    train_accs = []
    val_accs = []

    for epoch in range(num_epochs):
        model.train() # putting model into training mode
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad() # zeroing out the optimizer
            
            outputs = model(images)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_accuracy = correct / total
        val_accuracy, _, _ = evaluate_model(model, val_loader)

        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}")

    return train_accs, val_accs

        
def evaluate_model(model, dataloader):
    model.eval() # put model in evaluation mode

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_probs = torch.cat(all_probs)

    accuracy = accuracy_score(all_labels, all_preds)

    ap_scores = []
    for i in range(all_probs.shape[1]):
        binary_labels = (all_labels == i).float()

        ap = average_precision_score(
            binary_labels.cpu().numpy(),
            all_probs[:, i].cpu().numpy()
        )
        ap_scores.append(ap)

    map_score = sum(ap_scores) / len(ap_scores)
    
    return accuracy, ap_scores, map_score



if __name__ == "__main__":
    
    base_path = Path('/mnt/e/ml_projects/IN3310/2025/tut_data/oblig1/')
    
    device = torch.device('cuda') if torch.cuda.is_available else 'cpu'

    train_imgs, train_indices, val_imgs, val_indices, test_imgs, test_indices = create_data()
    verify_no_duplicates(train_imgs, val_imgs, test_imgs)

    train_dataset = datasets.ImageFolder(root=base_path / 'train', transform=transform)
    val_dataset = datasets.ImageFolder(root=base_path / 'val', transform=transform)
    test_dataset = datasets.ImageFolder(root=base_path / 'test', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(), # converts to a torch tensor
        transforms.Normalize((0.5,), (0.5,)) # normalizes to [-1, 1]
    ])

    criterion = nn.CrossEntropyLoss()
    model = ResNet(img_channels=3, num_layers=34, num_classes=len(classes))
    train_acc2, val_acc2 = train_model(model, train_loader, val_loader, criterion, optim.Adam, 0.001, 20)
