from pathlib import Path

import ResNet
from ResNetData import ResNetDataPreprocessor, ResNetDataset
from ResNetTrain import *
from plotting import plot_map_per_class, plot_losses

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from torchvision.models import resnet34, ResNet34_Weights

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def set_seed(seed):

    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch GPU

set_seed(42)

##### Creating data (and verifying no updates) ######

DATASET_PATH = Path('/mnt/e/ml_projects/IN3310/2025/tut_data/mandatory1_data/')
BASE_PATH = Path('/mnt/e/ml_projects/IN3310/2025/tut_data/oblig1/')
SOFTMAX_CSV = "saved_softmax_scores.csv"
SOFTMAX_CSV_PRE = "pretrained_softmax_scores.csv"
MODEL_PATH = '/mnt/e/ml_projects/IN3310/2025/tut_data/oblig1/resnet1.pth'

print("Creating data.")
preprocessor = ResNetDataPreprocessor(dataset_path=DATASET_PATH, base_path=BASE_PATH)

##### Defining transforms ######
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# data augmentation
augm_transform = transforms.Compose([
    transforms.RandomResizedCrop(150, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Creating data sets
train_dataset = ResNetDataset(preprocessor.annotations_file, BASE_PATH, split='train', transform=transform)
train_dataset_augm = ResNetDataset(preprocessor.annotations_file, BASE_PATH, split='train', transform=transform, augm_transform=augm_transform)
val_dataset = ResNetDataset(preprocessor.annotations_file, BASE_PATH, split='val', transform=transform)
test_dataset = ResNetDataset(preprocessor.annotations_file, BASE_PATH, split='test', transform=transform)

# Creating DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
train_loader_augm = DataLoader(train_dataset_augm, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


##### Training three models #####

##### Model 1 #####
print("\nStarting training of model 1/3: ResNet34, CrossEntropyLoss, Adam, lr: 0.001, basic transforms")
class_names = preprocessor.get_class_names()
criterion = nn.CrossEntropyLoss()
model = ResNet(img_channels=3, num_layers=34, num_classes=len(class_names))
optimizer = optim.Adam(model.parameters(), lr=0.001)
file_path = '/mnt/e/ml_projects/IN3310/2025/tut_data/oblig1/resnet1.pth'
train_accs1, val_acc1, map_scores1, class_accs1, train_losses1, val_losses1 = train_model(model, train_loader, val_loader, criterion, optimizer, file_path, num_epochs=20)

##### Model 2 #####
print("\nStarting training of model 2/3: ResNet34, CrossEntropyLoss, Adam, lr: 0.001, augmented transforms")
criterion = nn.CrossEntropyLoss()
model = ResNet(img_channels=3, num_layers=18, num_classes=len(class_names))
optimizer = optim.Adam(model.parameters(), lr=0.001)
file_path = '/mnt/e/ml_projects/IN3310/2025/tut_data/oblig1/resnet2.pth'
train_acc2, val_acc2, map_scores2, class_accs2, train_losses2, val_losses2 = train_model(model, train_loader_augm, val_loader, criterion, optimizer, file_path, num_epochs=20)

##### Model 3 #####
print("\nStarting training of model 3/3: ResNet34, CrossEntropyLoss, SGD, lr: 0.005, basic transforms")
criterion = nn.CrossEntropyLoss()
model = ResNet(img_channels=3, num_layers=34, num_classes=len(class_names))
optimizer = optim.SGD(model.parameters(), lr=0.005)
file_path = '/mnt/e/ml_projects/IN3310/2025/tut_data/oblig1/resnet3.pth'
train_acc3, val_acc3, map_scores3, class_accs3, train_losses3, val_losses3 = train_model(model, train_loader, val_loader, criterion, optimizer, file_path, num_epochs=20)

# deciding on best model
class_accs = [class_accs1, class_accs2, class_accs3]
map_scores = [map_scores1, map_scores2, map_scores3]
train_losses = [train_losses1, train_losses2, train_losses3]
val_losses = [val_losses1, val_losses2, val_losses3]

best_map_scores = [np.max(map_scores1), np.max(map_scores2), np.max(map_scores3)]
best_model = np.argmax(best_map_scores)

print(f"Finished training. The best model was number {best_model + 1}, with mAP-score {np.max(best_map_scores):.4f}")

##### Plotting ######
plot_map_per_class(class_names, class_accs[best_model], map_scores[best_model], BASE_PATH, f'plot_map_scores{best_model + 1}.png')
plot_losses(BASE_PATH, f'plot_train_val_loss{best_model + 1}.png', train_losses[best_model], val_losses[best_model], figsize=(10, 5))
print('Training/Val plots saved.')

##### Predicting on model and saving softmax results #####
print("Finished training. Predicting on best model")

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
predict_softmax(model, test_loader, SOFTMAX_CSV)
evaluate_on_test_set(model, test_loader, criterion)
compare_softmax(SOFTMAX_CSV, model, test_loader)

###### Using Pretrained Model ######
print("Loading pretrained model")
model = resnet34(weigths = ResNet34_Weights.DEFAULT)

# changing the output layer to fit our classes
num_classes = len(class_names)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# setting loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# training
print("Starting training on pretrained model: ResNet34, CrossEntropyLoss, Adam, lr: 0.0001, basic transforms")
train_accs0, val_accs0, map_scores0, class_accs0, train_losses0, val_losses0 = train_model(
    model, train_loader, val_loader, criterion, optimizer, "pretrained_resnet34.pth", num_epochs=10)

# plotting
plot_map_per_class(class_names, class_accs0, map_scores0, BASE_PATH, 'plot_map_scores0.png')
plot_losses(BASE_PATH, 'plot_train_val_loss0.png', train_losses0, val_losses0)

print('Plots saved.')

print("Finished training. Predicting on model")
predict_softmax(model, test_loader, SOFTMAX_CSV_PRE)
evaluate_on_test_set(model, test_loader, criterion)
compare_softmax(SOFTMAX_CSV_PRE, model, test_loader)