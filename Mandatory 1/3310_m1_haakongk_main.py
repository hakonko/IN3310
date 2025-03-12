from pathlib import Path

from ResNet import ResNet
from ResNetData import ResNetDataPreprocessor, ResNetDataset
from ResNetTrain import train_model, predict_softmax, compare_softmax, evaluate_on_test_set
from FeatureAnalysis import FeatureMapExtractor, SparsityAnalyzer
from plotting import plot_map_per_class, plot_losses

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from torchvision.models import resnet34, ResNet34_Weights

##############################################################################################
# Torch device and seed

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def set_seed(seed):

    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch GPU

set_seed(42)

##############################################################################################
# Tasks 1 a), b): CREATING AND VERIFYING DATA
BASE_PATH = Path(__file__).parent # this folder

# this is the directory of the zipped-up mandatory1_data folder
# also checking if it is actually there
DATASET_PATH = BASE_PATH / 'mandatory1_data'
if not DATASET_PATH.exists():
    raise FileNotFoundError(f"Cannot find {DATASET_PATH} - make sure that the unzipped 'mandatory_data' is in the folder of this program.")

# creating directories for the generated files
MODEL_PATH = BASE_PATH / 'models'
PLOTS_PATH = BASE_PATH / 'plots'

MODEL_PATH.mkdir(parents=True, exist_ok=True)
PLOTS_PATH.mkdir(parents=True, exist_ok=True)

SOFTMAX_CSV = "saved_softmax_scores.csv"
SOFTMAX_CSV_PRE = "pretrained_softmax_scores.csv"
linspc = '—' * 113

print(linspc)
print("Creating data")
preprocessor = ResNetDataPreprocessor(base_path=BASE_PATH, dataset_path=DATASET_PATH)

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

# creating data sets
train_dataset = ResNetDataset(preprocessor.annotations_file, BASE_PATH, split='train', transform=transform)
train_dataset_augm = ResNetDataset(preprocessor.annotations_file, BASE_PATH, split='train', transform=transform, augm_transform=augm_transform)
val_dataset = ResNetDataset(preprocessor.annotations_file, BASE_PATH, split='val', transform=transform)
test_dataset = ResNetDataset(preprocessor.annotations_file, BASE_PATH, split='test', transform=transform)

##############################################################################################
# Task 1 c): CREATING DATALOADERS

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
train_loader_augm = DataLoader(train_dataset_augm, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


##############################################################################################
# Tasks 2 a), b), c), d): TRAINING THREE DIFFERENT MODELS

##### Model 1 #####
print(linspc)
print("Starting training of model 1/3: ResNet34, CrossEntropyLoss, Adam, lr: 0.001, basic transforms\n")
class_names = preprocessor.get_class_names()
criterion = nn.CrossEntropyLoss()
model = ResNet(img_channels=3, num_layers=34, num_classes=len(class_names))
optimizer = optim.Adam(model.parameters(), lr=0.001)
file_path1 = MODEL_PATH / 'resnet1.pth'
train_accs1, val_acc1, map_scores1, class_accs1, train_losses1, val_losses1 = train_model(model, train_loader, val_loader, criterion, optimizer, file_path1, num_epochs=20)

##### Model 2 #####
print(linspc)
print("Starting training of model 2/3: ResNet34, CrossEntropyLoss, Adam, lr: 0.001, augmented transforms\n")
criterion = nn.CrossEntropyLoss()
model = ResNet(img_channels=3, num_layers=34, num_classes=len(class_names))
optimizer = optim.Adam(model.parameters(), lr=0.001)
file_path2 = MODEL_PATH / 'resnet2.pth'
train_acc2, val_acc2, map_scores2, class_accs2, train_losses2, val_losses2 = train_model(model, train_loader_augm, val_loader, criterion, optimizer, file_path2, num_epochs=20)

##### Model 3 #####
print(linspc)
print("Starting training of model 3/3: ResNet34, CrossEntropyLoss, SGD, lr: 0.005, basic transforms\n")
criterion = nn.CrossEntropyLoss()
model = ResNet(img_channels=3, num_layers=34, num_classes=len(class_names))
optimizer = optim.SGD(model.parameters(), lr=0.005)
file_path3 = MODEL_PATH / 'resnet3.pth'
train_acc3, val_acc3, map_scores3, class_accs3, train_losses3, val_losses3 = train_model(model, train_loader, val_loader, criterion, optimizer, file_path3, num_epochs=20)

# deciding on the best model
class_accs = [class_accs1, class_accs2, class_accs3]
map_scores = [map_scores1, map_scores2, map_scores3]
train_losses = [train_losses1, train_losses2, train_losses3]
val_losses = [val_losses1, val_losses2, val_losses3]
file_paths = [file_path1, file_path2, file_path3]

best_map_scores = [np.max(map_scores1), np.max(map_scores2), np.max(map_scores3)]
best_model = np.argmax(best_map_scores)

print(linspc)
print(f"Finished training. The best model was number {best_model + 1}, with mAP-score {np.max(best_map_scores):.4f}")

##### Plotting ######
plot_map_per_class(class_names, class_accs[best_model], map_scores[best_model], PLOTS_PATH, f'plot_map_scores{best_model + 1}.png')
plot_losses(PLOTS_PATH, f'plot_train_val_loss{best_model + 1}.png', train_losses[best_model], val_losses[best_model], figsize=(10, 5))
print('Training/Val plots saved.')

##############################################################################################
# Task 2 e): PREDICTING ON TEST SET 

# predicting on best model and saving softmax results
print(linspc)
print("Predicting on best model")
model.load_state_dict(torch.load(file_paths[best_model], map_location=device))
model = model.to(device)
predict_softmax(model, test_loader, SOFTMAX_CSV)
evaluate_on_test_set(model, test_loader, criterion)
compare_softmax(SOFTMAX_CSV, model, test_loader)

##############################################################################################
# Task 2 f): USING A PRETRAINED MODEL

# loading the model
print(linspc)
print("Loading pretrained model")
model = resnet34(weights = ResNet34_Weights.DEFAULT)

# changing the output layer to fit our classes
num_classes = len(class_names)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# setting loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# training
print(linspc)
print("Starting training on pretrained model: ResNet34, CrossEntropyLoss, Adam, lr: 0.001, basic transforms")
train_accs0, val_accs0, map_scores0, class_accs0, train_losses0, val_losses0 = train_model(
    model, train_loader, val_loader, criterion, optimizer, "pretrained_resnet34.pth", num_epochs=10)

# plotting training data
plot_map_per_class(class_names, class_accs0, map_scores0, PLOTS_PATH, 'plot_map_scores0.png')
plot_losses(PLOTS_PATH, 'plot_train_val_loss0.png', train_losses0, val_losses0)
print('Plots saved.')

# predicting on best model and saving softmax results
print(linspc)
print("Finished training. Predicting on model")
predict_softmax(model, test_loader, SOFTMAX_CSV_PRE)
evaluate_on_test_set(model, test_loader, criterion)
compare_softmax(SOFTMAX_CSV_PRE, model, test_loader)

##############################################################################################
# Tasks 3 a)-f):
#
# For a more thorough runthrough of this tasks, please refer to the 
# Jupyter Notebook file: feature_analysis.ipynb

# setting the layer names
layer_names = ['layer1', 'layer3', 'layer4']

print(linspc)
print("Feature analysis pre-test: Extracting feature maps")
# Setup data
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

# Extract feature maps
extractor = FeatureMapExtractor(model, layer_names)
feature_maps = extractor.extract_feature_maps(test_loader, num_images=5)
extractor.cleanup()

# Analyze activations
print(linspc)
print("Performing feature analysis")
module_names = ['layer1.0.relu', 'layer2.0.relu', 'layer3.0.relu', 'layer4.0.relu', 'relu']
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
analyzer = SparsityAnalyzer(model, module_names)
feature_stats = analyzer.analyze_activations(test_loader)
analyzer.print_statistics()
analyzer.cleanup()

print(linspc)
print("Program ended successfully")