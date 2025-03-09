from pathlib import Path

from ResNetData import ResNetDataPreprocessor, ResNetDataset
from ResNetTrain import *
from plotting import plot_losses

import torch

from torch.utils.data import DataLoader
from torchvision import transforms

from torchvision.models import resnet34, ResNet34_Weights

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image

import matplotlib.pyplot as plt
import os

import random


class FeatureMapExtractor:
    def __init__(self, model, layer_names):
        self.model = model
        self.layer_names = layer_names

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)

        self.feature_maps = {}
        self.hooks = []

        # stting up hooks
        self._setup_hooks()

    def _setup_hooks(self):
        for name in self.layer_names:
            layer = dict(self.model.named_children())[name]
            hook = layer.register_forward_hook(self._hook_function)
            self.hooks.append((name, hook))

    def _hook_function(self, module, input, output):
        for name, _ in self.hooks:
            if module == dict(self.model.named_children())[name]:
                self.feature_maps[name] = output.detach().cpu()

    def extract_feature_maps(self, dataloader, num_images=5):
        """
        Extracting feature maps for a chosen number of images.

        Args:
            dataloader: DataLoader containing images
            num_images: number of images
        Returns:
            Dictionary of feature maps for each layer
        """

        self.model.eval()

        with torch.no_grad():
            for i, (image, _) in enumerate(dataloader):
                image = image.to(self.device)
                self.model(image)

                print(f"Feature map stored for image {i + 1}")

                if (i + 1) >= num_images:
                    break
        
        return self.feature_maps
    
    def cleanup(self):
        for _, hook in self.hooks:
            hook.remove()
        self.hooks = []

class SparcityAnalyzer:
    def __init__(self, model, module_names):
        self.model = model
        self.module_names = module_names

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)

        self.feature_stats = {name: {"avg": 0.0, "count": 0} for name in module_names}
        self.hooks = []

        self._setup_hooks()

    def _setup_hooks(self):
        for name, module in self.model.named_modules():
            if name in self.module_names:
                hook = module.register_forward_hook(
                    lambda mod, inp, out, name = name: self._hook_function(mod, inp, out, name)
                )
                self.hooks.append(hook)

    def _hook_function(self, module, input, output, name):
        if name in self.module_names:
            feature_map = output.detach().cpu()
            total_elements = feature_map.numel()
            non_positive_count = torch.sum(feature_map <= 0).item()
            percentage = (non_positive_count / total_elements) * 100

            # updating the average using the formula: mt+1 = (mt * nt + ustep * nstep) / (nt + nstep)
            current_avg = self.feature_stats[name]['avg']
            current_count = self.feature_stats[name]['count']

            new_count = current_count + 1
            new_avg = (current_avg * current_count + percentage) / new_count

            # updating stats
            self.feature_stats[name]['avg'] = new_avg
            self.feature_stats[name]['count'] = new_count

    def analyze_activations(self, dataloader, num_images=200):

        self.model.eval()

        with torch.no_grad():
            for i, (images, _) in enumerate(dataloader):
                images = images.to(self.device)
                self.model(images)

                processed_images = (i + 1) * images.size(0)
                print(f'Processed {processed_images} images')

                if processed_images >= num_images:
                    break

        return self.feature_stats
    
    def print_statistics(self):
        for name, item in self.feature_stats.items():
            avg = item['avg']
            count = item['count']
            print(f'Module {name}: Average non-positive values: {avg:.2f}%, Count: {count}')

    def cleanup(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


if __name__ == "__main__":
    from torchvision.models import resnet34, ResNet34_Weights
    from ResNetData import ResNetDataPreprocessor, ResNetDataset
    import torchvision.transforms as transforms
    
    # Setup paths
    DATASET_PATH = Path('/mnt/e/ml_projects/IN3310/2025/tut_data/mandatory1_data/')
    BASE_PATH = Path('/mnt/e/ml_projects/IN3310/2025/tut_data/oblig1/')
    
    # Load model
    model = resnet34(weights=ResNet34_Weights.DEFAULT)
    
    # Define transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Setup data
    preprocessor = ResNetDataPreprocessor(dataset_path=DATASET_PATH, base_path=BASE_PATH)
    test_dataset = ResNetDataset(preprocessor.annotations_file, BASE_PATH, split="test", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    # Extract feature maps
    layer_names = ['layer1', 'layer3', 'layer4']
    extractor = FeatureMapExtractor(model, layer_names)
    feature_maps = extractor.extract_feature_maps(test_loader, num_images=5)
    extractor.cleanup()
    
    # Analyze activations
    module_names = ['layer1.0.relu', 'layer2.0.relu', 'layer3.0.relu', 'layer4.0.relu', 'relu']
    batch_loader = DataLoader(test_dataset, batch_size=200, shuffle=True)
    analyzer = SparcityAnalyzer(model, module_names)
    feature_stats = analyzer.analyze_activations(batch_loader)
    analyzer.print_statistics()
    analyzer.cleanup()

