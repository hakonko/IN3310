import torch

class FeatureMapExtractor:
    """
    Extracts feature maps from specified layers, using forward hooks
    
        Args:
            model: a ResNet model
            layer_names: the names of the layers that we will feature
    """

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
        """Registers forward hooks on the specified layers to get the feature maps"""

        for name in self.layer_names:
            layer = dict(self.model.named_children())[name]
            hook = layer.register_forward_hook(self._hook_function)
            self.hooks.append((name, hook))

    def _hook_function(self, module, input, output):
        """Captures the outpt feature maps from the hooked layers"""

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

                # stop after processing the wanted number of images
                if (i + 1) >= num_images:
                    break
        
        return self.feature_maps
    
    def cleanup(self):
        """Removes hooks to free up resources"""
        for _, hook in self.hooks:
            hook.remove()
        self.hooks = [] # clearing the list of hooks


class SparsityAnalyzer:
    """Analyzing the sparsity of activations (percentage of non-positive values) in the model layers"""

    def __init__(self, model, module_names):
        self.model = model
        self.module_names = module_names

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)

        self.feature_stats = {name: {"avg": 0.0, "count": 0} for name in module_names}
        self.hooks = []

        self._setup_hooks()
        self.processed_batches = {name: set() for name in module_names}

    def _setup_hooks(self):
        for name, module in self.model.named_modules():
            if name in self.module_names:

                # attaching a hook that calls _hook_function during forward passes
                hook = module.register_forward_hook(
                    lambda mod, inp, out, name = name: self._hook_function(mod, inp, out, name)
                )
                self.hooks.append(hook)

    def _hook_function(self, module, input, output, name):

        batch_id = getattr(self, 'current_batch_id', 0) # get the batch id
        
        if batch_id not in self.processed_batches[name]: # ensuring each bacth is processed once
            self.processed_batches[name].add(batch_id) # marking batch as processed
            
            feature_map = output.detach().cpu()
            total_elements = feature_map.numel() # total number of elements in the feature map
            non_positive_count = torch.sum(feature_map <= 0).item() # counting non-positive values
            percentage = (non_positive_count / total_elements) * 100 # good ol' percentage
            
            # compute averages and update count/statistics
            current_avg = self.feature_stats[name]['avg']
            current_count = self.feature_stats[name]['count']
            
            new_count = current_count + 1
            new_avg = (current_avg * current_count + percentage) / new_count
            
            self.feature_stats[name]['avg'] = new_avg
            self.feature_stats[name]['count'] = new_count

    def analyze_activations(self, dataloader, num_images=200):
        """Running sparsity analysis for a specified number of images

        Args:
            dataloader: DataLoader with input images
            num_images: Number of images to analyze

        Returns:
            Dictionary with sparsity statistics per layer
        """
        self.model.eval()
        
        with torch.no_grad():
            for i, (images, _) in enumerate(dataloader):

                self.current_batch_id = i
                
                images = images.to(self.device)
                self.model(images)
                
                # tracking number of processed images
                processed_images = (i + 1) * images.size(0)
                print(f'\rProcessed {processed_images} images', end='', flush=True) # printing on one line ;)
                
                # stopping when required number of images are processed
                if processed_images >= num_images:
                    print()
                    break
        
        return self.feature_stats
    
    def print_statistics(self):
        """Printing the sparsity statistics for each module we have anlyzed"""

        for name, item in self.feature_stats.items():
            avg = item['avg']
            count = item['count']
            print(f'Module {name}: Average non-positive values: {avg:.2f}%, Count: {count}')

    def cleanup(self):
        """Removing all hooks to free resources"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = [] # clearing the list of hooks