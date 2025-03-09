import torch

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
        self.processed_batches = {name: set() for name in module_names}

    def _setup_hooks(self):
        for name, module in self.model.named_modules():
            if name in self.module_names:
                hook = module.register_forward_hook(
                    lambda mod, inp, out, name = name: self._hook_function(mod, inp, out, name)
                )
                self.hooks.append(hook)

    def _hook_function(self, module, input, output, name):

        batch_id = getattr(self, 'current_batch_id', 0)
        
        if batch_id not in self.processed_batches[name]:
            self.processed_batches[name].add(batch_id)
            
            feature_map = output.detach().cpu()
            total_elements = feature_map.numel()
            non_positive_count = torch.sum(feature_map <= 0).item()
            percentage = (non_positive_count / total_elements) * 100
            
            current_avg = self.feature_stats[name]['avg']
            current_count = self.feature_stats[name]['count']
            
            new_count = current_count + 1
            new_avg = (current_avg * current_count + percentage) / new_count
            
            self.feature_stats[name]['avg'] = new_avg
            self.feature_stats[name]['count'] = new_count

    def analyze_activations(self, dataloader, num_images=200):
        self.model.eval()
        
        with torch.no_grad():
            for i, (images, _) in enumerate(dataloader):

                self.current_batch_id = i
                
                images = images.to(self.device)
                self.model(images)
                
                processed_images = (i + 1) * images.size(0)
                print(f'\rProcessed {processed_images} images', end='', flush=True)
                
                if processed_images >= num_images:
                    print()
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

