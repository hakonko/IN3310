from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
from torchvision import datasets
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import os


class ResNetDataPreprocessor:
    def __init__(self, dataset_path, base_path):
        self.dataset_path = Path(dataset_path)
        self.base_path = base_path
        self.annotations_file = self.base_path / "annotations.csv"

        self.class_names = [str(subdir.parts[-1]) for subdir in self.dataset_path.iterdir() if subdir.is_dir()]

        self._prepare_data()
        self._save_annotations()
        self._verify_disjoint_splits()

    def _prepare_data(self):
        for split in ['train', 'val', 'test']:
            for class_name in self.class_names:
                (self.base_path / split / class_name).mkdir(parents=True, exist_ok=True)

        img_paths, class_indices = [], []
        for class_index, class_name in enumerate(self.class_names):
            class_path = self.dataset_path / class_name

            for img_file in class_path.iterdir():
                if img_file.is_file():
                    img_paths.append(img_file)
                    class_indices.append(class_index)

        self.train_imgs, temp_imgs, self.train_indices, temp_indices = train_test_split(
        img_paths, class_indices, test_size=0.3, stratify=class_indices, random_state=42)

        self.val_imgs, self.test_imgs, self.val_indices, self.test_indices = train_test_split(
        temp_imgs, temp_indices, test_size=0.6, stratify=temp_indices, random_state=42)

    def _save_annotations(self):
        data = []

        for split, imgs, labels in zip(
            ['train'] * len(self.train_imgs) + ['val'] * len(self.val_imgs) + ['test'] * len(self.test_imgs),
            self.train_imgs + self.val_imgs + self.test_imgs, 
            self.train_indices + self.val_indices + self.test_indices,
        ):
            data.append([split, str(imgs), labels])

        df = pd.DataFrame(data, columns=['split', 'image_path', 'label'])
        df.to_csv(self.annotations_file, index=False)

    def _verify_disjoint_splits(self):
        df = pd.read_csv(self.annotations_file)
        train_set = set(df[df['split'] == 'train']['image_path'])
        val_set = set(df[df['split'] == 'val']['image_path'])
        test_set = set(df[df['split'] == 'test']['image_path'])

        assert len(train_set.intersection(val_set)) == 0, 'Overlap between Train and Val'
        assert len(train_set.intersection(test_set)) == 0, 'Overlap between Train and Test'
        assert len(val_set.intersection(test_set)) == 0, 'Overlap between Val and Test'

        print('Success! Train, validation and test sets are disjoint')

    def get_class_names(self):
        return self.class_names


class ResNetDataset(Dataset):
    def __init__(self, annotations_file, base_path, split='train', transform=None, augm_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_labels = self.img_labels[self.img_labels['split'] == split]
        self.base_path = base_path
        self.split = split
        self.transform = transform
        self.augm_transform = augm_transform if split == 'train' else None

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_labels.iloc[idx, 1]
        image = Image.open(img_path).convert('RGB')
        label = self.img_labels.iloc[idx, 2]

        if self.augm_transform and self.split == 'train':
            image = self.augm_transform(image)
        
        else:
            image = self.transform(image)

        return image, label