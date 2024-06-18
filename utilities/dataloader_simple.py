import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import glob

class iNatDataset(Dataset):
    def __init__(self, root, transform, mode, class_to_idx=None, unsupervised=False):
        self.root = os.path.join(root, mode)
        self.transform = transform
        self.unsupervised = unsupervised
        self.image_paths = []
        self.labels = []

        if not unsupervised:
            for genus in sorted(os.listdir(self.root)):
                class_id = class_to_idx[genus]
                for image_file in glob.glob(os.path.join(self.root, genus, '*.jpg')):
                    self.image_paths.append(image_file)
                    self.labels.append(class_id)
        else:
            for image_file in glob.glob(os.path.join(self.root, '**/*.jpg', recursive=True)):
                self.image_paths.append(image_file)
                self.labels.append(-1)  # Placeholder for unlabeled data

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[index]
        return (image, label) if not self.unsupervised else image

    def __len__(self):
        return len(self.image_paths)

class iNatDataLoader:
    def __init__(self, root, batch_size, num_workers):
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Create a consistent mapping of class labels
        self.genus_list = sorted(os.listdir(os.path.join(root, 'train')))
        self.class_to_idx = {genus: idx for idx, genus in enumerate(self.genus_list)}

        self.transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.RandomErasing(p=0.5),
            transforms.Normalize((0.4266, 0.4126, 0.3965), (0.2399, 0.2279, 0.2207)),
            transforms.Resize((224, 224))
        ])

        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4266, 0.4126, 0.3965), (0.2399, 0.2279, 0.2207)),
            transforms.Resize((224, 224))
        ])

    def get_loader(self, mode, unsupervised=False):
        if mode in ['train']:
            dataset = iNatDataset(self.root, self.transform_train, mode, self.class_to_idx, unsupervised)
            shuf = True
        else:
            dataset = iNatDataset(self.root, self.transform_test, mode, self.class_to_idx, unsupervised)
            shuf = False
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuf, num_workers=self.num_workers)
