from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import torch
import os


class iNatDataset(Dataset):
    def __init__(
        self,
        root,
        transform,
        mode,
        num_samples=0,
        pred=[],
        probability=[],
        paths=[],
        num_class=1000,
    ):
        self.root = root
        self.transform = transform
        self.mode = mode
        self.train_labels = {}
        self.test_labels = {}
        self.val_labels = {}
        self.num_class = num_class

        train_path = f"{root}/train"
        val_path = f"{root}/val"
        test_path = f"{root}/test"
        print("Current Working Directory:", os.getcwd())
        for idx, genus in enumerate(sorted(os.listdir(train_path))):
            for image in os.listdir(f"{train_path}/{genus}"):
                self.train_labels[f"{train_path}/{genus}/{image}"] = int(idx)

        for idx, genus in enumerate(sorted(os.listdir(val_path))):
            for image in os.listdir(f"{val_path}/{genus}"):
                self.val_labels[f"{val_path}/{genus}/{image}"] = int(idx)

        for idx, genus in enumerate(sorted(os.listdir(test_path))):
            for image in os.listdir(f"{test_path}/{genus}"):
                self.test_labels[f"{test_path}/{genus}/{image}"] = int(idx)

        if mode == "all":
            train_imgs = list(self.train_labels.keys())
            random.shuffle(train_imgs)
            class_num = torch.zeros(num_class)
            self.train_imgs = []
            for impath in train_imgs:
                label = self.train_labels[impath]
                if (
                    class_num[label] < (num_samples / self.num_class)
                    and len(self.train_imgs) < num_samples
                ):
                    self.train_imgs.append(impath)
                    class_num[label] += 1
            random.shuffle(self.train_imgs)

        elif self.mode == "labeled":
            train_imgs = paths
            pred_idx = pred.nonzero()[0]
            self.train_imgs = [train_imgs[i] for i in pred_idx]
            self.probability = [probability[i] for i in pred_idx]
            print("%s data has a size of %d" % (self.mode, len(self.train_imgs)))

        elif self.mode == "unlabeled":
            train_imgs = paths
            pred_idx = (1 - pred).nonzero()[0]
            self.train_imgs = [train_imgs[i] for i in pred_idx]
            self.probability = [probability[i] for i in pred_idx]
            print("%s data has a size of %d" % (self.mode, len(self.train_imgs)))

        elif mode == "test":
            self.test_imgs = list(self.test_labels.keys())

        elif mode == "val":
            self.val_imgs = list(self.val_labels.keys())

    def __getitem__(self, index):
        if self.mode == "labeled":
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]
            prob = self.probability[index]
            image = Image.open(img_path).convert("RGB")
            img1 = self.transform(image)
            img2 = self.transform(image)
            return img1, img2, target, prob
        elif self.mode == "unlabeled":
            img_path = self.train_imgs[index]
            image = Image.open(img_path).convert("RGB")
            img1 = self.transform(image)
            img2 = self.transform(image)
            return img1, img2
        elif self.mode == "all":
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]
            try:
                image = Image.open(img_path).convert("RGB")
            except IOError:
                print(f"Warning: Corrupted image {img_path}")
                image = Image.new('RGB', (224, 224), color=(0, 0, 0))
            img = self.transform(image)
            return img, target, img_path
        elif self.mode == "test":
            img_path = self.test_imgs[index]
            target = self.test_labels[img_path]
            try:
                image = Image.open(img_path).convert("RGB")
            except IOError:
                print(f"Warning: Corrupted image {img_path}")
                image = Image.new('RGB', (224, 224), color=(0, 0, 0))
            img = self.transform(image)
            return img, target
        elif self.mode == "val":
            img_path = self.val_imgs[index]
            target = self.val_labels[img_path]
            try:
                image = Image.open(img_path).convert("RGB")
            except IOError:
                print(f"Warning: Corrupted image {img_path}")
                image = Image.new('RGB', (224, 224), color=(0, 0, 0))
            img = self.transform(image)
            return img, target

    def __len__(self):
        if self.mode == "test":
            return len(self.test_imgs)
        if self.mode == "val":
            return len(self.val_imgs)
        else:
            return len(self.train_imgs)


class iNatDataLoader:
    def __init__(
        self,
        root,
        batch_size,
        num_batches,
        num_workers,
        num_tasks,
        global_rank,
        distributed=False,
        current_epoch=0
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_batches = num_batches
        self.root = root
        self.sampler = None
        self.distributed = distributed
        self.current_epoch = current_epoch
        self.num_tasks = num_tasks
        self.global_rank = global_rank

        self.transform_train = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
                ),
                transforms.RandomErasing(p=0.5),
                transforms.Normalize(
                    (0.4266, 0.4126, 0.3965), (0.2399, 0.2279, 0.2207)
                ),
                transforms.Resize((224, 224), antialias=True),  # type: ignore
            ]
        )
        self.transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4266, 0.4126, 0.3965), (0.2399, 0.2279, 0.2207)
                ),
                transforms.Resize((224, 224), antialias=True),  # type: ignore
            ]
        )

    def run(self, mode, pred=[], prob=[], paths=[], num_classes=1000):
        if mode == "warmup":
            warmup_dataset = iNatDataset(
                self.root,
                transform=self.transform_train,
                mode="all",
                num_samples=self.num_batches * self.batch_size * 10,
                num_class=num_classes,
            )
            if self.distributed:
                self.sampler = DistributedSampler(warmup_dataset,
                                                  num_replicas=self.num_tasks,
                                                  rank=self.global_rank)
                self.sampler.set_epoch(self.current_epoch)
            warmup_loader = DataLoader(
                dataset=warmup_dataset,
                batch_size=self.batch_size,
                shuffle=(self.sampler is None),
                num_workers=self.num_workers,
                prefetch_factor=4,
                sampler=self.sampler,
            )
            return warmup_loader
        elif mode == "train":
            labeled_dataset = iNatDataset(
                self.root,
                transform=self.transform_train,
                mode="labeled",
                pred=pred,
                probability=prob,
                paths=paths,
                num_class=num_classes,
            )
            if self.distributed:
                self.sampler = DistributedSampler(labeled_dataset,
                                                  num_replicas=self.num_tasks,
                                                  rank=self.global_rank)
                self.sampler.set_epoch(self.current_epoch)
            labeled_loader = DataLoader(
                dataset=labeled_dataset,
                batch_size=self.batch_size,
                shuffle=(self.sampler is None),
                num_workers=self.num_workers,
                prefetch_factor=4,
                drop_last=True,
                sampler=self.sampler,
            )
            unlabeled_dataset = iNatDataset(
                self.root,
                transform=self.transform_train,
                mode="unlabeled",
                pred=pred,
                probability=prob,
                paths=paths,
                num_class=num_classes,
            )
            if self.distributed:
                self.sampler = DistributedSampler(unlabeled_dataset,
                                                  num_replicas=self.num_tasks,
                                                  rank=self.global_rank)
                self.sampler.set_epoch(self.current_epoch)
            unlabeled_loader = DataLoader(
                dataset=unlabeled_dataset,
                batch_size=self.batch_size,
                # shuffle=True, # apparently this is bad?? but was original
                shuffle=False,
                num_workers=self.num_workers,
                prefetch_factor=4,
                sampler=self.sampler
            )
            return labeled_loader, unlabeled_loader
        elif mode == "eval_train":
            eval_dataset = iNatDataset(
                self.root,
                transform=self.transform_test,
                mode="all",
                num_samples=self.num_batches * self.batch_size * 10,
                num_class=num_classes,
            )
            if self.distributed:
                self.sampler = DistributedSampler(eval_dataset,
                                                  num_replicas=self.num_tasks,
                                                  rank=self.global_rank)
                self.sampler.set_epoch(self.current_epoch)
            eval_loader = DataLoader(
                dataset=eval_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                prefetch_factor=4,
                sampler=self.sampler,
            )
            return eval_loader
        elif mode == "test":
            test_dataset = iNatDataset(
                self.root,
                transform=self.transform_test,
                mode="test",
                num_class=num_classes,
            )
            if self.distributed:
                self.sampler = DistributedSampler(test_dataset,
                                                  num_replicas=self.num_tasks,
                                                  rank=self.global_rank)
                self.sampler.set_epoch(self.current_epoch)
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                prefetch_factor=4,
                sampler=self.sampler,
            )
            return test_loader
        elif mode == "val":
            val_dataset = iNatDataset(
                self.root,
                transform=self.transform_test,
                mode="val",
                num_class=num_classes,
            )
            if self.distributed:
                self.sampler = DistributedSampler(val_dataset,
                                                  num_replicas=self.num_tasks,
                                                  rank=self.global_rank)
                self.sampler.set_epoch(self.current_epoch)
            val_loader = DataLoader(
                dataset=val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                prefetch_factor=4,
                sampler=self.sampler,
            )
            return val_loader
