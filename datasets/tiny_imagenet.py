import os
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder

from .base_dataset import BaseDataset


class TinyImageNet(BaseDataset):
    """
    Tiny-ImageNet-200 loader (ImageFolder-based), following the same style as CIFAR10 loader:
      - build ImageFolder datasets with transforms
      - iterate dataset directly and torch.stack all images
      - concat train + val (or train + test if you use a different split)
      - return x, y as numpy arrays

    Expected folder structure (ImageFolder format):
      cfg.path/
        train/
          <class_id>/
            images/*.JPEG  (or directly *.JPEG under class folder)
        val/
          <class_id>/
            images/*.JPEG  (or directly *.JPEG under class folder)

    NOTE:
      Official Tiny-ImageNet val is not ImageFolder-ready by default.
      You must reorganize val into class subfolders beforehand if needed.
    """
    def __init__(self, cfg, rank, num_clients, num_models, dataset_mode):
        super(TinyImageNet, self).__init__(cfg, rank, num_clients, num_models, dataset_mode)
        self._image_path = cfg.path
        self._get_train_aux_test_image_tensor()

    def _load_image(self):
        # --- transforms ---
        # If your pipeline expects 32x32 (same as CIFAR), keep Resize((32, 32)).
        # If you want to train at native 64x64, replace 32 with 64 and adjust your models accordingly.
        data_tf = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        train_dir = os.path.join(self._image_path, "train")
        val_dir = os.path.join(self._image_path, "val")

        if not os.path.isdir(train_dir):
            raise FileNotFoundError(f"Tiny-ImageNet train dir not found: {train_dir}")
        if not os.path.isdir(val_dir):
            raise FileNotFoundError(f"Tiny-ImageNet val dir not found: {val_dir}")

        train_dataset = ImageFolder(root=train_dir, transform=data_tf)
        val_dataset = ImageFolder(root=val_dir, transform=data_tf)

        if len(train_dataset) == 0:
            raise RuntimeError(f"Empty train dataset at {train_dir} (check ImageFolder structure).")
        if len(val_dataset) == 0:
            raise RuntimeError(f"Empty val dataset at {val_dir} (check ImageFolder structure).")

        # --- load all into memory (same pattern as your CIFAR10 loader) ---
        x_train = torch.stack([img for img, _ in train_dataset])
        y_train = torch.tensor(train_dataset.targets, dtype=torch.long)

        x_val = torch.stack([img for img, _ in val_dataset])
        y_val = torch.tensor(val_dataset.targets, dtype=torch.long)

        x = torch.cat([x_train, x_val], dim=0)
        y = torch.cat([y_train, y_val], dim=0)

        return x.numpy(), y.numpy()
