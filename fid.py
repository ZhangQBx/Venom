# from torchmetrics.image.fid import FrechetInceptionDistance
# import torch

# fid = FrechetInceptionDistance(feature=2048)
# fid.update(dataset1_images, real=True)
# fid.update(dataset2_images, real=False)
# score = fid.compute()
# print("FID:", score.item())

import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, ImageFolder
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
# --- Change: import torchmetrics instead of scipy and inception_v3 ---
from torchmetrics.image.fid import FrechetInceptionDistance
import numpy as np  # Required by collate_fn


# --- 1. Add collate_fn to convert PIL images to uint8 tensors ---
def collate_fn(batch):
    """
    Convert a batch of (PIL image, label) pairs to (uint8 tensor, label tensor).
    """
    # 1. Convert PIL images to a NumPy array of shape [N, H, W, C]
    images = [np.array(item[0]) for item in batch]
    images_np = np.stack(images)

    # 2. Convert to a uint8 tensor of shape [N, C, H, W]
    images_tensor = torch.from_numpy(images_np).permute(0, 3, 1, 2)

    # Keep labels for completeness, although this script does not use them
    labels = [item[1] for item in batch]
    labels_tensor = torch.tensor(labels)

    return images_tensor, labels_tensor


# --- 2. Data loading and preprocessing ---
def get_data_loader(dataset_name, data_root="./data", batch_size=64):
    """
    Create a DataLoader for the specified dataset.
    Convert all images to the 3x299x299 format expected by InceptionV3.
    ---
    Return PIL images for collate_fn to convert into uint8 tensors.
    """

    # --- Change: remove ToTensor and Normalize ---
    inception_transform = transforms.Compose([
        transforms.Resize((299, 299), antialias=True),  # Antialiasing improves image quality
        transforms.Grayscale(num_output_channels=3),  # Convert single-channel MNIST to 3 channels
    ])

    print(f"Loading dataset: {dataset_name}")
    try:
        if dataset_name == 'cifar10':
            dataset = CIFAR10(root=data_root, train=False, download=True, transform=inception_transform)
        elif dataset_name == 'cifar100':
            dataset = CIFAR100(root=data_root, train=False, download=True, transform=inception_transform)
        elif dataset_name == 'mnist':
            dataset = MNIST(root=data_root, train=False, download=True, transform=inception_transform)
        elif dataset_name == 'tiny_imagenet':
            path = os.path.join("../datasets/image/", 'tiny-imagenet-200', 'val')
            if not os.path.exists(path):
                print(f"Error: Tiny ImageNet 'val' directory not found at {path}.")
                print("Ensure 'tiny-imagenet-200.zip' has been extracted into './data/'.")
                return None
            dataset = ImageFolder(root=path, transform=inception_transform)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {e}")
        return None

    # --- Change: add collate_fn ---
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn  # Use the custom collate function
    )
    return loader


# --- 3. Revised main function ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Load CIFAR-10 reference images ---
    print("\n--- Processing CIFAR-10 (reference) ---")
    cifar10_loader = get_data_loader('cifar10')
    if cifar10_loader is None: return

    # --- Initialize FID with the default settings ---
    # Defaults: feature=2048, normalize=False (expects uint8 in [0, 255])
    fid_metric = FrechetInceptionDistance().to(device)

    print("Computing features for CIFAR-10 (Real) images...")
    for images, _ in tqdm(cifar10_loader, desc="Processing CIFAR-10..."):
        # --- Ensure images have uint8 dtype ---
        images = images.to(device, dtype=torch.uint8)
        fid_metric.update(images, real=True)

    # --- 2. Prepare the OOD dataset list ---
    ood_datasets = ['cifar100', 'tiny_imagenet', 'mnist']
    results = {}

    for ood_name in ood_datasets:
        print(f"\n--- Processing {ood_name} (Fake) ---")
        ood_loader = get_data_loader(ood_name)
        if ood_loader is None:
            results[ood_name] = float('inf')  # Mark as failed
            continue

        print(f"Computing features for {ood_name} (Fake) images...")
        for images, _ in tqdm(ood_loader, desc=f"Processing {ood_name}..."):
            # --- Ensure images have uint8 dtype ---
            images = images.to(device, dtype=torch.uint8)
            fid_metric.update(images, real=False)

        # --- 3. Compute FID ---
        print(f"Computing FID(CIFAR-10 vs {ood_name})...")
        score = fid_metric.compute()
        results[ood_name] = score.item()

        # --- 4. Reset fake-image statistics for the next iteration ---
        fid_metric.reset_real_features = False  # Preserve real-image features
        fid_metric.reset_fake_features = True  # Reset only fake-image features
        fid_metric.reset()

    # --- 5. Print final results ---
    print("\n==============================================")
    print(f"FID score (lower means more similar) - Reference: CIFAR-10")
    print("==============================================")

    print(f"FID(CIFAR-10 vs CIFAR-100): {results.get('cifar100', 'N/A'):.2f}")
    print(f"FID(CIFAR-10 vs Tiny ImageNet): {results.get('tiny_imagenet', 'N/A'):.2f}")
    print(f"FID(CIFAR-10 vs MNIST):     {results.get('mnist', 'N/A'):.2f}")
    print("==============================================")


if __name__ == "__main__":
    main()


