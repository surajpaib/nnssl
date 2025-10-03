"""
Example demonstrating the data flow and shapes for IntraSample SimCLR.

This example shows:
1. How SimCLRTransform creates overlapping crops from a volume
2. How to split the crops into reference and overlapping views
3. How embeddings flow through the IntraSampleNTXentLoss
4. All intermediate shapes at each step

Usage:
    python example_intrasample_simclr_shapes.py
"""

import numpy as np
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from nnssl.ssl_data.dataloading.simclr_transform import SimCLRTransform
from nnssl.training.loss.intrasample_contrastive_loss import IntraSampleNTXentLoss


class DummyEncoder(nn.Module):
    """Simple encoder to simulate feature extraction."""

    def __init__(self, input_channels=1, embedding_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(64, embedding_dim)
        )

    def forward(self, x):
        return self.encoder(x)


def print_shape(name, tensor, indent=0):
    """Helper to print tensor shapes with labels."""
    prefix = "  " * indent
    if isinstance(tensor, (np.ndarray, torch.Tensor)):
        print(f"{prefix}{name}: {tuple(tensor.shape)}")
    else:
        print(f"{prefix}{name}: {tensor}")


def main():
    # Configuration
    batch_size = 1
    num_crops = 32
    input_channels = 1
    volume_size = (192, 192, 64)
    crop_size = (64, 64, 64)
    min_overlap_ratio = 0.5
    embedding_dim = 128
    temperature = 0.1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("IntraSample SimCLR - Data Flow and Shapes Example")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of crops per image: {num_crops}")
    print(f"  Input volume size: {volume_size}")
    print(f"  Crop size: {crop_size}")
    print(f"  Minimum overlap ratio: {min_overlap_ratio}")
    print(f"  Embedding dimension: {embedding_dim}")
    print(f"  Temperature: {temperature}")
    print(f"  Device: {device}")

    # Step 1: Create input volume
    print("\n" + "=" * 80)
    print("STEP 1: Input Volume")
    print("=" * 80)
    input_volume = np.random.randn(batch_size, input_channels, *volume_size).astype(np.float32)
    print_shape("Input volume", input_volume)

    # Step 2: Apply SimCLR Transform
    print("\n" + "=" * 80)
    print("STEP 2: SimCLR Transform")
    print("=" * 80)
    transform = SimCLRTransform(
        crop_size=crop_size,
        aug="train",  # Use augmentations
        crop_count_per_image=num_crops,
        min_overlap_ratio=min_overlap_ratio,
        data_key="data"
    )

    batch_dict = transform(data=input_volume)

    print("\nTransform output keys:", list(batch_dict.keys()))
    print_shape("all_crops", batch_dict["all_crops"])
    print_shape("reference_crop_index", batch_dict["reference_crop_index"])
    print_shape("batch_size", batch_dict["batch_size"])
    print_shape("n_crops_per_image", batch_dict["n_crops_per_image"])

    all_crops = batch_dict["all_crops"]
    reference_crop_index = batch_dict["reference_crop_index"]

    print(f"\nKey information:")
    print(f"  Total crops in all_crops: {all_crops.shape[0]}")
    print(f"  Reference crop index (split point): {reference_crop_index}")
    print(f"  Expected: batch_size * num_crops = {batch_size} * {num_crops} = {batch_size * num_crops}")

    # Step 3: Split into reference and overlapping crops
    print("\n" + "=" * 80)
    print("STEP 3: Split Crops into Reference and Overlapping Views")
    print("=" * 80)
    reference_crops = all_crops[:reference_crop_index]
    overlapping_crops = all_crops[reference_crop_index:]

    print_shape("Reference crops (view 1)", reference_crops)
    print_shape("Overlapping crops (view 2)", overlapping_crops)

    print(f"\nInterpretation:")
    print(f"  Each view contains {batch_size * num_crops} crops")
    print(f"  For batch {batch_size}, there are {num_crops} crops per image")
    print(f"  Crops are paired: reference_crops[i] overlaps with overlapping_crops[i]")

    # Step 4: Convert to PyTorch and pass through encoder
    print("\n" + "=" * 80)
    print("STEP 4: Encode Crops to Embeddings")
    print("=" * 80)
    encoder = DummyEncoder(input_channels=input_channels, embedding_dim=embedding_dim).to(device)

    reference_crops_tensor = torch.from_numpy(reference_crops).float().to(device)
    overlapping_crops_tensor = torch.from_numpy(overlapping_crops).float().to(device)

    print("Before encoding:")
    print_shape("  Reference crops tensor", reference_crops_tensor, indent=1)
    print_shape("  Overlapping crops tensor", overlapping_crops_tensor, indent=1)

    with torch.no_grad():
        zis = encoder(reference_crops_tensor)
        zjs = encoder(overlapping_crops_tensor)

    print("\nAfter encoding:")
    print_shape("  zis (reference embeddings)", zis, indent=1)
    print_shape("  zjs (overlapping embeddings)", zjs, indent=1)

    # Step 5: Compute loss
    print("\n" + "=" * 80)
    print("STEP 5: Compute IntraSample Contrastive Loss")
    print("=" * 80)
    loss_fn = IntraSampleNTXentLoss(
        temperature=temperature,
        permute=True,  # Compute loss for all crops (not just first)
        variance_weight=0.0,  # No variance regularization
        device=device
    )

    loss, accuracy = loss_fn(
        zis=zis,
        zjs=zjs,
        batch_size=batch_size,
        num_crops=num_crops
    )

    print(f"\nLoss computation:")
    print(f"  Input embeddings shape: [{batch_size * num_crops}, {embedding_dim}]")
    print(f"  Reshaped internally to: [{batch_size}, {num_crops}, {embedding_dim}]")
    print(f"\n  Loss: {loss.item():.6f}")
    print(f"  Accuracy: {accuracy:.4f}")

    # Step 6: Explain the loss computation
    print("\n" + "=" * 80)
    print("STEP 6: Understanding the Loss")
    print("=" * 80)
    print(f"\nFor each of {batch_size} images:")
    print(f"  For each of {num_crops} crops:")
    print(f"    - POSITIVE pair: reference_crop[i] <-> overlapping_crop[i]")
    print(f"    - NEGATIVE pairs: crop[i] <-> all other {num_crops-1} crops from same image")
    print(f"    - Total negatives per crop: 2 * ({num_crops} - 1) = {2 * (num_crops - 1)}")
    print(f"      (2x because each other crop has reference AND overlapping views)")
    print(f"\n  Total comparisons per image: {num_crops} crops")
    print(f"  Total comparisons in batch: {batch_size * num_crops}")

    print(f"\nKey insight:")
    print(f"  Unlike standard SimCLR (negatives = other images in batch),")
    print(f"  IntraSample SimCLR uses negatives = other crops from SAME image")
    print(f"  This encourages learning global image-level representations")
    print(f"  rather than just local crop-level features.")

    # Step 7: Show a detailed example for one crop
    print("\n" + "=" * 80)
    print("STEP 7: Detailed Example for Crop 0 of Image 0")
    print("=" * 80)

    # Manually compute for crop 0 to show the process
    zis_normalized = torch.nn.functional.normalize(zis, dim=1)
    zjs_normalized = torch.nn.functional.normalize(zjs, dim=1)

    # Reshape to [batch_size, num_crops, embedding_dim]
    zis_reshaped = zis_normalized.view(batch_size, num_crops, -1)
    zjs_reshaped = zjs_normalized.view(batch_size, num_crops, -1)

    b = 0  # First batch
    i = 0  # First crop

    positive_0 = zis_reshaped[b, i:i+1]
    positive_1 = zjs_reshaped[b, i:i+1]

    print(f"\nCrop {i} from image {b}:")
    print_shape("  Positive view 1 (reference)", positive_0, indent=1)
    print_shape("  Positive view 2 (overlapping)", positive_1, indent=1)

    # Compute positive similarity
    sim_pos = torch.mm(positive_0, positive_1.t())
    print(f"\n  Positive similarity (before temperature): {sim_pos.item():.4f}")

    # Build negatives
    negatives = []
    for j in range(num_crops):
        if j != i:
            negatives.extend([zis_reshaped[b, j], zjs_reshaped[b, j]])

    negatives = torch.stack(negatives)
    print_shape("  Negatives (other crops)", negatives, indent=1)

    sim_neg = torch.mm(positive_0, negatives.t())
    print(f"  Negative similarities shape: {sim_neg.shape}")
    print(f"  Negative similarities (first 5): {sim_neg[0, :5].detach().cpu().numpy()}")

    logits = torch.cat([sim_pos, sim_neg], dim=1) / temperature
    print_shape("  Logits (positive + negatives, scaled by temp)", logits, indent=1)
    print(f"  Target label: 0 (first position is the positive)")

    print("\n" + "=" * 80)
    print("Example Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
