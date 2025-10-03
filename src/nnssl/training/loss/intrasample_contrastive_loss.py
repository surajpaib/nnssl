import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class IntraSampleNTXentLoss(nn.Module):
    def __init__(
        self,
        temperature=0.1,
        permute=True,
        variance_weight=0.0,
        device=None,
    ):
        """
        IntraSample NTXent Loss for contrastive learning.

        Unlike standard NTXent which treats different images in a batch as negatives,
        this loss treats different crops from the same image as negatives (intra-sample).
        Within each crop, the two views are positives.

        Args:
            temperature: Temperature parameter for scaling similarities
            permute: Whether to compute loss across all crops or just first crop
            variance_weight: Weight for variance regularization (VICReg-style)
            device: Device to run computations on
        """
        super(IntraSampleNTXentLoss, self).__init__()
        self.temperature = temperature
        self.permute = permute
        self.variance_weight = variance_weight
        self.device = device if device is not None else torch.device("cpu")
        self.eps = 1e-8

        if abs(self.temperature) < self.eps:
            raise ValueError(f"Illegal temperature: abs({self.temperature}) < {self.eps}")

        self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, zis, zjs, batch_size, num_crops):
        """
        Forward pass of the IntraSampleNTXentLoss.

        Args:
            zis: First view embeddings [batch_size * num_crops, embedding_dim]
            zjs: Second view embeddings [batch_size * num_crops, embedding_dim]
            batch_size: Number of images in the batch
            num_crops: Number of crops per image

        Returns:
            tuple: (loss, accuracy)
        """
        # Ensure inputs are on the correct device
        zis = zis.to(self.device)
        zjs = zjs.to(self.device)

        # Normalize embeddings
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)

        # Reshape to [batch_size, num_crops, embedding_dim]
        zis_reshaped = zis.view(batch_size, num_crops, -1)
        zjs_reshaped = zjs.view(batch_size, num_crops, -1)

        inv_loss = 0
        total_correct = 0
        total_samples = 0

        labels = torch.zeros(1, dtype=torch.long, device=self.device)
        crop_range = range(num_crops) if self.permute else range(1)

        # Compute loss per sample
        for b in range(batch_size):
            per_sample_loss = 0
            for i in crop_range:
                # Get positive pair for current crop
                positive_0 = zis_reshaped[b, i : i + 1]  # [1, embedding_dim]
                positive_1 = zjs_reshaped[b, i : i + 1]  # [1, embedding_dim]

                # Build negatives from other crops of the same image
                negatives = []
                for j in range(num_crops):
                    if j != i:
                        negatives.extend([zis_reshaped[b, j], zjs_reshaped[b, j]])

                if negatives:  # Only process if we have negatives
                    negatives = torch.stack(negatives)  # [2*(num_crops-1), embedding_dim]

                    # Compute similarities
                    sim_pos = torch.mm(positive_0, positive_1.t())
                    sim_neg = torch.mm(positive_0, negatives.t())

                    # Compute logits
                    logits = torch.cat([sim_pos, sim_neg], dim=1) / self.temperature
                    per_sample_loss += self.cross_entropy(logits, labels)

                    # Track accuracy
                    total_correct += (torch.max(logits.detach(), dim=1)[1] == 0).sum().item()
                    total_samples += 1

            inv_loss += per_sample_loss / len(crop_range)

        inv_loss /= batch_size
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        # Add variance regularization if specified
        if self.variance_weight > 0:
            var_loss = 0.5 * (self._variance_loss(zis) + self._variance_loss(zjs))
            loss = (1 - self.variance_weight) * inv_loss + self.variance_weight * var_loss
            return loss, accuracy

        return inv_loss, accuracy

    def _variance_loss(self, x):
        """
        VICReg variance loss to prevent collapse.

        Args:
            x: Tensor with shape (batch_size * num_crops, embedding_dim)

        Returns:
            Variance loss value
        """
        return torch.mean(F.relu(1.0 - torch.sqrt(x.var(dim=0) + self.eps)))


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Example usage
    batch_size = 4
    num_crops = 3
    hidden_dim = 128

    # Simulate embeddings from batch_size images, each with num_crops crops
    z_i = torch.randn(batch_size * num_crops, hidden_dim).to(device)
    z_j = torch.randn(batch_size * num_crops, hidden_dim).to(device)

    loss_fn = IntraSampleNTXentLoss(
        temperature=0.1,
        permute=True,
        variance_weight=0.0,
        device=device,
    )

    loss, accuracy = loss_fn(z_i, z_j, batch_size, num_crops)
    print(f"Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")
