"""
Complete Grad-CAM Visualization with Subspace Computation
This script includes ALL the necessary steps to generate Grad-CAM visualizations
using the DSN (Discriminative Subspace Network) approach.

FIXES: Includes the missing subspace computation that causes "subspace not defined" error
"""

import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

# Ensure these are defined before running:
# - model: Your trained model (loaded from MODEL_PATH_FS)
# - support_loader: DataLoader for support set
# - query_loader: DataLoader for query set
# - query_indices: List of query set indices
# - full_dataset: Full dataset object
# - class_names: List of class names
# - DEVICE: torch.device
# - SUBSPACE_DIM: Subspace dimension (N_SUPPORT - 1)
# - OUTPUT_DIR: Directory for saving outputs

# ============================================================================
# STEP 1: Define GradCAM Class
# ============================================================================

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _capture_gradients(self, grad):
        self.gradients = grad

    def _capture_activations(self, module, input, output):
        self.activations = output
        output.register_hook(self._capture_gradients)

    def _register_hooks(self):
        self.target_layer.register_forward_hook(self._capture_activations)

    def generate_heatmap(self, score, target_class):
        self.model.zero_grad()
        score.backward(retain_graph=True)

        if self.activations is None:
            print("WARNING: Activations became None during Grad-CAM.")
            return np.zeros((7, 7))

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])

        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(self.activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)

        max_val = torch.max(heatmap)
        if max_val == 0:
            return np.zeros(heatmap.shape)

        heatmap /= (max_val + 1e-8)

        return heatmap.cpu().detach().numpy()


# ============================================================================
# STEP 2: Define Helper Functions
# ============================================================================

def inverse_normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Reverse ImageNet normalization for visualization."""
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def plot_gradcam(original_img, heatmap, alpha=0.4):
    """Overlay heatmap on original image."""
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Convert to float for blending
    original_img_float = original_img.astype(np.float32)
    heatmap_float = heatmap.astype(np.float32)

    superimposed_img = heatmap_float * alpha + original_img_float * (1 - alpha)
    return np.clip(superimposed_img, 0, 255).astype(np.uint8)


def get_resnet_like_last_conv(model):
    """
    Find the last convolutional layer in a ResNet-like model.
    """
    # Try explicit path first (for ResNet50)
    try:
        candidate = model.layer4[-1].conv3
        print("Using explicit target layer: model.layer4[-1].conv3")
        return candidate
    except (AttributeError, IndexError):
        pass

    # Try encoder path (for models with encoder attribute)
    try:
        candidate = model.encoder[7][-1].conv3
        print("Using explicit target layer: model.encoder[7][-1].conv3")
        return candidate
    except (AttributeError, IndexError, TypeError):
        pass

    # Fallback: find last Conv2d
    last = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last = (name, module)

    if last is not None:
        print(f"Using last Conv2d found: {last[0]}")
        return last[1]

    raise RuntimeError("Could not find a Conv2d layer for Grad-CAM. "
                      "Please specify the target layer manually.")


# ============================================================================
# STEP 3: CRITICAL - Compute Subspaces (THIS WAS MISSING!)
# ============================================================================

def compute_subspaces(model, support_loader, device, subspace_dim=4):
    """
    Compute subspaces for each class using support samples via SVD.

    Returns:
        hyperplanes: Tensor [num_classes, feature_dim, subspace_dim]
        means: Tensor [num_classes, feature_dim]
        class_indices: List of class indices
        class_counts: Dict of sample counts per class
    """
    print("\nComputing subspaces from support set...")

    class_embeddings = defaultdict(list)
    class_counts = defaultdict(int)

    # Extract embeddings for all support samples
    with torch.no_grad():
        for imgs, labels in support_loader:
            imgs = imgs.to(device)
            embeddings = model(imgs)

            for emb, label in zip(embeddings, labels):
                class_embeddings[label.item()].append(emb)
                class_counts[label.item()] += 1

    all_hyperplanes = []
    all_means = []
    class_indices = sorted(class_embeddings.keys())

    if not class_indices:
        raise ValueError("No class embeddings found! Check your support_loader.")

    # Compute subspace for each class
    for cls_idx in class_indices:
        embs = class_embeddings[cls_idx]
        embs_tensor = torch.stack(embs)  # [num_samples, feature_dim]

        # Compute mean
        mean_vec = torch.mean(embs_tensor, dim=0)
        all_means.append(mean_vec)

        # Center the embeddings
        centered = embs_tensor - mean_vec.unsqueeze(0)

        # Validate subspace dimension
        sample_size = len(embs)
        num_dim = subspace_dim
        if sample_size < num_dim + 1:
            num_dim = sample_size - 1
            if cls_idx == class_indices[0]:
                print(f"Warning: Reduced subspace dim to {num_dim} for classes with {sample_size} samples")

        # SVD on transposed centered embeddings
        uu, s, v = torch.svd(centered.transpose(0, 1).double(), some=False)
        uu = uu.float()

        # Take first num_dim columns as hyperplane basis
        hyperplane = uu[:, :num_dim]  # [feature_dim, num_dim]
        all_hyperplanes.append(hyperplane)

    # Stack into tensors
    all_hyperplanes = torch.stack(all_hyperplanes, dim=0)
    all_means = torch.stack(all_means, dim=0)

    print(f"Computed subspaces for {len(class_indices)} classes")
    print(f"Hyperplanes shape: {all_hyperplanes.shape}")
    print(f"Means shape: {all_means.shape}")

    return all_hyperplanes, all_means, class_indices, class_counts


# ============================================================================
# STEP 4: Main Grad-CAM Visualization
# ============================================================================

def generate_gradcam_visualization(model, support_loader, query_indices, full_dataset,
                                   class_names, device, subspace_dim, output_dir):
    """
    Complete Grad-CAM visualization pipeline with subspace computation.
    """

    # CRITICAL: Compute subspaces FIRST
    hyperplanes, means, cls_indices, support_counts = compute_subspaces(
        model, support_loader, device, subspace_dim
    )

    # Build subspaces dictionary for easy access
    subspaces = {}
    for i, cls_idx in enumerate(cls_indices):
        subspaces[cls_idx] = {
            'mean': means[i],
            'basis': hyperplanes[i]
        }

    print(f"\nSubspaces dictionary created with {len(subspaces)} classes")

    # Setup Grad-CAM
    print("\nSelecting target layer for Grad-CAM...")
    target_layer = get_resnet_like_last_conv(model)
    grad_cam = GradCAM(model, target_layer)

    # Select images to visualize (one per class)
    print("Selecting images for visualization...")
    images_to_visualize = []
    visualized_classes = set()

    for idx in query_indices:
        _, label = full_dataset.samples[idx]
        if label not in visualized_classes:
            images_to_visualize.append(idx)
            visualized_classes.add(label)
        if len(visualized_classes) == len(class_names):
            break

    # Limit to 25 images for visualization
    images_to_visualize = images_to_visualize[:25]

    print(f"Generating Grad-CAM for {len(images_to_visualize)} images...")

    # Create visualization grid
    fig, axes = plt.subplots(5, 5, figsize=(20, 25))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i >= len(images_to_visualize):
            ax.axis('off')
            continue

        img_idx = images_to_visualize[i]
        img_tensor, true_label_idx = full_dataset[img_idx]

        # Prepare input with gradient tracking
        input_tensor = img_tensor.unsqueeze(0).to(device)
        input_tensor.requires_grad = True

        # Get query embedding
        query_embedding = model(input_tensor)

        # Find predicted class by computing distances to all subspaces
        distances = []
        with torch.no_grad():
            for cls_idx in sorted(subspaces.keys()):
                subspace = subspaces[cls_idx]
                mean = subspace['mean']
                basis = subspace['basis']

                centered_query = query_embedding.detach() - mean
                projection = torch.matmul(torch.matmul(centered_query, basis), basis.t())
                residual = centered_query - projection
                recon_error = torch.sqrt(torch.sum(residual.pow(2)) + 1e-12)
                distances.append(recon_error.item())

        pred_label_idx = np.argmin(distances)

        # Compute score for Grad-CAM (WITH gradients)
        pred_subspace = subspaces[pred_label_idx]
        centered_query = query_embedding - pred_subspace['mean']
        projection = torch.matmul(torch.matmul(centered_query, pred_subspace['basis']),
                                 pred_subspace['basis'].t())
        score_for_gradcam = -torch.sum((centered_query - projection).pow(2))

        # Generate heatmap
        heatmap = grad_cam.generate_heatmap(score=score_for_gradcam, target_class=pred_label_idx)

        # Prepare image for visualization
        img_vis = inverse_normalize(img_tensor.clone()).cpu().numpy().transpose(1, 2, 0)
        img_vis = np.clip(img_vis * 255, 0, 255).astype(np.uint8)

        # Create overlay
        overlay = plot_gradcam(img_vis.copy(), heatmap)

        # Display
        ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        ax.axis('off')

        # Set title with color coding
        true_label_name = class_names[true_label_idx].replace("_", " ").split("___")[-1]
        pred_label_name = class_names[pred_label_idx].replace("_", " ").split("___")[-1]
        is_correct = (pred_label_idx == true_label_idx)
        title_color = 'green' if is_correct else 'red'

        title = f"True: {true_label_name}\nPred: {pred_label_name}"
        ax.set_title(title, color=title_color, fontsize=12, pad=10)

    plt.tight_layout(pad=3.0)
    output_path = os.path.join(output_dir, 'gradcam_visualization_dsn.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\nGrad-CAM visualization complete! Saved to {output_path}")

    return subspaces


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    """
    To use this script, make sure you have defined:
    - model: Your trained model
    - support_loader: DataLoader for support set
    - query_indices: List of query indices
    - full_dataset: Full dataset object
    - class_names: List of class names
    - DEVICE: torch.device
    - SUBSPACE_DIM: Subspace dimension
    - OUTPUT_DIR: Output directory

    Then run:
    """

    # Example call (uncomment and adapt to your variables):
    # subspaces = generate_gradcam_visualization(
    #     model=model,
    #     support_loader=support_loader,
    #     query_indices=query_indices,
    #     full_dataset=full_dataset,
    #     class_names=class_names,
    #     device=DEVICE,
    #     subspace_dim=SUBSPACE_DIM,
    #     output_dir=OUTPUT_DIR
    # )

    print("Script loaded. Call generate_gradcam_visualization() with your data.")
