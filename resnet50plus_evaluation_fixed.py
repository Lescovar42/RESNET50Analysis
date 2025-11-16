"""
Fixed ResNet50Plus Few-Shot Learning Evaluation Script for DATA_DIR_2
This script contains corrected implementations of the DSN evaluation pipeline.

FIXES APPLIED:
1. Model instantiation: pretrained=False when loading from checkpoint
2. Grad-CAM gradient flow: Score computation moved outside no_grad block
3. Distance metric consistency: Added sqrt() to match evaluation logic
4. Removed duplicate model calls
5. Added proper variable validation
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from collections import defaultdict
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
import torchvision.models as models

# --- CONFIGURATION (re-using global variables) ---

# Ensure these variables are defined before running this script:
# DATA_DIR_2: Path to the target dataset
# MODEL_PATH_PLUS: Path to the trained model checkpoint
# OUTPUT_DIR: Directory for saving outputs
# N_SUPPORT: Number of support samples per class (e.g., 5 or 15)
# N_QUERY: Number of query samples per class (e.g., 10)
# SUBSPACE_DIM: Subspace dimensionality (typically N_SUPPORT - 1)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Device: {DEVICE}")
print(f"Configuration: {N_SUPPORT}-shot, {N_QUERY} queries per class")
print(f"Subspace dimensionality: {SUBSPACE_DIM}")


# --- STEP 1: Load Class Information & Prepare Dataset for DATA_DIR_2 ---

print("\n" + "-"*80)
print(f"STEP 1: Preparing dataset for DATA_DIR_2: {DATA_DIR_2}...")
print("-"*80)

data_transforms = transforms.Compose([
    transforms.Resize((84, 84)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load full dataset from DATA_DIR_2
full_dataset_raw = datasets.ImageFolder(DATA_DIR_2, transform=data_transforms)

# Derive class names and mapping directly from DATA_DIR_2
all_classes_ds2_names = sorted(full_dataset_raw.classes)
print(f"Found {len(all_classes_ds2_names)} classes in {DATA_DIR_2}")

# Make all classes from DATA_DIR_2 novel classes
novel_classes_names = all_classes_ds2_names
base_classes_names = []  # No base classes for this scenario
num_novel_classes = len(novel_classes_names)

print(f"Number of novel classes: {len(novel_classes_names)}")
print(f"Novel classes: {novel_classes_names}")
print(f"Number of base classes: {len(base_classes_names)}")
print(f"Base classes: {base_classes_names}")

# Map class names to indices
class_to_idx_novel = {cls_name: i for i, cls_name in enumerate(novel_classes_names)}

# Filter dataset to only include novel classes for few-shot evaluation
filtered_samples_novel_ds2 = []
for path, cls_idx_raw in full_dataset_raw.samples:
    cls_name = full_dataset_raw.classes[cls_idx_raw]
    if cls_name in class_to_idx_novel:
        new_idx = class_to_idx_novel[cls_name]
        filtered_samples_novel_ds2.append((path, new_idx))

# Create a new dataset object for novel classes only
full_dataset_novel_ds2 = datasets.ImageFolder(DATA_DIR_2, transform=data_transforms)
full_dataset_novel_ds2.samples = filtered_samples_novel_ds2
full_dataset_novel_ds2.targets = [s[1] for s in filtered_samples_novel_ds2]
full_dataset_novel_ds2.classes = novel_classes_names
full_dataset_novel_ds2.class_to_idx = class_to_idx_novel

class_names_ds2 = novel_classes_names

print(f"Filtered dataset for novel classes: {len(full_dataset_novel_ds2)} images across {len(novel_classes_names)} classes")

# Verify class distribution for novel classes
class_counts_novel_ds2 = defaultdict(int)
for _, label in full_dataset_novel_ds2.samples:
    class_counts_novel_ds2[label] += 1

print(f"\nSample class distribution for novel classes:")
for cls_idx in sorted(class_counts_novel_ds2.keys())[:5]:
    print(f"  {novel_classes_names[cls_idx][:40]:40s}: {class_counts_novel_ds2[cls_idx]:4d} samples")
print("  ...")


# --- STEP 2: Split into Support and Query Sets (for Novel Classes) ---

print("\n" + "-"*80)
print("STEP 2: Creating support and query split for novel classes...")
print("-"*80)

def create_support_query_split_ds2(dataset, n_support=5, n_query=10, seed=42):
    """
    Split dataset into support and query sets for few-shot evaluation.
    """
    np.random.seed(seed)

    class_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):
        class_to_indices[label].append(idx)

    support_indices = []
    query_indices = []

    for cls_idx, indices in class_to_indices.items():
        indices = np.array(indices)
        np.random.shuffle(indices)

        total_needed = n_support + n_query
        # Ensure we have enough samples for support and query
        if len(indices) < total_needed:
            print(f"Warning: Class {cls_idx} ({dataset.classes[cls_idx]}) has only {len(indices)} samples, need {total_needed}.")
            n_sup = min(n_support, len(indices) // 2)
            n_qry = len(indices) - n_sup
            if n_qry == 0 and n_sup > 0:
                n_qry = n_sup
                n_sup = 0
            elif n_sup == 0 and n_qry == 0 and len(indices) > 0:
                n_qry = len(indices)
        else:
            n_sup = n_support
            n_qry = n_query

        support_indices.extend(indices[:n_sup])
        query_indices.extend(indices[n_sup:n_sup + n_qry])

    return support_indices, query_indices

support_indices_ds2, query_indices_ds2 = create_support_query_split_ds2(
    full_dataset_novel_ds2, n_support=N_SUPPORT, n_query=N_QUERY
)

print(f"Support set: {len(support_indices_ds2)} samples ({N_SUPPORT} per class)")
print(f"Query set: {len(query_indices_ds2)} samples ({N_QUERY} per class)")

# Create data loaders
support_dataset_ds2 = Subset(full_dataset_novel_ds2, support_indices_ds2)
query_dataset_ds2 = Subset(full_dataset_novel_ds2, query_indices_ds2)
support_loader_ds2 = DataLoader(support_dataset_ds2, batch_size=32, shuffle=False)
query_loader_ds2 = DataLoader(query_dataset_ds2, batch_size=32, shuffle=False)


# --- STEP 3: Define ResNet50Plus Model Architecture ---

print("\n" + "-"*80)
print("STEP 3: Defining ResNet50Plus model architecture...")
print("-"*80)

class Resnet50Plus(nn.Module):
    def __init__(self, pretrained=True, freeze_bn=True, out_dim=2048):
        super().__init__()

        base_model = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        )

        self.encoder = nn.Sequential(*list(base_model.children())[:-1])
        self.out_channels = out_dim

        self.dropout = nn.Dropout(0.5)  # 50% dropout

        self.normalize = True

        if freeze_bn:
            self._freeze_batchnorm()

    def _freeze_batchnorm(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, 1)  # -> (B, 2048)

        # Apply dropout
        if self.training:
            x = self.dropout(x)

        if self.normalize:
            x = nn.functional.normalize(x, p=2, dim=1)

        return x

# FIX #1: Load model WITHOUT pretrained weights since we're loading from checkpoint
model_ds2 = Resnet50Plus(pretrained=False)  # Don't waste time loading ImageNet weights
model_ds2.load_state_dict(torch.load(MODEL_PATH_PLUS, map_location=DEVICE))
model_ds2 = model_ds2.to(DEVICE)
model_ds2.eval()
print("ResNet50Plus model loaded successfully!")


# --- STEP 4: Compute Subspaces for Each Class ---

print("\n" + "-"*80)
print("STEP 4: Computing subspaces from support set...")
print("-"*80)

def compute_subspaces_ds2(model, support_loader, device, subspace_dim=4):
    """
    Compute subspaces for each class using support samples.
    """
    class_embeddings = defaultdict(list)
    class_counts = defaultdict(int)

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
        return torch.empty(0), torch.empty(0), [], {}

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
    all_hyperplanes = torch.stack(all_hyperplanes, dim=0)  # [num_classes, feature_dim, num_dim]
    all_means = torch.stack(all_means, dim=0)  # [num_classes, feature_dim]

    return all_hyperplanes, all_means, class_indices, class_counts

hyperplanes_ds2, means_ds2, class_indices_ds2, support_class_counts_ds2 = compute_subspaces_ds2(
    model_ds2, support_loader_ds2, DEVICE, subspace_dim=SUBSPACE_DIM
)

print(f"Computed subspaces for {len(class_indices_ds2)} classes")
print(f"Support samples per class (first 5):")
for i, cls_idx in enumerate(class_indices_ds2[:5]):
    print(f"  {novel_classes_names[cls_idx][:40]:40s}: {support_class_counts_ds2[cls_idx]:2d} samples")
print("  ...")


# --- STEP 5: Evaluate on Query Set ---

print("\n" + "-"*80)
print("STEP 5: Evaluating on query set using reconstruction error...")
print("-"*80)

def evaluate_dsn_ds2(model, query_loader, hyperplanes, means, class_indices, device):
    """
    Evaluate using DSN method: classify based on subspace projection distance.
    """
    all_preds = []
    all_targets = []
    all_probs = []

    eps = 1e-12

    if len(class_indices) == 0 or len(query_loader.dataset) == 0:
        print("No classes or query samples to evaluate. Skipping evaluation.")
        return np.array([]), np.array([]), np.array([])

    with torch.no_grad():
        for imgs, labels in query_loader:
            imgs = imgs.to(device)
            query_embeddings = model(imgs)

            batch_size = query_embeddings.shape[0]
            num_classes = hyperplanes.shape[0]

            similarities = []

            # For each class, compute projection distance
            for j in range(num_classes):
                h_plane_j = hyperplanes[j].unsqueeze(0).repeat(batch_size, 1, 1).to(device)
                tf_centered = (query_embeddings - means[j].expand_as(query_embeddings)).unsqueeze(-1)
                proj = torch.bmm(h_plane_j, torch.bmm(h_plane_j.transpose(1, 2), tf_centered))
                proj = torch.squeeze(proj, -1) + means[j].unsqueeze(0).repeat(batch_size, 1)

                diff = query_embeddings - proj
                query_loss = -torch.sqrt(torch.sum(diff * diff, dim=-1) + eps)  # Negative distance

                similarities.append(query_loss)

            similarities = torch.stack(similarities, dim=1).to(device)

            # Normalize by standard deviation (only if more than one class)
            if num_classes > 1:
                similarities = similarities / similarities.std(dim=1, keepdim=True).clamp_min(1e-6)

            # Higher similarity = better, so use argmax for prediction
            preds_idx = torch.argmax(similarities, dim=1)

            # Map back to actual class labels
            preds = torch.tensor([class_indices[p.item()] for p in preds_idx], device=device)

            # Convert to probabilities
            probs = torch.softmax(similarities, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(labels.numpy())
            all_probs.append(probs.cpu().numpy())

    return (np.concatenate(all_preds),
            np.concatenate(all_targets),
            np.concatenate(all_probs))

preds_ds2, targets_ds2, probs_ds2 = evaluate_dsn_ds2(
    model_ds2, query_loader_ds2, hyperplanes_ds2, means_ds2, class_indices_ds2, DEVICE
)


# --- STEP 6: Results and Analysis ---

print("\n" + "-"*80)
print(f"EVALUATION RESULTS FOR DATA_DIR_2 (NOVEL CLASSES): {N_SUPPORT}-SHOT, {N_QUERY} QUERIES PER CLASS")
print("-"*80)

if len(targets_ds2) == 0:
    print("No query samples were evaluated. Skipping accuracy and report generation.")
else:
    acc_ds2 = (preds_ds2 == targets_ds2).mean()
    print(f"\nOverall Query Set Accuracy: {acc_ds2*100:.2f}%")
    print(f"Total query samples: {len(targets_ds2)}")

    # Per-class accuracy
    class_correct_ds2 = defaultdict(int)
    class_total_ds2 = defaultdict(int)
    for pred, target in zip(preds_ds2, targets_ds2):
        class_total_ds2[target] += 1
        if pred == target:
            class_correct_ds2[target] += 1

    print(f"\n{'Class':<50} {'Accuracy':>10} {'Samples':>8}")
    print("-" * 70)
    for cls_idx in sorted(class_total_ds2.keys()):
        cls_acc = class_correct_ds2[cls_idx] / class_total_ds2[cls_idx] * 100
        cls_name = novel_classes_names[cls_idx].replace("_", " ")
        cls_name = cls_name[:47] + "..." if len(cls_name) > 50 else cls_name
        print(f"{cls_name:<50} {cls_acc:>9.1f}% {class_total_ds2[cls_idx]:>8d}")

    # Overall statistics
    print(f"\n{'-'*70}")
    print("SUMMARY STATISTICS")
    print(f"{'-'*70}")
    accuracies_ds2 = [class_correct_ds2[cls] / class_total_ds2[cls] for cls in class_total_ds2.keys()]
    print(f"Mean per-class accuracy: {np.mean(accuracies_ds2)*100:.2f}%")
    print(f"Std per-class accuracy:  {np.std(accuracies_ds2)*100:.2f}%")
    print(f"Min per-class accuracy:  {np.min(accuracies_ds2)*100:.2f}%")
    print(f"Max per-class accuracy:  {np.max(accuracies_ds2)*100:.2f}%")

    # Detailed classification report
    print(f"\n{'-'*70}")
    print("DETAILED CLASSIFICATION REPORT")
    print(f"{'-'*70}")
    print(classification_report(
        targets_ds2, preds_ds2,
        target_names=class_names_ds2,
        zero_division=0
    ))

print(f"\n{'-'*70}")
print("EVALUATION COMPLETE FOR DATA_DIR_2 (NOVEL CLASSES)!")
print(f"{'-'*70}")
print(f"Evaluated on {len(novel_classes_names)} novel classes total")


# --- STEP 7: Visualize Results ---

if len(targets_ds2) == 0:
    print("No query samples to visualize. Skipping visualization generation.")
else:
    print("\n" + "-"*80)
    print("STEP 7: Generating visualizations...")
    print("-"*80)

    # Confusion Matrix
    cm_ds2 = confusion_matrix(targets_ds2, preds_ds2)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_ds2, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names_ds2, yticklabels=class_names_ds2,
                cbar_kws={'label': 'Count'})
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("True", fontsize=12)
    plt.title(f"Confusion Matrix - {N_SUPPORT}-Shot (DATA_DIR_2 - Novel Classes)", fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix_ds2_novel.png'), dpi=150, bbox_inches='tight')
    plt.show()

    # Confidence Analysis
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    correct_mask_ds2 = preds_ds2 == targets_ds2
    correct_probs_ds2 = probs_ds2[correct_mask_ds2, targets_ds2[correct_mask_ds2]]
    incorrect_probs_ds2 = probs_ds2[~correct_mask_ds2, preds_ds2[~correct_mask_ds2]]

    if len(correct_probs_ds2) > 0 and not np.all(np.isnan(correct_probs_ds2)):
        axes[0].hist(correct_probs_ds2, bins=50, alpha=0.7, label='Correct', color='green')
    if len(incorrect_probs_ds2) > 0 and not np.all(np.isnan(incorrect_probs_ds2)):
        axes[0].hist(incorrect_probs_ds2, bins=50, alpha=0.7, label='Incorrect', color='red')

    axes[0].set_xlabel('Confidence')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Prediction Confidence Distribution (DATA_DIR_2 - Novel Classes)')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    class_confidences_ds2 = defaultdict(list)
    for pred, target, prob in zip(preds_ds2, targets_ds2, probs_ds2):
        class_confidences_ds2[target].append(prob[pred])

    class_mean_conf_ds2 = {cls: np.mean(confs) for cls, confs in class_confidences_ds2.items()}
    sorted_classes_ds2 = sorted(class_mean_conf_ds2.items(), key=lambda x: x[1])

    class_indices_sorted_ds2 = [x[0] for x in sorted_classes_ds2]
    class_confs_ds2 = [x[1] for x in sorted_classes_ds2]
    class_labels_ds2 = [novel_classes_names[i][:20] for i in class_indices_sorted_ds2]

    axes[1].barh(range(len(class_labels_ds2)), class_confs_ds2, color='steelblue')
    axes[1].set_yticks(range(len(class_labels_ds2)))
    axes[1].set_yticklabels(class_labels_ds2, fontsize=8)
    axes[1].set_xlabel('Mean Confidence')
    axes[1].set_title('Mean Prediction Confidence per Class (DATA_DIR_2 - Novel Classes)')
    axes[1].grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confidence_analysis_ds2_novel.png'), dpi=150, bbox_inches='tight')
    plt.show()

    # --- Grad-CAM Visualization ---

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
                print("  WARNING: Activations became None during Grad-CAM.")
                return np.zeros((7, 7))  # Return empty heatmap

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

    def inverse_normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        return tensor

    def plot_gradcam(original_img, heatmap, alpha=0.7):
        heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
        heatmap_normalized = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)

        original_img_float = original_img.astype(np.float32)
        heatmap_colored_float = heatmap_colored.astype(np.float32)

        superimposed_img = original_img_float * (1 - alpha) + heatmap_colored_float * alpha
        return np.clip(superimposed_img, 0, 255).astype(np.uint8)

    def get_resnet_like_last_conv_model_ds2(model):
        """
        Finds the last convolutional layer in a ResNet50Plus model.
        """
        try:
            # Accessing through model.encoder[7][-1].conv3 for the ResNet50 backbone
            candidate = model.encoder[7][-1].conv3
            print("Using explicit target layer: model.encoder[7][-1].conv3")
            return candidate
        except (IndexError, AttributeError):
            pass

        # Fallback: find the last Conv2d inside model.encoder
        last = None
        for name, module in model.encoder.named_modules():
            if isinstance(module, nn.Conv2d):
                last = (name, module)
        if last is not None:
            print("Using last Conv2d found in encoder:", last[0])
            return last[1]

        raise RuntimeError("Could not find a Conv2d in model.encoder to use for Grad-CAM.")

    print("\nSelecting target layer for Grad-CAM...")
    target_layer_ds2 = get_resnet_like_last_conv_model_ds2(model_ds2)
    grad_cam_ds2 = GradCAM(model_ds2, target_layer_ds2)

    print("Generating Grad-CAM visualizations...")

    images_to_visualize_ds2 = []
    unique_labels_in_query_ds2 = sorted(list(set(targets_ds2)))

    for label_to_find in unique_labels_in_query_ds2:
        for i, original_idx in enumerate(query_indices_ds2):
            _, original_label = full_dataset_novel_ds2.samples[original_idx]
            if original_label == label_to_find:
                images_to_visualize_ds2.append(original_idx)
                break

    # Limit to 25 images for the plot
    images_to_visualize_ds2 = images_to_visualize_ds2[:25]

    fig, axes = plt.subplots(5, 5, figsize=(20, 25))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i >= len(images_to_visualize_ds2):
            ax.axis('off')
            continue

        img_idx = images_to_visualize_ds2[i]
        img_tensor, true_label_idx = full_dataset_novel_ds2[img_idx]

        # FIX #4: Remove duplicate model call - only use one with gradient tracking
        input_tensor = img_tensor.unsqueeze(0).to(DEVICE)
        input_tensor.requires_grad = True

        # Compute embedding with gradients for Grad-CAM
        query_embedding_grad = model_ds2(input_tensor)

        # FIX #2 & #3: Move score computation OUTSIDE no_grad block
        # First, find predicted class with no gradients
        distances = []
        with torch.no_grad():
            for j in range(len(class_indices_ds2)):
                h_plane_j = hyperplanes_ds2[j]
                mean_j = means_ds2[j]

                tf_centered = query_embedding_grad.detach() - mean_j
                projection_on_basis = torch.matmul(tf_centered, h_plane_j)
                reconstructed_centered = torch.matmul(projection_on_basis, h_plane_j.t())

                residual = tf_centered - reconstructed_centered
                # FIX #3: Add sqrt() to match evaluation logic
                recon_error = torch.sqrt(torch.sum(residual.pow(2)) + 1e-12)
                distances.append(recon_error.item())

        pred_idx_relative = np.argmin(distances)
        pred_label_idx = class_indices_ds2[pred_idx_relative]

        # FIX #2: NOW compute score WITH gradients for backpropagation
        h_plane_pred = hyperplanes_ds2[pred_idx_relative]
        mean_pred = means_ds2[pred_idx_relative]

        tf_centered_pred = query_embedding_grad - mean_pred
        projection_on_basis_pred = torch.matmul(tf_centered_pred, h_plane_pred)
        reconstructed_centered_pred = torch.matmul(projection_on_basis_pred, h_plane_pred.t())

        residual_pred = tf_centered_pred - reconstructed_centered_pred
        score_for_gradcam = -torch.sum(residual_pred.pow(2))  # Negative for gradient ascent

        # Generate heatmap with proper gradient flow
        heatmap = grad_cam_ds2.generate_heatmap(score=score_for_gradcam, target_class=pred_label_idx)

        # Visualize
        img_vis = inverse_normalize(img_tensor.clone()).cpu().numpy().transpose(1, 2, 0)
        img_vis = np.clip(img_vis * 255, 0, 255).astype(np.uint8)

        overlay = plot_gradcam(img_vis.copy(), heatmap, alpha=0.7)

        ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        ax.axis('off')

        true_label_name = novel_classes_names[true_label_idx].replace("_", " ")
        pred_label_name = novel_classes_names[pred_label_idx].replace("_", " ")
        is_correct = (pred_label_idx == true_label_idx)
        title_color = 'green' if is_correct else 'red'

        title = f"True: {true_label_name}\nPred: {pred_label_name}"
        ax.set_title(title, color=title_color, fontsize=12, pad=10)

    plt.tight_layout(pad=3.0)
    plt.savefig(os.path.join(OUTPUT_DIR, 'gradcam_visualization_ds2_novel.png'), dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\nGrad-CAM visualization complete! Results saved to {OUTPUT_DIR}")

print("\n" + "="*80)
print("ALL FIXES APPLIED SUCCESSFULLY!")
print("="*80)
print("\nFixes included:")
print("  1. Model instantiation with pretrained=False")
print("  2. Grad-CAM gradient flow (score computation outside no_grad)")
print("  3. Distance metric consistency (added sqrt)")
print("  4. Removed duplicate model calls")
print("  5. Proper variable validation")
print("="*80)
