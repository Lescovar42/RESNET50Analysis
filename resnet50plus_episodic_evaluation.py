"""
ResNet50Plus Episodic Few-Shot Learning Evaluation
100 Episodes with Statistical Analysis

This script performs episodic evaluation (100 episodes) for few-shot learning,
providing robust statistical measures of model performance.
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
from tqdm import tqdm
import scipy.stats as stats
import torchvision.models as models

# --- CONFIGURATION ---

# Ensure these variables are defined:
# DATA_DIR_2: Path to dataset
# MODEL_PATH_PLUS: Path to trained model checkpoint
# OUTPUT_DIR: Directory for saving outputs
# N_SUPPORT: Number of support samples per class (e.g., 5 or 15)
# N_QUERY: Number of query samples per class (e.g., 10)
# SUBSPACE_DIM: Subspace dimensionality (typically N_SUPPORT - 1)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_EPISODES = 100  # Number of episodes to run
CONFIDENCE_LEVEL = 0.95  # For confidence intervals

print(f"Device: {DEVICE}")
print(f"Configuration: {N_SUPPORT}-shot, {N_QUERY} queries per class")
print(f"Subspace dimensionality: {SUBSPACE_DIM}")
print(f"Number of episodes: {NUM_EPISODES}")
print(f"Confidence level: {CONFIDENCE_LEVEL*100}%")


# --- STEP 1: Load Dataset ---

print("\n" + "="*80)
print(f"STEP 1: Preparing dataset for DATA_DIR_2: {DATA_DIR_2}...")
print("="*80)

data_transforms = transforms.Compose([
    transforms.Resize((84, 84)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load full dataset
full_dataset_raw = datasets.ImageFolder(DATA_DIR_2, transform=data_transforms)
all_classes_names = sorted(full_dataset_raw.classes)

print(f"Found {len(all_classes_names)} classes in {DATA_DIR_2}")
print(f"Total images: {len(full_dataset_raw)}")

# Create class to index mapping
class_to_idx = {cls_name: i for i, cls_name in enumerate(all_classes_names)}

# Filter dataset
filtered_samples = []
for path, cls_idx_raw in full_dataset_raw.samples:
    cls_name = full_dataset_raw.classes[cls_idx_raw]
    if cls_name in class_to_idx:
        new_idx = class_to_idx[cls_name]
        filtered_samples.append((path, new_idx))

# Create filtered dataset
full_dataset = datasets.ImageFolder(DATA_DIR_2, transform=data_transforms)
full_dataset.samples = filtered_samples
full_dataset.targets = [s[1] for s in filtered_samples]
full_dataset.classes = all_classes_names
full_dataset.class_to_idx = class_to_idx

class_names = all_classes_names

# Verify class distribution
class_counts = defaultdict(int)
for _, label in full_dataset.samples:
    class_counts[label] += 1

print(f"\nClass distribution (first 5 classes):")
for cls_idx in sorted(class_counts.keys())[:5]:
    print(f"  {class_names[cls_idx][:40]:40s}: {class_counts[cls_idx]:4d} samples")
print("  ...")


# --- STEP 2: Define ResNet50Plus Model ---

print("\n" + "="*80)
print("STEP 2: Defining ResNet50Plus model architecture...")
print("="*80)

class Resnet50Plus(nn.Module):
    def __init__(self, pretrained=True, freeze_bn=True, out_dim=2048):
        super().__init__()

        base_model = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        )

        self.encoder = nn.Sequential(*list(base_model.children())[:-1])
        self.out_channels = out_dim
        self.dropout = nn.Dropout(0.5)
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
        x = torch.flatten(x, 1)

        if self.training:
            x = self.dropout(x)

        if self.normalize:
            x = nn.functional.normalize(x, p=2, dim=1)

        return x

# Load model
model = Resnet50Plus(pretrained=False)
model.load_state_dict(torch.load(MODEL_PATH_PLUS, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()
print("ResNet50Plus model loaded successfully!")


# --- STEP 3: Define Episodic Evaluation Functions ---

print("\n" + "="*80)
print("STEP 3: Defining episodic evaluation functions...")
print("="*80)

def create_episode_split(dataset, n_support=5, n_query=10, seed=None):
    """
    Create a single episode by randomly sampling support and query sets.

    Args:
        dataset: Full dataset
        n_support: Number of support samples per class
        n_query: Number of query samples per class
        seed: Random seed for reproducibility

    Returns:
        support_indices: List of support sample indices
        query_indices: List of query sample indices
    """
    if seed is not None:
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
        if len(indices) < total_needed:
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


def compute_subspaces(model, support_loader, device, subspace_dim=4):
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
        embs_tensor = torch.stack(embs)

        # Compute mean
        mean_vec = torch.mean(embs_tensor, dim=0)
        all_means.append(mean_vec)

        # Center embeddings
        centered = embs_tensor - mean_vec.unsqueeze(0)

        # Validate subspace dimension
        sample_size = len(embs)
        num_dim = min(subspace_dim, sample_size - 1)

        # SVD
        uu, s, v = torch.svd(centered.transpose(0, 1).double(), some=False)
        uu = uu.float()

        # Take first num_dim columns
        hyperplane = uu[:, :num_dim]
        all_hyperplanes.append(hyperplane)

    all_hyperplanes = torch.stack(all_hyperplanes, dim=0)
    all_means = torch.stack(all_means, dim=0)

    return all_hyperplanes, all_means, class_indices, class_counts


def evaluate_episode(model, query_loader, hyperplanes, means, class_indices, device):
    """
    Evaluate a single episode using DSN method.
    """
    all_preds = []
    all_targets = []

    eps = 1e-12

    if len(class_indices) == 0 or len(query_loader.dataset) == 0:
        return np.array([]), np.array([])

    with torch.no_grad():
        for imgs, labels in query_loader:
            imgs = imgs.to(device)
            query_embeddings = model(imgs)

            batch_size = query_embeddings.shape[0]
            num_classes = hyperplanes.shape[0]

            similarities = []

            for j in range(num_classes):
                h_plane_j = hyperplanes[j].unsqueeze(0).repeat(batch_size, 1, 1).to(device)
                tf_centered = (query_embeddings - means[j].expand_as(query_embeddings)).unsqueeze(-1)
                proj = torch.bmm(h_plane_j, torch.bmm(h_plane_j.transpose(1, 2), tf_centered))
                proj = torch.squeeze(proj, -1) + means[j].unsqueeze(0).repeat(batch_size, 1)

                diff = query_embeddings - proj
                query_loss = -torch.sqrt(torch.sum(diff * diff, dim=-1) + eps)

                similarities.append(query_loss)

            similarities = torch.stack(similarities, dim=1).to(device)

            if num_classes > 1:
                similarities = similarities / similarities.std(dim=1, keepdim=True).clamp_min(1e-6)

            preds_idx = torch.argmax(similarities, dim=1)
            preds = torch.tensor([class_indices[p.item()] for p in preds_idx], device=device)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(labels.numpy())

    return np.concatenate(all_preds), np.concatenate(all_targets)


def run_single_episode(model, dataset, episode_num, n_support, n_query, subspace_dim, device):
    """
    Run a single episode and return results.
    """
    # Create episode split
    support_indices, query_indices = create_episode_split(
        dataset, n_support=n_support, n_query=n_query, seed=episode_num
    )

    # Create data loaders
    support_dataset = Subset(dataset, support_indices)
    query_dataset = Subset(dataset, query_indices)
    support_loader = DataLoader(support_dataset, batch_size=32, shuffle=False)
    query_loader = DataLoader(query_dataset, batch_size=32, shuffle=False)

    # Compute subspaces
    hyperplanes, means, class_indices, _ = compute_subspaces(
        model, support_loader, device, subspace_dim=subspace_dim
    )

    # Evaluate
    preds, targets = evaluate_episode(
        model, query_loader, hyperplanes, means, class_indices, device
    )

    # Compute metrics
    if len(targets) == 0:
        return None

    overall_acc = (preds == targets).mean()

    # Per-class accuracy
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    for pred, target in zip(preds, targets):
        class_total[target] += 1
        if pred == target:
            class_correct[target] += 1

    per_class_acc = {cls: class_correct[cls] / class_total[cls]
                     for cls in class_total.keys() if class_total[cls] > 0}

    return {
        'episode': episode_num,
        'overall_accuracy': overall_acc,
        'per_class_accuracy': per_class_acc,
        'predictions': preds,
        'targets': targets
    }


# --- STEP 4: Run Episodic Evaluation ---

print("\n" + "="*80)
print(f"STEP 4: Running {NUM_EPISODES} episodes...")
print("="*80)

episode_results = []

for episode in tqdm(range(NUM_EPISODES), desc="Evaluating episodes"):
    result = run_single_episode(
        model=model,
        dataset=full_dataset,
        episode_num=episode,
        n_support=N_SUPPORT,
        n_query=N_QUERY,
        subspace_dim=SUBSPACE_DIM,
        device=DEVICE
    )

    if result is not None:
        episode_results.append(result)

print(f"\nCompleted {len(episode_results)} episodes")


# --- STEP 5: Compute Statistics ---

print("\n" + "="*80)
print("STEP 5: Computing statistics across episodes...")
print("="*80)

# Overall accuracy statistics
overall_accuracies = [r['overall_accuracy'] for r in episode_results]
mean_acc = np.mean(overall_accuracies)
std_acc = np.std(overall_accuracies)
sem_acc = stats.sem(overall_accuracies)

# Confidence interval
ci = stats.t.interval(
    CONFIDENCE_LEVEL,
    len(overall_accuracies) - 1,
    loc=mean_acc,
    scale=sem_acc
)

print(f"\nOVERALL ACCURACY STATISTICS ({NUM_EPISODES} episodes)")
print(f"{'='*70}")
print(f"Mean accuracy:        {mean_acc*100:.2f}%")
print(f"Std deviation:        {std_acc*100:.2f}%")
print(f"Min accuracy:         {np.min(overall_accuracies)*100:.2f}%")
print(f"Max accuracy:         {np.max(overall_accuracies)*100:.2f}%")
print(f"Median accuracy:      {np.median(overall_accuracies)*100:.2f}%")
print(f"95% Confidence Int:   [{ci[0]*100:.2f}%, {ci[1]*100:.2f}%]")

# Per-class statistics across episodes
per_class_stats = defaultdict(list)
for result in episode_results:
    for cls_idx, acc in result['per_class_accuracy'].items():
        per_class_stats[cls_idx].append(acc)

print(f"\n{'='*70}")
print(f"PER-CLASS ACCURACY STATISTICS")
print(f"{'='*70}")
print(f"{'Class':<45} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
print(f"{'-'*70}")

class_stats_summary = []
for cls_idx in sorted(per_class_stats.keys()):
    accs = per_class_stats[cls_idx]
    cls_name = class_names[cls_idx].replace("_", " ")[:42]

    mean_cls = np.mean(accs)
    std_cls = np.std(accs)
    min_cls = np.min(accs)
    max_cls = np.max(accs)

    print(f"{cls_name:<45} {mean_cls*100:>7.1f}% {std_cls*100:>7.1f}% {min_cls*100:>7.1f}% {max_cls*100:>7.1f}%")

    class_stats_summary.append({
        'class': cls_name,
        'class_idx': cls_idx,
        'mean': mean_cls,
        'std': std_cls,
        'min': min_cls,
        'max': max_cls
    })


# --- STEP 6: Visualizations ---

print("\n" + "="*80)
print("STEP 6: Generating visualizations...")
print("="*80)

# Create output directory if needed
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Figure 1: Overall Accuracy Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
axes[0].hist(np.array(overall_accuracies) * 100, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
axes[0].axvline(mean_acc * 100, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_acc*100:.2f}%')
axes[0].axvline(ci[0] * 100, color='orange', linestyle=':', linewidth=2, label=f'95% CI: [{ci[0]*100:.2f}%, {ci[1]*100:.2f}%]')
axes[0].axvline(ci[1] * 100, color='orange', linestyle=':', linewidth=2)
axes[0].set_xlabel('Accuracy (%)', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title(f'Overall Accuracy Distribution ({NUM_EPISODES} Episodes)', fontsize=14)
axes[0].legend()
axes[0].grid(alpha=0.3)

# Box plot
axes[1].boxplot([np.array(overall_accuracies) * 100], labels=['Overall Accuracy'])
axes[1].set_ylabel('Accuracy (%)', fontsize=12)
axes[1].set_title('Accuracy Box Plot', fontsize=14)
axes[1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'episodic_overall_accuracy.png'), dpi=150, bbox_inches='tight')
plt.show()

# Figure 2: Per-Class Accuracy
fig, ax = plt.subplots(figsize=(12, max(8, len(per_class_stats) * 0.3)))

sorted_classes = sorted(class_stats_summary, key=lambda x: x['mean'], reverse=True)
class_labels = [x['class'][:30] for x in sorted_classes]
class_means = [x['mean'] * 100 for x in sorted_classes]
class_stds = [x['std'] * 100 for x in sorted_classes]

y_pos = np.arange(len(class_labels))
ax.barh(y_pos, class_means, xerr=class_stds, alpha=0.7, color='steelblue', capsize=3)
ax.set_yticks(y_pos)
ax.set_yticklabels(class_labels, fontsize=9)
ax.set_xlabel('Mean Accuracy (%) ± Std', fontsize=12)
ax.set_title(f'Per-Class Accuracy Across {NUM_EPISODES} Episodes', fontsize=14)
ax.grid(alpha=0.3, axis='x')
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'episodic_per_class_accuracy.png'), dpi=150, bbox_inches='tight')
plt.show()

# Figure 3: Episode-wise accuracy trend
fig, ax = plt.subplots(figsize=(14, 6))

episode_nums = [r['episode'] for r in episode_results]
episode_accs = [r['overall_accuracy'] * 100 for r in episode_results]

ax.plot(episode_nums, episode_accs, alpha=0.5, color='steelblue', linewidth=1, label='Episode Accuracy')
ax.axhline(mean_acc * 100, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_acc*100:.2f}%')
ax.fill_between(episode_nums, ci[0] * 100, ci[1] * 100, alpha=0.2, color='orange', label=f'95% CI')
ax.set_xlabel('Episode Number', fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title(f'Episode-wise Accuracy Trend ({NUM_EPISODES} Episodes)', fontsize=14)
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'episodic_accuracy_trend.png'), dpi=150, bbox_inches='tight')
plt.show()


# --- STEP 7: Save Results ---

print("\n" + "="*80)
print("STEP 7: Saving results...")
print("="*80)

# Save summary statistics
summary = {
    'num_episodes': NUM_EPISODES,
    'n_support': N_SUPPORT,
    'n_query': N_QUERY,
    'mean_accuracy': mean_acc,
    'std_accuracy': std_acc,
    'min_accuracy': np.min(overall_accuracies),
    'max_accuracy': np.max(overall_accuracies),
    'median_accuracy': np.median(overall_accuracies),
    'ci_lower': ci[0],
    'ci_upper': ci[1],
    'confidence_level': CONFIDENCE_LEVEL
}

# Save to CSV
pd.DataFrame([summary]).to_csv(
    os.path.join(OUTPUT_DIR, 'episodic_summary.csv'), index=False
)

# Save per-class statistics
pd.DataFrame(class_stats_summary).to_csv(
    os.path.join(OUTPUT_DIR, 'episodic_per_class_stats.csv'), index=False
)

# Save all episode results
episode_data = [{
    'episode': r['episode'],
    'overall_accuracy': r['overall_accuracy']
} for r in episode_results]
pd.DataFrame(episode_data).to_csv(
    os.path.join(OUTPUT_DIR, 'episodic_all_results.csv'), index=False
)

print(f"\nResults saved to {OUTPUT_DIR}")
print(f"  - episodic_summary.csv")
print(f"  - episodic_per_class_stats.csv")
print(f"  - episodic_all_results.csv")
print(f"  - episodic_overall_accuracy.png")
print(f"  - episodic_per_class_accuracy.png")
print(f"  - episodic_accuracy_trend.png")


# --- FINAL SUMMARY ---

print("\n" + "="*80)
print("EPISODIC EVALUATION COMPLETE!")
print("="*80)
print(f"\nConfiguration:")
print(f"  Dataset: {DATA_DIR_2}")
print(f"  Model: ResNet50Plus")
print(f"  Episodes: {NUM_EPISODES}")
print(f"  Support samples per class: {N_SUPPORT}")
print(f"  Query samples per class: {N_QUERY}")
print(f"\nFinal Results:")
print(f"  Mean Accuracy: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
print(f"  95% CI: [{ci[0]*100:.2f}%, {ci[1]*100:.2f}%]")
print("="*80)
