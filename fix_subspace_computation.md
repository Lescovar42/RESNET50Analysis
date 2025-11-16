# Fix: Missing Subspace Computation in Cell 14

## Error Description
**NameError**: `hyperplanes_ds2`, `means_ds2`, `class_indices_ds2` are not defined

## Root Cause
Cell 14 creates the support/query loaders and loads the model, but **never calls `compute_subspaces()`** to create the subspace variables needed for DSN evaluation and Grad-CAM visualization.

## Location
- **Notebook**: `fork-of-dsn-few-shot-learning-resnet50-analysis.ipynb`
- **Cell**: 14 (DATA_DIR_2 evaluation section)
- **Missing code**: Should appear after line ~155 (after creating loaders)

## Solution

### Add this code after creating `support_loader_ds2` and `query_loader_ds2`:

```python
# --- STEP 4: Compute Subspaces for Each Class ---

print("\n" + "-"*80)
print("STEP 4: Computing subspaces from support set...")
print("-"*80)

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

# CRITICAL: Compute subspaces for DATA_DIR_2 evaluation
hyperplanes_ds2, means_ds2, class_indices_ds2, support_class_counts_ds2 = compute_subspaces(
    model_ds2, support_loader_ds2, DEVICE, subspace_dim=SUBSPACE_DIM
)

print(f"Computed subspaces for {len(class_indices_ds2)} classes")
print(f"Support samples per class (first 5):")
for i, cls_idx in enumerate(class_indices_ds2[:5]):
    print(f"  {novel_classes_names[cls_idx][:40]:40s}: {support_class_counts_ds2[cls_idx]:2d} samples")
print("  ...")
```

## Exact Insertion Point

**After this section:**
```python
support_dataset_ds2 = Subset(full_dataset_novel_ds2, support_indices_ds2)
query_dataset_ds2 = Subset(full_dataset_novel_ds2, query_indices_ds2)
support_loader_ds2 = DataLoader(support_dataset_ds2, batch_size=32, shuffle=False)
query_loader_ds2 = DataLoader(query_dataset_ds2, batch_size=32, shuffle=False)
```

**Before this section:**
```python
# --- STEP 3: Define ResNet50Plus (from scratch) Model Architecture ---
print("\n" + "-"*80)
print("STEP 3: Defining ResNet50Plus model architecture...")
```

## Why This Fixes The Error

1. **Creates `hyperplanes_ds2`**: Subspace basis vectors for each class
2. **Creates `means_ds2`**: Mean embedding for each class
3. **Creates `class_indices_ds2`**: Mapping of class indices
4. **Creates `support_class_counts_ds2`**: Sample counts per class

These variables are then properly available for:
- `evaluate_dsn_ds2()` function call (line 254)
- Grad-CAM visualization loop (lines 527-544)

## Comparison: Why "From Scratch" Section Works

The "From Scratch" section (Cell 21) correctly wraps this in an `evaluate_dataset()` function that:
1. Loads the model
2. **Computes subspaces** (lines 246-248)
3. Evaluates on query set
4. Returns results in a dictionary

Cell 14 attempted to do this manually but **skipped step 2**.

## Files Referenced
- This fix is already included in: `resnet50plus_evaluation_fixed.py` (lines 194-250)
- Documentation: `FIXES_DOCUMENTATION.md`
