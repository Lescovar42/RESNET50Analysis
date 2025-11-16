# Fix: "subspace not defined" Error in Grad-CAM Code

## Problem

Your Grad-CAM code tries to use `subspaces` variable directly:

```python
for cls_idx in sorted(subspaces.keys()):
    subspace = subspaces[cls_idx]
```

But the `subspaces` variable is **never defined** before this point, causing:
```
NameError: name 'subspaces' is not defined
```

## Root Cause

The Grad-CAM code is missing the critical setup steps that:
1. Compute the subspaces from the support set using SVD
2. Store them in a dictionary structure
3. Make them available for the Grad-CAM loop

## Solution

You need to add the missing code **BEFORE** your Grad-CAM visualization loop. Here's the complete fix:

### Step 1: Compute Subspaces from Support Set

Add this code BEFORE your Grad-CAM section:

```python
print("\n" + "-"*80)
print("COMPUTING SUBSPACES FROM SUPPORT SET...")
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

    for cls_idx in class_indices:
        embs = class_embeddings[cls_idx]
        embs_tensor = torch.stack(embs)

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
                print(f"Warning: Reduced subspace dim to {num_dim}")

        # SVD on transposed centered embeddings
        uu, s, v = torch.svd(centered.transpose(0, 1).double(), some=False)
        uu = uu.float()

        # Take first num_dim columns as hyperplane basis
        hyperplane = uu[:, :num_dim]
        all_hyperplanes.append(hyperplane)

    # Stack into tensors
    all_hyperplanes = torch.stack(all_hyperplanes, dim=0)
    all_means = torch.stack(all_means, dim=0)

    return all_hyperplanes, all_means, class_indices, class_counts

# CRITICAL: Call the function to compute subspaces
hyperplanes, means, cls_indices, support_counts = compute_subspaces(
    model,
    support_loader,  # Make sure this exists!
    DEVICE,
    subspace_dim=SUBSPACE_DIM
)

print(f"Computed subspaces for {len(cls_indices)} classes")
```

### Step 2: Build the subspaces Dictionary

After computing hyperplanes and means, build the dictionary:

```python
# Build subspaces dictionary for Grad-CAM
subspaces = {}
for i, cls_idx in enumerate(cls_indices):
    subspaces[cls_idx] = {
        'mean': means[i],
        'basis': hyperplanes[i]
    }

print(f"Subspaces dictionary created with {len(subspaces)} classes")
```

### Step 3: NOW Your Grad-CAM Code Will Work

Now your existing Grad-CAM code will work correctly:

```python
# Your existing Grad-CAM code (now will work!)
for cls_idx in sorted(subspaces.keys()):
    subspace = subspaces[cls_idx]
    mean = subspace['mean']
    basis = subspace['basis']
    # ... rest of your code ...
```

## Complete Code Structure

Here's how your code should be structured:

```python
# 1. IMPORTS (you already have these)
import cv2
import torch.nn.functional as F
from collections import defaultdict

# 2. DEFINE GradCAM CLASS (you already have this)
class GradCAM:
    # ... your existing GradCAM code ...

# 3. DEFINE HELPER FUNCTIONS (you already have these)
def inverse_normalize(...):
    # ... your existing code ...

def plot_gradcam(...):
    # ... your existing code ...

def get_resnet_like_last_conv(model):
    # ... your existing code ...

# 4. **NEW: COMPUTE SUBSPACES** (THIS IS MISSING!)
def compute_subspaces(model, support_loader, device, subspace_dim=4):
    # ... code from Step 1 above ...

hyperplanes, means, cls_indices, support_counts = compute_subspaces(
    model, support_loader, DEVICE, subspace_dim=SUBSPACE_DIM
)

# 5. **NEW: BUILD SUBSPACES DICTIONARY** (THIS IS MISSING!)
subspaces = {}
for i, cls_idx in enumerate(cls_indices):
    subspaces[cls_idx] = {
        'mean': means[i],
        'basis': hyperplanes[i]
    }

# 6. NOW YOUR EXISTING GRAD-CAM CODE
print("Selecting target layer for Grad-CAM...")
target_layer = get_resnet_like_last_conv(model)
grad_cam = GradCAM(model, target_layer)

print("Generating Grad-CAM visualizations...")

# ... rest of your existing Grad-CAM code ...
```

## Prerequisites You Need

Make sure these variables are defined BEFORE the above code:

1. **`model`** - Your trained ResNet50 model (loaded from MODEL_PATH_FS)
2. **`support_loader`** - DataLoader for support set
3. **`query_loader`** - DataLoader for query set
4. **`query_indices`** - List of indices in query set
5. **`full_dataset`** - Full dataset object
6. **`class_names`** - List of class names
7. **`DEVICE`** - torch.device('cuda' or 'cpu')
8. **`SUBSPACE_DIM`** - Subspace dimension (typically N_SUPPORT - 1)

## Verification

To verify the fix worked:

```python
# After building subspaces dictionary
print(f"Subspaces computed: {len(subspaces)}")
print(f"Keys: {sorted(subspaces.keys())}")
print(f"First subspace mean shape: {subspaces[cls_indices[0]]['mean'].shape}")
print(f"First subspace basis shape: {subspaces[cls_indices[0]]['basis'].shape}")
```

Expected output:
```
Subspaces computed: 38
Keys: [0, 1, 2, 3, ...]
First subspace mean shape: torch.Size([512])
First subspace basis shape: torch.Size([512, 14])
```

## Why This Happens

In the notebook's Cell 21, the subspace computation is wrapped in an `evaluate_dataset()` function that returns a dictionary. Cell 25 then uses `generate_gradcam_for_dataset()` which receives this dictionary and extracts:

```python
subspaces = dataset_results['subspaces']
```

Your code is trying to run the Grad-CAM visualization in isolation, so you need to manually compute and build the `subspaces` dictionary yourself.

## Alternative: Use the Notebook's Functions

Instead of running Grad-CAM code directly, you could use the notebook's existing structure:

```python
# Option 1: Use the evaluate_dataset function (if defined)
results = evaluate_dataset(
    data_dir=DATA_DIR,
    dataset_name="PlantVillage",
    model=model,
    device=DEVICE,
    n_support=N_SUPPORT,
    n_query=N_QUERY,
    subspace_dim=SUBSPACE_DIM
)

# Then extract subspaces
subspaces = results['subspaces']

# Now your Grad-CAM code will work
for cls_idx in sorted(subspaces.keys()):
    # ...
```

---

## Files for Reference

- **Complete working example**: See Cell 21 and Cell 25 in the notebook
- **Code snippets with line numbers**: `/home/user/RESNET50Analysis/CODE_REFERENCE.md`
- **Flow diagram**: `/home/user/RESNET50Analysis/QUICK_REFERENCE.txt`

---

**Summary**: Add the `compute_subspaces()` function call and build the `subspaces` dictionary BEFORE your Grad-CAM visualization loop.
