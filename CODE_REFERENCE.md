# Code Reference: Complete Subspace Flow in Notebook

## Visual Flow Diagram

```
Cell 4: MODEL_PATH_FS Defined
    |
    v
Cell 21: Load Model with MODEL_PATH_FS
    |
    +---> Define compute_subspaces()
    |
    +---> Define evaluate_dataset()
            |
            +---> Call compute_subspaces()
            |       (Creates hyperplanes, means, class_indices)
            |
            +---> Build subspaces dictionary
            |       subspaces[cls_idx] = {'mean': ..., 'basis': ...}
            |
            +---> Return dict with 'subspaces' key
    |
    +---> results_1 = evaluate_dataset(DATA_DIR, ...)
    |
    +---> results_2 = evaluate_dataset(DATA_DIR_2, ...)
    |
    v
Cell 25: Use Grad-CAM
    |
    +---> generate_gradcam_for_dataset(results_1, ...)
    |       subspaces = dataset_results['subspaces']
    |       for cls_idx in sorted(subspaces.keys()):
    |
    +---> generate_gradcam_for_dataset(results_2, ...)
            subspaces = dataset_results['subspaces']
            for cls_idx in sorted(subspaces.keys()):
```

---

## Code Snippets by Location

### 1. CELL 4: Define MODEL_PATH_FS Path

**File**: `/home/user/RESNET50Analysis/fork-of-dsn-few-shot-learning-resnet50-analysis.ipynb`
**Cell**: 4
**Lines**: ~75

```python
MODEL_PATH_NORMAL = '/kaggle/input/resnet50-26102025/pytorch/default/1/Resnet50_plant_disease_model_26102025.pth'
MODEL_PATH_PLUS = '/kaggle/input/resnet50plus-31102025/pytorch/default/1/Resnet50plus_plant_disease_model_31102025.pth'
MODEL_PATH_FS = '/kaggle/input/resnet50plus-31102025-fromscratch/pytorch/default/1/Resnet50plus_plant_disease_model_31102025_fromsractch.pth'
DATA_DIR = "/kaggle/input/plantvillage-dataset/color"
DATA_DIR_2 = "/kaggle/input/datasetold/Datasetold/Train"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = './gradcam_outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_SUPPORT = 15
N_QUERY = 10

print('Device:', DEVICE)
print('Model path exists?', os.path.exists(MODEL_PATH_FS))
```

---

### 2. CELL 21: Primary "From Scratch" Evaluation

**File**: `/home/user/RESNET50Analysis/fork-of-dsn-few-shot-learning-resnet50-analysis.ipynb`
**Cell**: 21
**Critical Sections Below**

#### 2A. Load Model with MODEL_PATH_FS

```python
# LOAD MODEL
print("\n" + "="*80)
print("LOADING MODEL")
print("="*80)

class Bottleneck(nn.Module):
    # ... implementation ...

class ResNet50Plus(nn.Module):
    # ... implementation ...

# Load the trained model
model = ResNet50Plus(num_classes=512)
model.load_state_dict(torch.load(MODEL_PATH_FS, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()
print("Model loaded successfully!")
```

#### 2B. Define compute_subspaces() Function

```python
def compute_subspaces(model, support_loader, device, subspace_dim=4):
    """
    Compute subspaces for each class using support samples.
    Uses SVD on centered embeddings - matches training implementation.
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
        
        # SVD on transposed centered embeddings (matching training code)
        uu, s, v = torch.svd(centered.transpose(0, 1).double(), some=False)
        uu = uu.float()
        
        # Take first num_dim columns as hyperplane basis
        hyperplane = uu[:, :num_dim]  # [feature_dim, num_dim]
        all_hyperplanes.append(hyperplane)
    
    # Stack into tensors
    all_hyperplanes = torch.stack(all_hyperplanes, dim=0)  # [num_classes, feature_dim, num_dim]
    all_means = torch.stack(all_means, dim=0)  # [num_classes, feature_dim]
    
    return all_hyperplanes, all_means, class_indices, class_counts
```

#### 2C. Define evaluate_dataset() Function (CONDENSED - see full version below)

```python
def evaluate_dataset(data_dir, dataset_name, model, device, n_support=5, n_query=10, subspace_dim=4):
    """
    Complete evaluation pipeline for a single dataset.
    Returns: dict with preds, targets, probs, class_names, support_indices, 
             query_indices, full_dataset, hyperplanes, means, class_indices, 
             subspaces, accuracy
    """
    # STEP 1: Load Class Information
    train_classes = get_classes_from_csv(TRAIN_CSV)
    val_classes = get_classes_from_csv(VAL_CSV)
    test_classes = get_classes_from_csv(TEST_CSV)
    all_classes = sorted(set(train_classes + val_classes + test_classes))
    
    # STEP 2: Prepare Dataset
    full_dataset = prepare_dataset(data_dir, all_classes)
    class_names = all_classes
    
    # STEP 3: Split into Support and Query Sets
    support_indices, query_indices = create_support_query_split(
        full_dataset, class_names, n_support=n_support, n_query=n_query
    )
    support_dataset = Subset(full_dataset, support_indices)
    query_dataset = Subset(full_dataset, query_indices)
    support_loader = DataLoader(support_dataset, batch_size=32, shuffle=False)
    query_loader = DataLoader(query_dataset, batch_size=32, shuffle=False)
    
    # STEP 4: Compute Subspaces
    hyperplanes, means, cls_indices, support_class_counts = compute_subspaces(
        model, support_loader, device, subspace_dim=subspace_dim
    )
    
    # STEP 5: Evaluate on Query Set
    preds, targets, probs = evaluate_dsn(
        model, query_loader, hyperplanes, means, cls_indices, device
    )
    
    # STEP 6: Results and Analysis (print statements)
    acc = (preds == targets).mean()
    # ... detailed results printing ...
    
    # BUILD SUBSPACES DICTIONARY FOR GRAD-CAM
    subspaces = {}
    for i, cls_idx in enumerate(cls_indices):
        subspaces[cls_idx] = {
            'mean': means[i],
            'basis': hyperplanes[i]
        }
    
    return {
        'preds': preds,
        'targets': targets,
        'probs': probs,
        'class_names': class_names,
        'support_indices': support_indices,
        'query_indices': query_indices,
        'full_dataset': full_dataset,
        'hyperplanes': hyperplanes,
        'means': means,
        'class_indices': cls_indices,
        'subspaces': subspaces,  # <-- INCLUDES SUBSPACES HERE
        'accuracy': acc
    }
```

#### 2D. Call evaluate_dataset() - Where Results Are Created

```python
# EVALUATE DATASET 1 (Original PlantVillage Dataset)
results_1 = evaluate_dataset(
    data_dir=DATA_DIR,
    dataset_name="PlantVillage (Original Training Dataset)",
    model=model,
    device=DEVICE,
    n_support=N_SUPPORT,
    n_query=N_QUERY,
    subspace_dim=SUBSPACE_DIM
)

# EVALUATE DATASET 2 (External Dataset: Hama, Sehat, Virus)
results_2 = evaluate_dataset(
    data_dir=DATA_DIR_2,
    dataset_name="External Dataset (Hama, Sehat, Virus)",
    model=model,
    device=DEVICE,
    n_support=N_SUPPORT,
    n_query=N_QUERY,
    subspace_dim=SUBSPACE_DIM
)

print("\n" + "="*80)
print("EVALUATION COMPLETE FOR BOTH DATASETS")
print("="*80)
print(f"Dataset 1 (PlantVillage) Accuracy: {results_1['accuracy']*100:.2f}%")
print(f"Dataset 2 (External) Accuracy: {results_2['accuracy']*100:.2f}%")
print("="*80)
```

---

### 3. CELL 25: Grad-CAM Visualization Using Subspaces

**File**: `/home/user/RESNET50Analysis/fork-of-dsn-few-shot-learning-resnet50-analysis.ipynb`
**Cell**: 25
**Key Function**: `generate_gradcam_for_dataset()`

#### 3A. Function Definition - Extracting Subspaces

```python
def generate_gradcam_for_dataset(dataset_results, dataset_name, model, device, output_dir):
    """
    Generate Grad-CAM visualizations for a dataset.
    """
    print("\n" + "="*80)
    print(f"GENERATING GRAD-CAM FOR: {dataset_name}")
    print("="*80)

    # Get target layer and initialize Grad-CAM
    target_layer = get_resnet_like_last_conv(model)
    grad_cam = GradCAM(model, target_layer)

    # Get data from results
    class_names = dataset_results['class_names']
    query_indices = dataset_results['query_indices']
    full_dataset = dataset_results['full_dataset']
    subspaces = dataset_results['subspaces']  # <-- RETRIEVE SUBSPACES FROM DICT
    
    # ... select images to visualize ...
```

#### 3B. Key Loop - Using subspaces.keys() and subspaces[cls_idx]

```python
    for i, ax in enumerate(axes):
        if i >= n_images:
            ax.axis('off')
            continue

        img_idx = images_to_visualize[i]
        img_tensor, true_label_idx = full_dataset[img_idx]

        input_tensor = img_tensor.unsqueeze(0).to(device)
        query_embedding = model(input_tensor)

        # Compute distances to all subspaces
        distances = []
        for cls_idx in sorted(subspaces.keys()):  # <-- USE subspaces.keys()
            subspace = subspaces[cls_idx]  # <-- ACCESS BY INDEX
            mean = subspace['mean']        # <-- GET mean
            basis = subspace['basis']      # <-- GET basis

            centered_query = query_embedding - mean
            projection = torch.matmul(torch.matmul(centered_query, basis), basis.t())
            residual = centered_query - projection
            recon_error = torch.sum(residual.pow(2))
            distances.append(recon_error.item())

        pred_label_idx = np.argmin(distances)

        # Generate Grad-CAM
        pred_subspace = subspaces[pred_label_idx]  # <-- ANOTHER ACCESS BY INDEX
        centered_query = query_embedding - pred_subspace['mean']
        projection = torch.matmul(torch.matmul(centered_query, pred_subspace['basis']), 
                                   pred_subspace['basis'].t())
        score_for_gradcam = -torch.sum((centered_query - projection).pow(2))

        heatmap = grad_cam.generate_heatmap(score=score_for_gradcam, target_class=pred_label_idx)
        
        # ... visualization code ...
```

#### 3C. Call generate_gradcam_for_dataset() - Using Results from Cell 21

```python
# GENERATE GRAD-CAM FOR BOTH DATASETS

print("\n" + "="*80)
print("GENERATING GRAD-CAM VISUALIZATIONS")
print("="*80)

# Generate for Dataset 1
generate_gradcam_for_dataset(
    dataset_results=results_1,  # <-- Pass results_1 from Cell 21
    dataset_name="PlantVillage (Original)",
    model=model,
    device=DEVICE,
    output_dir=OUTPUT_DIR
)

# Generate for Dataset 2
generate_gradcam_for_dataset(
    dataset_results=results_2,  # <-- Pass results_2 from Cell 21
    dataset_name="External Dataset (Hama, Sehat, Virus)",
    model=model,
    device=DEVICE,
    output_dir=OUTPUT_DIR
)

print("\n" + "="*80)
print("ALL GRAD-CAM VISUALIZATIONS COMPLETE!")
print("="*80)
```

---

## Data Structure Reference

### Result Dictionary Structure (from evaluate_dataset)

```python
result = {
    'preds': np.ndarray,                    # Predictions
    'targets': np.ndarray,                  # Ground truth
    'probs': np.ndarray,                    # Class probabilities
    'class_names': list[str],               # Class names
    'support_indices': list[int],           # Indices used for support
    'query_indices': list[int],             # Indices used for query
    'full_dataset': Dataset,                # Full dataset object
    'hyperplanes': torch.Tensor,            # Shape: [num_classes, feature_dim, subspace_dim]
    'means': torch.Tensor,                  # Shape: [num_classes, feature_dim]
    'class_indices': list[int],             # Class indices
    'subspaces': {                          # <-- CRITICAL FOR GRAD-CAM
        cls_idx: {
            'mean': torch.Tensor,           # Shape: [feature_dim]
            'basis': torch.Tensor           # Shape: [feature_dim, subspace_dim]
        },
        # ... for each class ...
    },
    'accuracy': float                       # Overall accuracy
}
```

---

## Summary of Variable Relationships

| Variable | Created In | Used In | Type |
|----------|-----------|---------|------|
| `MODEL_PATH_FS` | Cell 4 | Cell 21 | `str` (path) |
| `model` | Cell 21 | Cell 21, 25 | `ResNet50Plus` instance |
| `hyperplanes` | Cell 21 (compute_subspaces) | Cell 21 (evaluate_dataset) | `torch.Tensor` |
| `means` | Cell 21 (compute_subspaces) | Cell 21 (evaluate_dataset) | `torch.Tensor` |
| `class_indices` | Cell 21 (compute_subspaces) | Cell 21 (evaluate_dataset) | `list` |
| `subspaces` | Cell 21 (evaluate_dataset) | Cell 21 (return), Cell 25 (use) | `dict[int, dict]` |
| `results_1` | Cell 21 (evaluate_dataset call) | Cell 25 | `dict` |
| `results_2` | Cell 21 (evaluate_dataset call) | Cell 25 | `dict` |

---

## Execution Order

1. **Cell 4**: Define paths and configuration
2. **Cell 21**:
   - Define helper functions
   - Load model with MODEL_PATH_FS
   - Call evaluate_dataset(DATA_DIR) → returns results_1 (includes subspaces)
   - Call evaluate_dataset(DATA_DIR_2) → returns results_2 (includes subspaces)
3. **Cell 25**:
   - Extract subspaces from results_1: `subspaces = results_1['subspaces']`
   - Use subspaces for Grad-CAM visualization
   - Extract subspaces from results_2: `subspaces = results_2['subspaces']`
   - Use subspaces for Grad-CAM visualization

---

## Key Code Patterns

### Pattern 1: Subspace Storage (Cell 21)
```python
subspaces = {}
for i, cls_idx in enumerate(cls_indices):
    subspaces[cls_idx] = {
        'mean': means[i],
        'basis': hyperplanes[i]
    }
```

### Pattern 2: Subspace Retrieval (Cell 25)
```python
subspaces = dataset_results['subspaces']
```

### Pattern 3: Subspace Access (Cell 25)
```python
for cls_idx in sorted(subspaces.keys()):
    subspace = subspaces[cls_idx]
    mean = subspace['mean']
    basis = subspace['basis']
```

