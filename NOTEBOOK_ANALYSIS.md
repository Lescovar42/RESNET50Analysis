# Comprehensive Analysis: ResNet50 Notebook MODEL_PATH_FS and Subspaces Issue

## Executive Summary

The notebook has code that uses the "from scratch" model (MODEL_PATH_FS) in **Cell 21** and tries to use Grad-CAM visualizations in **Cell 25**. The Grad-CAM code properly receives subspaces from an evaluated result dictionary, but the notebook structure shows that the "From Scratch" evaluation flow is complete and working, while the "External Dataset" (DATA_DIR_2) evaluation in Cell 14 was missing the subspace computation entirely.

---

## Question 1: All Cells Referencing MODEL_PATH_FS

### Cell 4 (Lines ~75)
**Purpose**: Configuration and path setup
```python
MODEL_PATH_FS = '/kaggle/input/resnet50plus-31102025-fromscratch/pytorch/default/1/Resnet50plus_plant_disease_model_31102025_fromsractch.pth'
print('Model path exists?', os.path.exists(MODEL_PATH_FS))
```
- Defines the "from scratch" model path
- Validates that the path exists

### Cell 21 (Lines ~850)
**Purpose**: Primary "From Scratch" evaluation block
```python
# Load the trained model
model = ResNet50Plus(num_classes=512)
model.load_state_dict(torch.load(MODEL_PATH_FS, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()
print("Model loaded successfully!")
```
- Loads the MODEL_PATH_FS weights into a ResNet50Plus architecture
- Runs complete evaluation pipeline with two datasets:
  - `results_1 = evaluate_dataset(..., DATA_DIR, ...)`
  - `results_2 = evaluate_dataset(..., DATA_DIR_2, ...)`
- Each result includes the computed subspaces in the returned dictionary

---

## Question 2: Grad-CAM Code Using `subspaces[cls_idx]` or `subspaces.keys()`

### Cell 25 (Lines ~1520-1600)
**Function**: `generate_gradcam_for_dataset()`

The Grad-CAM code **PROPERLY** retrieves and uses subspaces:

```python
def generate_gradcam_for_dataset(dataset_results, dataset_name, model, device, output_dir):
    # ... setup code ...
    
    # Get data from results
    subspaces = dataset_results['subspaces']  # <-- Retrieved from result dictionary
    
    # ... visualization loop ...
    
    for i, ax in enumerate(axes):
        # ... per-image processing ...
        
        # USES subspaces.keys() pattern:
        for cls_idx in sorted(subspaces.keys()):
            subspace = subspaces[cls_idx]  # <-- Access by class index
            mean = subspace['mean']         # <-- Access nested dict
            basis = subspace['basis']       # <-- Access nested dict
            
            # Compute reconstruction error
            centered_query = query_embedding - mean
            projection = torch.matmul(torch.matmul(centered_query, basis), basis.t())
            residual = centered_query - projection
            recon_error = torch.sum(residual.pow(2))
            distances.append(recon_error.item())
        
        # Use the predicted class from distances
        pred_label_idx = np.argmin(distances)
        
        # Generate Grad-CAM using the predicted subspace
        pred_subspace = subspaces[pred_label_idx]  # <-- Another access by index
```

**Key patterns found:**
- Line ~1580: `for cls_idx in sorted(subspaces.keys()):`
- Line ~1581: `subspace = subspaces[cls_idx]`
- Line ~1597: `pred_subspace = subspaces[pred_label_idx]`

---

## Question 3: Where `subspaces` Should Be Defined or Computed

### Cell 21: `evaluate_dataset()` Function (Lines ~820-920)

This function properly creates the subspaces:

```python
def evaluate_dataset(data_dir, dataset_name, model, device, n_support=5, n_query=10, subspace_dim=4):
    """
    Complete evaluation pipeline for a single dataset.
    Returns: dict with preds, targets, probs, class_names, ..., subspaces, accuracy
    """
    # ... steps 1-3: load data, split into support/query ...
    
    # STEP 4: Compute Subspaces (lines ~880-900)
    hyperplanes, means, cls_indices, support_class_counts = compute_subspaces(
        model, support_loader, device, subspace_dim=subspace_dim
    )
    
    # ... STEP 5: Evaluate on Query Set ...
    
    # BUILD SUBSPACES DICTIONARY (lines ~910-920):
    subspaces = {}
    for i, cls_idx in enumerate(cls_indices):
        subspaces[cls_idx] = {
            'mean': means[i],
            'basis': hyperplanes[i]
        }
    
    # RETURN RESULTS (line ~927):
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
        'subspaces': subspaces,  # <-- INCLUDED HERE
        'accuracy': acc
    }
```

### Usage in Cell 21 (Lines ~930-945):
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
```

---

## Question 4: Correct Variable Name - Stored in Result Dictionary

**Yes, subspaces are correctly stored in the result dictionary.**

The structure is:
```python
result = evaluate_dataset(...)
result['subspaces']  # Type: dict[int, dict[str, torch.Tensor]]

# Each value in result['subspaces'] is:
{
    'mean': torch.Tensor,   # Shape: [embedding_dim]
    'basis': torch.Tensor   # Shape: [embedding_dim, subspace_dim]
}
```

Accessed in Grad-CAM as:
```python
subspaces = dataset_results['subspaces']
for cls_idx in sorted(subspaces.keys()):
    subspace = subspaces[cls_idx]
    mean = subspace['mean']
    basis = subspace['basis']
```

---

## Question 5: Code Patterns for Subspace Assignment

### Cell 21: Computing Subspaces (Lines ~910-920)

```python
# BUILD SUBSPACES DICTIONARY FOR GRAD-CAM USE
subspaces = {}
for i, cls_idx in enumerate(cls_indices):
    subspaces[cls_idx] = {
        'mean': means[i],
        'basis': hyperplanes[i]
    }
```

### Cell 21: Returning in Result Dictionary (Line ~927)

```python
return {
    # ... other fields ...
    'subspaces': subspaces,
    'accuracy': acc
}
```

### Cell 25: Retrieving from Result (Line ~1570)

```python
def generate_gradcam_for_dataset(dataset_results, dataset_name, model, device, output_dir):
    # ...
    subspaces = dataset_results['subspaces']
```

---

## Complete Execution Flow

### Cell 4
- Sets up paths (MODEL_PATH_FS defined here)
- Creates OUTPUT_DIR
- Sets configuration (N_SUPPORT, N_QUERY, etc.)

### Cells 5-20
- Various setup, helper functions, model definitions

### **Cell 21** (THE KEY CELL - Uses MODEL_PATH_FS)
1. Loads the model using MODEL_PATH_FS
2. Defines helper functions including:
   - `compute_subspaces()` - Computes class subspaces using SVD
   - `evaluate_dsn()` - Evaluates using DSN classification
   - `evaluate_dataset()` - Complete pipeline that:
     - Loads and prepares dataset
     - Creates support/query splits
     - **Computes subspaces** from support set
     - Evaluates on query set
     - **Returns dict with 'subspaces' key**
3. Calls `evaluate_dataset()` twice:
   - `results_1` = evaluation on PlantVillage dataset
   - `results_2` = evaluation on External dataset

### Cells 22-24
- Visualization and debugging (confusion matrices, confidence plots)
- Model inspection

### **Cell 25** (Uses Grad-CAM with Subspaces)
1. Defines GradCAM class
2. Defines `generate_gradcam_for_dataset()` function
3. This function:
   - Takes `dataset_results` (dict from Cell 21)
   - Extracts `subspaces = dataset_results['subspaces']`
   - Iterates through `subspaces.keys()`
   - For each class: accesses `subspaces[cls_idx]`
   - Accesses nested fields: `subspace['mean']` and `subspace['basis']`
4. Calls the function twice:
   ```python
   generate_gradcam_for_dataset(
       dataset_results=results_1,
       ...
   )
   generate_gradcam_for_dataset(
       dataset_results=results_2,
       ...
   )
   ```

---

## Summary Table

| Question | Answer |
|----------|--------|
| 1. Which cells reference MODEL_PATH_FS? | **Cell 4** (definition), **Cell 21** (usage) |
| 2. Where does Grad-CAM use `subspaces[cls_idx]`? | **Cell 25**, function `generate_gradcam_for_dataset()`, line ~1581 and ~1597 |
| 3. Where is `subspaces` defined? | **Cell 21**, lines ~910-920, built in `evaluate_dataset()` function |
| 4. What is the correct variable name? | `result['subspaces']` where result comes from `evaluate_dataset()` |
| 5. Code patterns for subspace storage? | `subspaces[cls_idx] = {'mean': tensor, 'basis': tensor}` (Cell 21) and `result['subspaces']` (returned from evaluate_dataset) |
| 6. Which cell has failing Grad-CAM? | **Cell 25** (but it works correctly when called with results from Cell 21) |
| 7. What creates the subspaces? | **Cell 21** in the `evaluate_dataset()` function (lines ~880-900) |
| 8. Actual flow | Cell 21 computes subspaces and returns them in result dict → Cell 25 receives result and uses result['subspaces'] |

---

## Potential Issues and Notes

**The notebook structure is actually CORRECT for the From-Scratch flow:**
1. Cell 21 properly wraps the entire evaluation pipeline in `evaluate_dataset()`
2. This function computes subspaces and returns them in the dictionary
3. Cell 25 properly retrieves subspaces from the result dictionary

**However, there was a separate issue in Cell 14** (not MODEL_PATH_FS related):
- The "External Dataset (Novel Classes Only)" section in Cell 14 also needs subspace computation
- The fix documentation indicates Cell 14 was missing the `compute_subspaces()` call
- This has been documented in `/home/user/RESNET50Analysis/fix_subspace_computation.md`

---

## File Paths (Absolute)
- Notebook: `/home/user/RESNET50Analysis/fork-of-dsn-few-shot-learning-resnet50-analysis.ipynb`
- Documentation: `/home/user/RESNET50Analysis/fix_subspace_computation.md`

