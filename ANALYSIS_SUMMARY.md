# Analysis Summary: MODEL_PATH_FS and Subspaces

## Quick Answers

### 1. All Cells Referencing MODEL_PATH_FS (From-Scratch Model)

**Cell 4** (~line 75):
- Defines the path constant
- Validates the file exists

**Cell 21** (~line 850):
- Loads the model using `torch.load(MODEL_PATH_FS)`
- Runs the complete "From Scratch" evaluation pipeline

---

### 2. Grad-CAM Code Using `subspaces[cls_idx]` or `subspaces.keys()`

**Cell 25**, function `generate_gradcam_for_dataset()` (~lines 1580-1597):

```python
for cls_idx in sorted(subspaces.keys()):
    subspace = subspaces[cls_idx]
    mean = subspace['mean']
    basis = subspace['basis']
```

---

### 3. Where `subspaces` Should Be Defined or Computed

**Cell 21**, in the `evaluate_dataset()` function (~lines 910-920):

```python
# BUILD SUBSPACES DICTIONARY FOR GRAD-CAM
subspaces = {}
for i, cls_idx in enumerate(cls_indices):
    subspaces[cls_idx] = {
        'mean': means[i],
        'basis': hyperplanes[i]
    }
```

The subspaces are computed from the support set using SVD (lines ~880-900).

---

### 4. Correct Variable Name - Stored in Result Dictionary

**YES** - Subspaces are stored in the result dictionary returned by `evaluate_dataset()`:

```python
result = evaluate_dataset(...)
result['subspaces']  # Access the subspaces dictionary
```

Structure:
```python
result['subspaces'] = {
    class_idx: {
        'mean': torch.Tensor,     # Class embedding mean
        'basis': torch.Tensor     # Subspace basis from SVD
    },
    # ... for each class ...
}
```

---

### 5. Code Patterns for Subspace Assignment

**Cell 21**: Define and build subspaces
```python
subspaces = {}
for i, cls_idx in enumerate(cls_indices):
    subspaces[cls_idx] = {
        'mean': means[i],
        'basis': hyperplanes[i]
    }
```

**Cell 21**: Return in result dictionary
```python
return {
    # ... other fields ...
    'subspaces': subspaces,
    'accuracy': acc
}
```

**Cell 25**: Retrieve from result
```python
subspaces = dataset_results['subspaces']
for cls_idx in sorted(subspaces.keys()):
    subspace = subspaces[cls_idx]
```

---

## Complete Data Flow

```
Cell 4: Define MODEL_PATH_FS
    |
    v
Cell 21: Load Model with MODEL_PATH_FS
    |
    +-- Defines compute_subspaces()
    |
    +-- Defines evaluate_dataset() which:
    |   - Loads and prepares data
    |   - Creates support/query splits
    |   - Calls compute_subspaces() to get:
    |       * hyperplanes (subspace basis)
    |       * means (class centers)
    |       * class_indices (mapping)
    |   - Computes subspaces dict from above
    |   - Returns dict with 'subspaces' key
    |
    +-- Calls evaluate_dataset() twice:
    |   - results_1 = evaluate_dataset(DATA_DIR, ...)
    |   - results_2 = evaluate_dataset(DATA_DIR_2, ...)
    |
    v
Cell 25: Use Grad-CAM with Subspaces
    |
    +-- generate_gradcam_for_dataset(results_1, ...)
    |   - Extracts: subspaces = results_1['subspaces']
    |   - Uses: for cls_idx in sorted(subspaces.keys())
    |
    +-- generate_gradcam_for_dataset(results_2, ...)
        - Extracts: subspaces = results_2['subspaces']
        - Uses: for cls_idx in sorted(subspaces.keys())
```

---

## Key Variables Tracking

| Variable | Where Created | How Created | Where Used | Format |
|----------|--------------|------------|-----------|--------|
| `MODEL_PATH_FS` | Cell 4 | String literal | Cell 21 | `str` |
| `model` | Cell 21 | Loaded from MODEL_PATH_FS | Cell 21, 25 | `ResNet50Plus` |
| `hyperplanes` | Cell 21 | compute_subspaces() | evaluate_dataset() | `torch.Tensor [num_classes, feature_dim, subspace_dim]` |
| `means` | Cell 21 | compute_subspaces() | evaluate_dataset() | `torch.Tensor [num_classes, feature_dim]` |
| `class_indices` | Cell 21 | compute_subspaces() | evaluate_dataset() | `list[int]` |
| `subspaces` | Cell 21 | Built from hyperplanes+means | Return in dict | `dict[int, dict]` |
| `results_1` | Cell 21 | evaluate_dataset() returns dict | Cell 25 | `dict` with key 'subspaces' |
| `results_2` | Cell 21 | evaluate_dataset() returns dict | Cell 25 | `dict` with key 'subspaces' |

---

## Why This Design Works

1. **Encapsulation**: The `evaluate_dataset()` function wraps the entire pipeline
   - Inputs: data directory, model, configuration
   - Outputs: results dictionary with everything needed downstream

2. **Clean Interface**: Cell 21 returns complete results
   - Cell 25 doesn't need to recompute anything
   - Just extracts what it needs: `dataset_results['subspaces']`

3. **Reusability**: Same code evaluates two different datasets
   - results_1 for PlantVillage
   - results_2 for External dataset
   - Same Grad-CAM code works for both

4. **Maintainability**: Changes to subspace computation only need to be made in:
   - The `compute_subspaces()` function definition
   - Or the subspace dict building code
   - Doesn't affect Grad-CAM code at all

---

## The Complete Subspace Computation Chain

```
Support Data
    |
    v
compute_subspaces(model, support_loader, device, subspace_dim):
    |
    +-- Extract embeddings from support samples
    |
    +-- For each class:
    |   - Compute mean embedding
    |   - Center the embeddings
    |   - Apply SVD to centered embeddings
    |   - Extract first N singular vectors as subspace basis
    |
    +-- Return: hyperplanes, means, class_indices, class_counts
    |
    v
evaluate_dataset() builds subspaces dict:
    |
    +-- subspaces = {}
    +-- For each class:
    |   - subspaces[cls_idx] = {
    |       'mean': means[i],
    |       'basis': hyperplanes[i]
    |     }
    |
    v
Return in result dict:
    |
    +-- result['subspaces'] = subspaces
    |
    v
Grad-CAM uses subspaces:
    |
    +-- For each class in sorted(subspaces.keys()):
    |   - Access: subspace = subspaces[cls_idx]
    |   - Get: mean, basis
    |   - Compute reconstruction error for classification
    |   - Generate Grad-CAM heatmap
```

---

## Comparison: Expected vs Actual

### Expected (What We Looked For)
- NameError or missing variable
- `subspaces` not defined
- Grad-CAM code fails

### Actual (What We Found)
- The "From Scratch" flow (Cell 21 → Cell 25) is **PROPERLY IMPLEMENTED**
- `subspaces` are correctly:
  - Computed in Cell 21
  - Returned in result dictionary
  - Retrieved in Cell 25
  - Used with proper `subspaces.keys()` iteration

### There WAS a Separate Issue
- Cell 14 (External Dataset evaluation) was missing the `compute_subspaces()` call
- This has been documented in `fix_subspace_computation.md`
- But the MODEL_PATH_FS flow is complete and correct

---

## Files for Reference

1. **NOTEBOOK_ANALYSIS.md** - Comprehensive detailed analysis
2. **CODE_REFERENCE.md** - Code snippets with line numbers and context
3. **fix_subspace_computation.md** - Documentation of Cell 14 issue (separate problem)

All files are located in: `/home/user/RESNET50Analysis/`

---

## Conclusion

The notebook's "From Scratch" evaluation pipeline (Cell 21 with MODEL_PATH_FS) properly:

1. Loads the model using MODEL_PATH_FS
2. Computes subspaces from support samples
3. Stores them in a result dictionary
4. Returns the complete result to Cell 25
5. Cell 25 retrieves and uses subspaces correctly

The design follows good software engineering practices:
- Functional decomposition
- Data encapsulation in dictionaries
- Clean separation of concerns
- Reusable code patterns

No errors expected in the MODEL_PATH_FS → Grad-CAM flow.

