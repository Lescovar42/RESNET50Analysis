# Quick Start: Episodic Evaluation

## 1-Minute Setup

### Step 1: Define Required Variables

```python
# In your notebook, define these variables BEFORE running the episodic script
DATA_DIR_2 = "/kaggle/input/datasetold/Datasetold/Train"  # Your dataset path
MODEL_PATH_PLUS = "/kaggle/input/resnet50plus-31102025/pytorch/default/1/Resnet50plus_plant_disease_model_31102025.pth"
OUTPUT_DIR = './episodic_results'
N_SUPPORT = 15  # Number of support samples per class
N_QUERY = 10    # Number of query samples per class
SUBSPACE_DIM = N_SUPPORT - 1  # Subspace dimension (typically N_SUPPORT - 1)
```

### Step 2: Run the Episodic Evaluation

```python
# Option A: Run the script
%run resnet50plus_episodic_evaluation.py

# Option B: Execute inline
exec(open('resnet50plus_episodic_evaluation.py').read())
```

### Step 3: Check Results

Results are automatically displayed and saved to `OUTPUT_DIR`.

## Expected Output

```
Device: cuda
Configuration: 15-shot, 10 queries per class
Subspace dimensionality: 14
Number of episodes: 100
Confidence level: 95.0%

================================================================================
STEP 1: Preparing dataset for DATA_DIR_2: /path/to/dataset...
================================================================================
Found 38 classes in /path/to/dataset
Total images: 5420

...

================================================================================
STEP 4: Running 100 episodes...
================================================================================
Evaluating episodes: 100%|████████████████████| 100/100 [05:23<00:00,  3.23s/it]

Completed 100 episodes

================================================================================
STEP 5: Computing statistics across episodes...
================================================================================

OVERALL ACCURACY STATISTICS (100 episodes)
======================================================================
Mean accuracy:        87.34%
Std deviation:        2.15%
Min accuracy:         81.20%
Max accuracy:         92.50%
Median accuracy:      87.60%
95% Confidence Int:   [86.91%, 87.77%]

======================================================================
PER-CLASS ACCURACY STATISTICS
======================================================================
Class                                         Mean      Std      Min      Max
----------------------------------------------------------------------
Apple___Apple_scab                           89.2%     3.4%    82.0%    95.0%
Apple___Black_rot                            85.7%     4.1%    76.0%    93.0%
...

================================================================================
EPISODIC EVALUATION COMPLETE!
================================================================================

Configuration:
  Dataset: /path/to/dataset
  Model: ResNet50Plus
  Episodes: 100
  Support samples per class: 15
  Query samples per class: 10

Final Results:
  Mean Accuracy: 87.34% ± 2.15%
  95% CI: [86.91%, 87.77%]
================================================================================
```

## Saved Files

Check `OUTPUT_DIR` for:

1. **episodic_summary.csv** - Overall statistics
2. **episodic_per_class_stats.csv** - Per-class breakdown
3. **episodic_all_results.csv** - All 100 episode results
4. **episodic_overall_accuracy.png** - Accuracy histogram and boxplot
5. **episodic_per_class_accuracy.png** - Per-class bar chart
6. **episodic_accuracy_trend.png** - Episode-wise trend plot

## Interpretation

### What do the numbers mean?

**Mean Accuracy: 87.34%**
- On average, across 100 different random support/query splits, the model achieves 87.34% accuracy

**Std Deviation: 2.15%**
- Low std (<3%) means the model is very stable across different episodes
- Performance doesn't vary much based on which samples are in support vs query

**95% Confidence Interval: [86.91%, 87.77%]**
- We're 95% confident the true accuracy is between 86.91% and 87.77%
- Narrower interval = more precise estimate

### Is this good?

Compare to:
- **Random guessing** (38 classes): 2.63% accuracy
- **Your model**: 87.34% accuracy
- **Improvement**: 33x better than random!

### Which classes are hard?

Look at per-class statistics:
- **High mean, low std**: Easy, consistent class
- **Low mean, high std**: Hard, inconsistent class
- Focus improvement efforts on low-performing classes

## Customization

### Run Fewer Episodes (Faster Testing)

```python
# Edit resnet50plus_episodic_evaluation.py
NUM_EPISODES = 10  # Change from 100 to 10

# Or directly in notebook:
NUM_EPISODES = 10
%run resnet50plus_episodic_evaluation.py
```

### Run More Episodes (Publication Quality)

```python
NUM_EPISODES = 500  # Takes ~25 minutes
%run resnet50plus_episodic_evaluation.py
```

### Change Shot Configuration

```python
# 5-shot instead of 15-shot
N_SUPPORT = 5
N_QUERY = 10
SUBSPACE_DIM = N_SUPPORT - 1  # = 4

%run resnet50plus_episodic_evaluation.py
```

## Compare Multiple Models

```python
# Evaluate ResNet50Plus
MODEL_PATH_PLUS = "/path/to/resnet50plus.pth"
%run resnet50plus_episodic_evaluation.py
# Rename OUTPUT_DIR to save results
!mv episodic_results episodic_results_resnet50plus

# Evaluate ResNet50 Normal
MODEL_PATH_PLUS = "/path/to/resnet50_normal.pth"
OUTPUT_DIR = './episodic_results_normal'
%run resnet50plus_episodic_evaluation.py

# Compare:
# - Check if 95% CIs overlap
# - Compare mean accuracies
# - Compare per-class performance
```

## Visualizations Explained

### 1. Overall Accuracy Distribution
- **Histogram**: Shows how episode accuracies are distributed
- **Red dashed line**: Mean accuracy
- **Orange dotted lines**: 95% confidence interval bounds
- **Boxplot**: Shows median, quartiles, and outliers

### 2. Per-Class Accuracy
- **Bars**: Mean accuracy per class
- **Error bars**: Standard deviation across episodes
- **Sorted**: Highest accuracy at top
- Helps identify which classes need improvement

### 3. Accuracy Trend
- **Blue line**: Individual episode accuracies
- **Red dashed line**: Mean across all episodes
- **Orange band**: 95% confidence interval
- Check for trends or outliers

## Troubleshooting

### Error: "CUDA out of memory"

```python
# Edit resnet50plus_episodic_evaluation.py, line ~320
# Change batch_size from 32 to 16
support_loader = DataLoader(support_dataset, batch_size=16, shuffle=False)
query_loader = DataLoader(query_dataset, batch_size=16, shuffle=False)
```

### Error: "subspace not defined"

Make sure you're running `resnet50plus_episodic_evaluation.py`, NOT the old single-split code.

### Very high std (>5%)

Could indicate:
- Classes with very different difficulties
- Classes with few samples
- Model not well-trained

Check per-class statistics to identify problematic classes.

### Results seem wrong

Verify:
1. Model loaded correctly: `print(model)`
2. Dataset loaded correctly: `print(len(full_dataset))`
3. Correct number of classes: `print(len(class_names))`

## Next Steps

### After Episodic Evaluation

1. **Analyze per-class results**
   - Which classes are hardest?
   - Why might they be hard?

2. **Try different configurations**
   - 5-shot vs 15-shot
   - Different query set sizes
   - Different subspace dimensions

3. **Generate Grad-CAM for worst classes**
   - Identify which features the model uses
   - Understand failure modes

4. **Compare with other models**
   - Run episodic evaluation on all model variants
   - Statistical comparison using confidence intervals

## Complete Example Workflow

```python
# Setup
import os
import torch
from torchvision import datasets, transforms

# Define paths
DATA_DIR_2 = "/kaggle/input/datasetold/Datasetold/Train"
MODEL_PATH_PLUS = "/kaggle/input/resnet50plus-31102025/pytorch/default/1/Resnet50plus_plant_disease_model_31102025.pth"
OUTPUT_DIR = './episodic_results'

# Configuration
N_SUPPORT = 15
N_QUERY = 10
SUBSPACE_DIM = N_SUPPORT - 1

# Run episodic evaluation
%run resnet50plus_episodic_evaluation.py

# Load and inspect results
import pandas as pd

summary = pd.read_csv(os.path.join(OUTPUT_DIR, 'episodic_summary.csv'))
print(summary)

per_class = pd.read_csv(os.path.join(OUTPUT_DIR, 'episodic_per_class_stats.csv'))
print(per_class.sort_values('mean', ascending=False).head(10))

# Worst performing classes
print("\nWorst performing classes:")
print(per_class.sort_values('mean', ascending=True).head(5))

# Most inconsistent classes (high variance)
print("\nMost inconsistent classes:")
print(per_class.sort_values('std', ascending=False).head(5))
```

## Reporting Results

### In Papers/Reports

```
We evaluated our model using standard episodic evaluation with 100 episodes.
For each episode, we randomly sampled 15 support samples and 10 query samples
per class. Our model achieved a mean accuracy of 87.34% ± 2.15% (95% CI:
[86.91%, 87.77%]), significantly outperforming the baseline of X%.
```

### In Presentations

```
ResNet50Plus Few-Shot Learning Results
- Configuration: 15-shot, 10 queries/class
- Episodes: 100
- Mean Accuracy: 87.34% ± 2.15%
- 95% CI: [86.91%, 87.77%]

[Show episodic_overall_accuracy.png]
[Show episodic_per_class_accuracy.png]
```

## Summary

✅ **Fast**: ~5 minutes for 100 episodes
✅ **Robust**: Accounts for sampling variability
✅ **Statistical**: Provides confidence intervals
✅ **Standard**: Matches few-shot learning literature
✅ **Comprehensive**: Overall + per-class statistics
✅ **Visual**: Automatic plot generation

Use episodic evaluation for all final results and model comparisons!
