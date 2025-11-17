# Episodic Evaluation Guide for ResNet50Plus

## Overview

This guide explains the episodic evaluation approach for few-shot learning, which provides more robust and statistically meaningful results than single train/test splits.

## What is Episodic Evaluation?

### Traditional Evaluation (Single Split)
- Split dataset once into support and query sets
- Train/compute subspaces on support set
- Evaluate on query set
- Get a single accuracy number
- **Problem**: Results depend heavily on the random split

### Episodic Evaluation (100 Episodes)
- Run 100 independent episodes
- Each episode:
  1. Randomly sample new support set (N_SUPPORT samples per class)
  2. Randomly sample new query set (N_QUERY samples per class)
  3. Compute subspaces from support set
  4. Evaluate on query set
  5. Record accuracy
- Compute statistics across all episodes:
  - Mean accuracy
  - Standard deviation
  - 95% confidence intervals
  - Min/max accuracy
  - Per-class statistics

**Benefit**: More robust measure of model performance that accounts for sampling variability

## File Structure

### Main Script
**`resnet50plus_episodic_evaluation.py`** - Complete episodic evaluation implementation

### Key Components

1. **Episode Creation** (`create_episode_split`)
   - Randomly samples support and query sets for each episode
   - Uses different random seed for each episode
   - Ensures balanced sampling across classes

2. **Subspace Computation** (`compute_subspaces`)
   - Computes DSN subspaces for each episode
   - Uses SVD on centered support embeddings
   - Handles variable support set sizes

3. **Episode Evaluation** (`evaluate_episode`)
   - Evaluates query set using DSN distance metric
   - Returns predictions and targets

4. **Single Episode Runner** (`run_single_episode`)
   - Orchestrates one complete episode
   - Returns episode results including per-class accuracy

## Usage

### Prerequisites

Define these variables before running:

```python
DATA_DIR_2 = "/path/to/dataset"
MODEL_PATH_PLUS = "/path/to/resnet50plus.pth"
OUTPUT_DIR = "./episodic_results"
N_SUPPORT = 15  # Support samples per class
N_QUERY = 10    # Query samples per class
SUBSPACE_DIM = N_SUPPORT - 1  # Typically N_SUPPORT - 1
```

### Running the Evaluation

```python
# Option 1: Run the script directly
python resnet50plus_episodic_evaluation.py

# Option 2: Import in notebook
%run resnet50plus_episodic_evaluation.py

# Option 3: Execute in notebook cell
exec(open('resnet50plus_episodic_evaluation.py').read())
```

### Expected Runtime

- **Per Episode**: ~2-5 seconds (depending on dataset size and GPU)
- **100 Episodes**: ~3-8 minutes total
- Progress bar shows real-time progress

## Output

### Console Output

```
OVERALL ACCURACY STATISTICS (100 episodes)
======================================================================
Mean accuracy:        87.34%
Std deviation:        2.15%
Min accuracy:         81.20%
Max accuracy:         92.50%
Median accuracy:      87.60%
95% Confidence Int:   [86.91%, 87.77%]

PER-CLASS ACCURACY STATISTICS
======================================================================
Class                                         Mean      Std      Min      Max
----------------------------------------------------------------------
Apple___Apple_scab                           89.2%     3.4%    82.0%    95.0%
Apple___Black_rot                            85.7%     4.1%    76.0%    93.0%
...
```

### Saved Files

All files saved to `OUTPUT_DIR`:

1. **`episodic_summary.csv`**
   - Overall statistics summary
   - Mean, std, min, max, median, confidence intervals

2. **`episodic_per_class_stats.csv`**
   - Per-class statistics across all episodes
   - Mean, std, min, max for each class

3. **`episodic_all_results.csv`**
   - Individual episode results
   - Episode number and accuracy for each episode

4. **`episodic_overall_accuracy.png`**
   - Histogram of accuracy distribution
   - Box plot of accuracies
   - Mean and 95% CI marked

5. **`episodic_per_class_accuracy.png`**
   - Horizontal bar chart
   - Mean accuracy per class with error bars (std)
   - Sorted by mean accuracy

6. **`episodic_accuracy_trend.png`**
   - Line plot showing episode-wise accuracy
   - Mean accuracy line
   - 95% confidence interval band

## Statistical Interpretation

### Mean Accuracy
The average accuracy across all 100 episodes. This is your primary performance metric.

```
Mean accuracy: 87.34%
```

### Standard Deviation
Measures variability in accuracy across episodes. Lower is better (more consistent).

```
Std deviation: 2.15%
```
- **< 2%**: Very stable performance
- **2-5%**: Moderate variability
- **> 5%**: High variability (may indicate dataset or model issues)

### 95% Confidence Interval
We can be 95% confident that the true population accuracy falls within this range.

```
95% CI: [86.91%, 87.77%]
```

Narrower intervals indicate more precise estimates.

### Per-Class Statistics
Shows which classes are:
- **Easy**: High mean, low std
- **Hard**: Low mean, high std
- **Inconsistent**: High std (accuracy varies across episodes)

## Comparison with Single-Split Evaluation

| Aspect | Single Split | Episodic (100 episodes) |
|--------|--------------|-------------------------|
| Robustness | Low | High |
| Statistical validity | Weak | Strong |
| Confidence intervals | No | Yes |
| Runtime | Fast (~5 sec) | Slower (~5 min) |
| Recommended for | Quick testing | Final evaluation |

## Customization

### Change Number of Episodes

```python
NUM_EPISODES = 200  # More episodes = more robust, but slower
```

Recommended values:
- **Development/testing**: 10-20 episodes
- **Publication/final results**: 100-1000 episodes
- **Standard practice**: 100 episodes

### Change Confidence Level

```python
CONFIDENCE_LEVEL = 0.99  # 99% CI instead of 95%
```

### Modify Support/Query Sizes

Already defined via global variables:
```python
N_SUPPORT = 5   # 5-shot instead of 15-shot
N_QUERY = 20    # 20 query samples instead of 10
```

### Save Additional Metrics

Add to `run_single_episode()`:

```python
# Add confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(targets, preds)

return {
    'episode': episode_num,
    'overall_accuracy': overall_acc,
    'per_class_accuracy': per_class_acc,
    'confusion_matrix': cm,  # Added
    'predictions': preds,
    'targets': targets
}
```

## Troubleshooting

### Memory Issues

If running out of GPU memory:

```python
# Reduce batch size in DataLoader
support_loader = DataLoader(support_dataset, batch_size=16, shuffle=False)  # Was 32
query_loader = DataLoader(query_dataset, batch_size=16, shuffle=False)
```

### Slow Execution

- Reduce `NUM_EPISODES` for testing
- Use GPU if available
- Reduce batch size if memory allows larger batches

### Classes with Insufficient Samples

If some classes have < N_SUPPORT + N_QUERY samples:

```python
# The code automatically handles this:
if len(indices) < total_needed:
    n_sup = min(n_support, len(indices) // 2)
    n_qry = len(indices) - n_sup
```

But for better results, ensure all classes have at least N_SUPPORT + N_QUERY samples.

## Best Practices

1. **Always use episodic evaluation for final results**
   - Single-split is fine for debugging
   - Report episodic results in papers/reports

2. **Report mean ± std or mean with 95% CI**
   ```
   Accuracy: 87.34% ± 2.15%
   or
   Accuracy: 87.34% (95% CI: [86.91%, 87.77%])
   ```

3. **Run at least 100 episodes**
   - Standard in few-shot learning literature
   - Provides reliable statistical estimates

4. **Check per-class statistics**
   - Identify problematic classes
   - Understand model strengths/weaknesses

5. **Visualize results**
   - Use generated plots to understand distribution
   - Check for outliers or bimodal distributions

## Example Results Interpretation

```
Mean accuracy:        87.34% ± 2.15%
95% CI:               [86.91%, 87.77%]
```

**Interpretation**:
- On average, the model achieves 87.34% accuracy on novel tasks
- Performance is quite stable (std = 2.15%)
- We're 95% confident true accuracy is between 86.91% and 87.77%
- This is significantly better than random guessing (for 38 classes, random = 2.63%)

## Comparison with Other Models

To compare ResNet50Plus with other models:

1. Run episodic evaluation for each model
2. Compare mean accuracies
3. Check if 95% CIs overlap:
   - **No overlap**: Statistically significant difference
   - **Overlap**: Difference may not be significant

Example:
```
Model A: 87.34% (95% CI: [86.91%, 87.77%])
Model B: 85.20% (95% CI: [84.75%, 85.65%])
Result: No overlap → Model A is significantly better
```

## References

### Few-Shot Learning Papers Using Episodic Evaluation

1. **Prototypical Networks** (Snell et al., 2017)
   - Standard episodic evaluation protocol
   - 600 episodes for meta-learning

2. **MAML** (Finn et al., 2017)
   - 600 episodes for testing
   - Reports mean ± 95% CI

3. **DSN** (Simon et al., 2020)
   - Uses episodic evaluation
   - Reports mean accuracy with confidence intervals

### Why 100 Episodes?

Based on statistical power analysis:
- 100 episodes provides good balance between accuracy and runtime
- Standard error decreases as 1/√n
- 100 episodes → SE ≈ 0.1 * std
- Doubling to 200 only reduces SE by ~30%

## Integration with Existing Code

### Replace Single-Split Evaluation

**Before** (single split):
```python
support_indices, query_indices = create_support_query_split(...)
# ... evaluate once ...
print(f"Accuracy: {acc*100:.2f}%")
```

**After** (episodic):
```python
%run resnet50plus_episodic_evaluation.py
# Results automatically computed and saved
```

### Use with Grad-CAM

After episodic evaluation, you can still run Grad-CAM on the best episode:

```python
# Find best episode
best_episode = max(episode_results, key=lambda x: x['overall_accuracy'])

# Use that episode's support/query split for Grad-CAM
# (You'd need to save and load the split)
```

## Summary

- **What**: Run 100 independent few-shot learning episodes
- **Why**: More robust and statistically valid results
- **How**: Use `resnet50plus_episodic_evaluation.py`
- **Output**: Mean accuracy, std, 95% CI, visualizations
- **Runtime**: ~5 minutes for 100 episodes
- **Recommended**: Always use for final evaluation and reporting

For quick testing, use single-split evaluation. For publication or final results, always use episodic evaluation with 100+ episodes.
