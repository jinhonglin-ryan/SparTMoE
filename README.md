# SparTMoE: Sparse Transformer Mixture-of-Experts

This repository contains experiments comparing different gating strategies in Mixture-of-Experts (MoE) models, with a focus on sparse and attention-based routing mechanisms.

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

The main dependencies include:
- PyTorch
- Transformers
- PEFT
- Accelerate
- Datasets
- Evaluate
- ROUGE score
- Entmax (for sparse activation functions)

## Project Structure

- `main_experiment.ipynb`: **Primary notebook** for running experiments with various gating strategies (recommended)
- `basline.ipynb`: Initial experimental attempts for standard MoE gating mechanisms
- `flops_comparison.ipynb`: Notebook for comparing computational efficiency (FLOPs) across different routing methods (top-k v.s. sparsemax)

## Experiments

### Running the Experiments

The experiments should be run through the `main_experiment.ipynb` Jupyter notebook. This notebook contains optimized code for implementing and evaluating different gating strategies.

### 1. Baseline Gating Strategies

To run the baseline gating experiments (as described in the paper):

#### Linear + Softmax + Top-k

This is the standard MoE gating mechanism that applies a linear projection to compute expert scores, followed by softmax normalization and top-k selection.

```python
# In main_experiment.ipynb
router_type = "linear"
norm_type = "softmax"
top_k = 2  # Set to the desired k value
```

#### Linear + Soft Routing (All Experts Weighted)

This applies softmax over all expert scores without top-k filtering, where all experts are activated and contribute to the final output.

```python
# In main_experiment.ipynb
router_type = "linear"
norm_type = "softmax"
top_k = None  # No top-k filtering - all experts are used
```

### 2. Proposed Gating Variants

#### Linear + Sparsemax

Replaces softmax with sparsemax applied to the linear gating outputs for differentiable sparse routing without requiring explicit top-k filtering.

```python
# In main_experiment.ipynb
router_type = "linear"
norm_type = "sparsemax"
top_k = None  # Sparsemax naturally produces sparse outputs
```

#### Attention-Based Gating + Softmax + Top-k

Computes expert relevance scores via scaled dot-product attention between the input and learned expert embeddings. The scores are passed through softmax and then top-k selection is applied.

```python
# In main_experiment.ipynb
router_type = "attention"
norm_type = "softmax"
top_k = 2  # Set to the desired k value
```

#### Attention-Based Gating + Soft Routing (All Experts Weighted)

Similar to the above, but without top-k filtering. All experts are softly weighted according to attention scores and contribute to the output.

```python
# In main_experiment.ipynb
router_type = "attention"
norm_type = "softmax"
top_k = None  # No top-k filtering - all experts are used
```

#### Attention-Based Gating + Sparsemax

Applies sparsemax instead of softmax to the attention-derived scores, promoting sparse yet differentiable expert activation.

```python
# In main_experiment.ipynb
router_type = "attention"
norm_type = "sparsemax"
top_k = None  # Sparsemax naturally produces sparse outputs
```

## Execution Environment

The experiments are designed to run on a GPU-enabled environment. The code automatically configures the appropriate GPU usage.

## Data

The experiments use the SAMSum dataset for summarization tasks. The dataset is automatically loaded from Hugging Face Datasets.

## Model

The base model is `google/switch-base-8`, which is a Switch Transformer with 8 experts per layer. The experiments modify the routing mechanisms of this model. We also scale to `google/switch-base-16` with 16 experts per layer. 

## Evaluation Metrics

The experiments are evaluated using ROUGE scores, which measure the overlap between the generated and reference summaries.

## Results

After running each experiment, the results are displayed in the notebook, showing the performance metrics for each gating strategy.

