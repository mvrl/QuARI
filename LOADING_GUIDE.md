# Model Loading Guide

## Summary of Changes

### Problem
The original checkpoint files contained both:
- `backbone_extractor.*` keys (CLIP/SigLIP model weights)
- `hypernetwork.*` keys (TransformerHypernetwork weights)

When loading the `TransformerHypernetwork` model, it expected keys without any prefix, causing a mismatch.

### Solution
Modified `hf_prep.py` to:
1. **Filter** the state dict to extract only `hypernetwork.*` keys
2. **Strip** the `hypernetwork.` prefix from these keys
3. **Save** the cleaned weights for direct loading

This eliminates loading overhead and reduces file size significantly (659MB → much smaller).

## Files Modified

### 1. `hf_prep.py`
- Added `filter_hypernetwork_weights()` function to extract and clean hypernetwork weights
- Modified checkpoint processing to automatically filter weights before saving
- Backward compatible - detects if filtering is needed

### 2. `load_model.py` (NEW)
- Simple utility script for loading TransformerHypernetwork models
- Automatically detects checkpoint format (legacy vs. new)
- Handles both formats transparently
- Includes basic usage example

### 3. `test_loading.py`
- Enhanced diagnostic script for debugging loading issues
- Attempts multiple loading strategies
- Provides detailed error messages

## Usage

### To Regenerate Checkpoint Files

If you have old checkpoint files with the mixed format:

```bash
# Regenerate a single checkpoint directory
python hf_prep.py --root_dir ./ckpts/clip-vit-base-patch16 --pattern "model.ckpt" --no-recursive

# Or regenerate all checkpoints
python hf_prep.py --root_dir ./ckpts --pattern "model.ckpt"
```

This will:
- Delete old `model_weights.pt` files
- Create new filtered `model_weights.pt` files (hypernetwork only, no prefix)
- Keep `model_hyperparams.yml` files unchanged

### To Load a Model

Simple approach using the utility:

```python
from load_model import load_hypernetwork

# Load model
model = load_hypernetwork("./ckpts/clip-vit-base-patch16/", device='cpu')
model.eval()

# Use model
import torch
query_emb = torch.randn(batch_size, 512)
with torch.no_grad():
    output = model(query_emb)
    refined_query = output['refined_query']
    W_image = output['W_image']
```

Manual approach:

```python
import torch
import yaml
from transformer_hypernetwork import TransformerHypernetwork

# Load hyperparameters
with open("./ckpts/clip-vit-base-patch16/model_hyperparams.yml", "r") as f:
    metadata = yaml.safe_load(f)

# Create model
model = TransformerHypernetwork(**metadata['hparams'])
model.feature_extractor = metadata['feature_extractor']

# Load weights (new format - direct loading)
checkpoint = torch.load("./ckpts/clip-vit-base-patch16/model_weights.pt", 
                        map_location='cpu')
model.load_state_dict(checkpoint['state_dict'], strict=True)
```

## Checkpoint Format

### Old Format (Legacy)
```python
{
    'state_dict': {
        'backbone_extractor.model.logit_scale': ...,
        'backbone_extractor.model.text_model...': ...,
        'backbone_extractor.model.vision_model...': ...,
        'hypernetwork.pos_emb': ...,
        'hypernetwork.query_encoder.0.weight': ...,
        # ... more hypernetwork keys with prefix
    }
}
```
Total: ~500+ keys, ~660MB

### New Format (Filtered)
```python
{
    'state_dict': {
        'pos_emb': ...,
        'pos_scale': ...,
        'query_encoder.0.weight': ...,
        'query_encoder.0.bias': ...,
        # ... 74 keys total, no prefix
    }
}
```
Total: 74 keys, significantly smaller file

## Benefits

1. **Faster Loading**: No filtering needed at runtime
2. **Smaller Files**: Only hypernetwork weights, not entire CLIP/SigLIP model
3. **Cleaner Code**: Direct loading without prefix manipulation
4. **Backward Compatible**: `load_model.py` handles both formats automatically

## Status

✅ `clip-vit-base-patch16`: Already converted to new format
- Run `hf_prep.py` on other checkpoint directories as needed

