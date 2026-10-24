# Model Architecture Summary

## 1. CT model: `ct_model_Swin_tiny`

- Input: one case is represented by 3 aligned CT slices, `max / above / below`, stacked as a 3-channel image.
- Backbone: `Swin-Tiny` from `timm`, usually frozen during training.
- Feature refinement: optional `SE`; current config enables `CBAM`.
- Pooling: spatial global average pooling on the feature map.
- Classifier: `768 -> 64 -> 8 -> 1` with `LayerNorm`, `GELU`, and `Dropout`.
- Task: binary recurrence prediction.

Code references:
- `ct_model_Swin_tiny/dataset.py`
- `ct_model_Swin_tiny/model.py`

## 2. Fusion model: `fusion_model`

- CT branch:
  - same 3-slice CT input as the CT-only model
  - `Swin-Tiny + optional CBAM`
  - pooled and projected from `768 -> 16`
- Pathology branch:
  - precomputed UNI2 slide feature
  - adapter `1536 -> 64`
- Clinical branch:
  - 11 clinical variables
  - adapter `11 -> 12`
- Fusion:
  - all three modalities projected to `d_model = 64`
  - 3 modality tokens with learnable modality embedding
  - 2-layer cross-modal Transformer encoder
  - clinical-token-driven gating:
    - gate logits from clinical token
    - temperature parameterized as `exp(log_temperature)`
    - gate = `sigmoid(logits / temperature)`
  - learned modality-weight aggregation
- Classifier:
  - `64 -> 32 -> 1`

Code references:
- `fusion_model/dataset.py`
- `fusion_model/model.py`

## 3. Pathology model: `pathology_model`

- Input: one UNI2 slide feature vector per patient, shape `1536`.
- Network: pure MLP.
- Classifier stack:
  - `1536 -> 512 -> 128 -> 32 -> 1`
  - each hidden layer uses `LayerNorm + ReLU + Dropout`
- Task: binary recurrence prediction.

Code references:
- `pathology_model/dataset.py`
- `pathology_model/model.py`

## Notes

- The current architecture diagrams are drawn from actual code, not only from each folder README.
- In `ct_model_Swin_tiny`, the README text is partly outdated. The current implementation is not a slice-aggregation Transformer over per-slice features; it directly treats the three slices as 3 channels and sends them into Swin-Tiny.
- In `fusion_model` and `pathology_model`, training code still uses test-set AUC for model selection. That is a training protocol issue, not a structure issue.
