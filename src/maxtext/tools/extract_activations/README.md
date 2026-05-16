# Activation Extraction

Offline tool that runs a MaxText model in prefill mode, captures
intermediate activations (post-block residual stream, MLP output,
attention output) via Flax `sow("intermediates", …)`, and writes them
to sharded safetensors files for SAE training and mechanistic
interpretability research.

**Documentation** lives in the main MaxText docs:

- [Activation Extraction overview](../../../../docs/guides/activation_extraction.md)
- [Usage guide](../../../../docs/guides/activation_extraction/usage.md) —
  quick start, CLI, output layout, `sparsify` / `sae_lens` integration.
- [Design](../../../../docs/guides/activation_extraction/design.md) —
  hook mechanism, multi-host correctness contract, MoE handling.

Quick start:

```bash
python -m maxtext.tools.extract_activations.main \
  src/maxtext/configs/extract_activations.yml \
  model_name=qwen3-30b-a3b \
  load_parameters_path=gs://your-bucket/path/items/0 \
  activation_extraction_output_path=gs://your-bucket/sae-data/v1 \
  activation_extraction_dataset_path=gs://your-bucket/pile_subset.jsonl
```

Tests:

```bash
pytest tests/unit/extract_activations/ tests/integration/extract_activations/
```
