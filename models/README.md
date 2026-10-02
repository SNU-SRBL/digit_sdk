# Sparsh Models for Force Estimation

This directory contains pretrained Sparsh models for vision-based tactile force estimation.

## Models

### Encoder: `sparsh_dino_base_encoder.ckpt`
- **Architecture**: Vision Transformer (ViT-base)
- **Parameters**: 
  - Patch size: 16×16
  - Embedding dimension: 768
  - Depth: 12 layers
  - Attention heads: 12
- **Source**: `facebook/sparsh-dino-base` on HuggingFace
- **Training**: Self-supervised learning on tactile sensor data (DINO method)
- **Format**: PyTorch Lightning checkpoint
- **Size**: ~1.8 GB

### Decoder: `sparsh_digit_forcefield_decoder.pth`
- **Architecture**: Multi-scale DPT-style decoder
- **Outputs**: 
  - Normal force field: [224, 224, 1]
  - Shear force field: [224, 224, 2]
- **Source**: `facebook/sparsh-digit-forcefield-decoder` on HuggingFace (epoch 31)
- **Training**: Supervised on DIGIT sensor force data
- **Compatibility**: Requires sparsh-dino-base encoder (embed_dim=768)
- **Size**: ~15 MB

## Source dependency

These checkpoints are only the force-model artifacts. Force runtime also needs
a compatible [Sparsh source checkout](https://github.com/facebookresearch/sparsh)
at `../sparsh-main` relative to this directory (that is,
`digit_sdk/sparsh-main/`) or in the launch working directory. Install its
upstream dependencies in the Python environment that runs force estimation.
This optional dependency is unrelated to DIGIT tactile depth.

## Download

To download these models, run:

```bash
python scripts/download_models.py
```

This will automatically download both models from HuggingFace Hub.

## License

These models are released under **CC-BY-NC 4.0** (Creative Commons Attribution-NonCommercial 4.0 International).

**Usage restrictions**:
- ✓ Research and educational use
- ✓ Non-commercial applications
- ✗ Commercial use (requires separate licensing)

See https://github.com/facebookresearch/sparsh for more details.

## References

- **Sparsh Repository**: https://github.com/facebookresearch/sparsh
- **HuggingFace Collection**: https://huggingface.co/collections/facebook/sparsh-67167ce57566196a4526c328
- **Paper**: (Check Sparsh repository for latest publications)

## File Integrity

After downloading, verify files exist and are not corrupted:

```bash
ls -lh models/
# Should show:
# sparsh_dino_base_encoder.ckpt (~1.8 GB)
# sparsh_digit_forcefield_decoder.pth (~15 MB)
```

You can also run a verification check:

```bash
python scripts/download_models.py --check-only
```
