#!/usr/bin/env python3
"""
Download Sparsh models from HuggingFace Hub for force estimation.

This script downloads the required Sparsh models:
- facebook/sparsh-dino-base: ViT-base encoder (embed_dim=768)
- facebook/sparsh-digit-forcefield-decoder: Force field decoder

Models are saved to the models/ directory.
"""

import sys
import argparse
from pathlib import Path
import hashlib

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("Error: Required packages not found.")
    print("Please install: pip install huggingface_hub")
    sys.exit(1)


# Model configurations
MODELS = {
    "encoder": {
        "repo_id": "facebook/sparsh-dino-base",
        "filename": "dino_vitbase.ckpt",
        "local_filename": "sparsh_dino_base_encoder.ckpt",
        "description": "Sparsh DINO-base encoder (ViT-base, embed_dim=768)",
    },
    "decoder": {
        "repo_id": "facebook/sparsh-digit-forcefield-decoder",
        "filename": "digit_t1_forcefield_dino_vitbase_bg/checkpoints/epoch-0031.pth",
        "local_filename": "sparsh_digit_forcefield_decoder.pth",
        "description": "Sparsh DIGIT force field decoder (epoch 31)",
    },
}


def compute_file_hash(filepath: Path, algorithm: str = "sha256") -> str:
    """
    Compute hash of a file for verification.

    Args:
        filepath: Path to file
        algorithm: Hash algorithm (sha256, md5, etc.)

    Returns
    -------
    Hex digest of file hash

    """
    hash_obj = hashlib.new(algorithm)
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()


def verify_model_file(filepath: Path) -> bool:
    """
    Verify that a downloaded model file is valid.

    Args:
        filepath: Path to model file

    Returns
    -------
    True if file appears valid (exists, non-empty, loadable header)

    """
    if not filepath.exists():
        return False

    if filepath.stat().st_size == 0:
        print(f"  ⚠ File is empty: {filepath}")
        return False

    # Basic check: try to read first few bytes to verify it's not corrupted
    try:
        with open(filepath, "rb") as f:
            header = f.read(8)
            # PyTorch files typically start with specific magic numbers
            if len(header) < 8:
                print(f"  ⚠ File too small: {filepath}")
                return False
    except Exception as e:
        print(f"  ⚠ Cannot read file: {e}")
        return False

    return True


def download_model(
    repo_id: str,
    filename: str,
    local_filename: str,
    models_dir: Path,
    force: bool = False,
) -> bool:
    """
    Download a model from HuggingFace Hub.

    Args:
        repo_id: HuggingFace repository ID
        filename: Filename in the repository
        local_filename: Local filename to save as
        models_dir: Directory to save models
        force: Force re-download even if file exists

    Returns
    -------
    True if download successful, False otherwise

    """
    local_path = models_dir / local_filename

    # Check if already exists
    if local_path.exists() and not force:
        if verify_model_file(local_path):
            print(f"  ✓ Already exists: {local_filename}")
            file_size_mb = local_path.stat().st_size / (1024 * 1024)
            print(f"    Size: {file_size_mb:.1f} MB")
            return True
        else:
            print("  ⚠ Existing file appears corrupted, re-downloading...")
            local_path.unlink()

    print(f"  → Downloading from {repo_id}/{filename}...")

    try:
        # Download to cache first
        cached_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=None,  # Use default cache
            resume_download=True,  # Resume if interrupted
        )

        # Copy to models directory
        import shutil
        shutil.copy2(cached_path, local_path)

        # Verify download
        if not verify_model_file(local_path):
            print("  ✗ Download verification failed")
            return False

        file_size_mb = local_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ Downloaded: {local_filename} ({file_size_mb:.1f} MB)")

        # Compute hash for reference
        file_hash = compute_file_hash(local_path)
        print(f"    SHA256: {file_hash[:16]}...")

        return True

    except Exception as e:
        print(f"  ✗ Download failed: {e}")
        if local_path.exists():
            local_path.unlink()
        return False


def check_models_exist(models_dir: Path) -> dict:
    """
    Check which models already exist.

    Args:
        models_dir: Directory containing models

    Returns
    -------
    Dict mapping model keys to existence status

    """
    status = {}
    for key, config in MODELS.items():
        local_path = models_dir / config["local_filename"]
        status[key] = local_path.exists() and verify_model_file(local_path)
    return status


def main():
    parser = argparse.ArgumentParser(
        description="Download Sparsh models for DIGIT force estimation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all models
  python scripts/download_models.py

  # Force re-download even if files exist
  python scripts/download_models.py --force

  # Download to custom directory
  python scripts/download_models.py --models-dir /path/to/models
        """,
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="Directory to save models (default: models/ in SDK root)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if models already exist",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check which models exist, don't download",
    )

    args = parser.parse_args()

    # Determine models directory
    if args.models_dir:
        models_dir = args.models_dir
    else:
        # Assume script is in scripts/ directory
        script_dir = Path(__file__).parent
        models_dir = script_dir.parent / "models"

    models_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Sparsh Model Downloader for DIGIT SDK")
    print("=" * 70)
    print(f"\nModels directory: {models_dir.absolute()}\n")

    # Check existing models
    existing = check_models_exist(models_dir)

    if args.check_only:
        print("Model Status:")
        for key, config in MODELS.items():
            status = "✓ EXISTS" if existing[key] else "✗ MISSING"
            print(f"  [{status}] {config['description']}")
            print(f"           {config['local_filename']}")
        print()
        return

    # Download models
    print("Downloading models...\n")

    success_count = 0
    for key, config in MODELS.items():
        print(f"[{key.upper()}] {config['description']}")

        success = download_model(
            repo_id=config["repo_id"],
            filename=config["filename"],
            local_filename=config["local_filename"],
            models_dir=models_dir,
            force=args.force,
        )

        if success:
            success_count += 1
        print()

    # Summary
    print("=" * 70)
    total = len(MODELS)
    if success_count == total:
        print(f"✓ SUCCESS: All {total} models ready")
        print(f"\nModels saved to: {models_dir.absolute()}")
        print("\nYou can now use force estimation in DIGIT SDK!")
    else:
        print(f"⚠ PARTIAL: {success_count}/{total} models downloaded")
        print("\nSome downloads failed. Please check errors above and retry.")
        sys.exit(1)


if __name__ == "__main__":
    main()
