# LIBERO CLIP Utilities

This folder contains small scripts for exploring LIBERO Object instructions and
encoding them with HuggingFace CLIP on CUDA.

## Expected local setup

The scripts default to the paths used on this Windows machine:

- CLIP model: `D:\huggingface_models\openai_clip-vit-base-patch32`
- LIBERO Object dataset: `D:\LIBERO\datasets\libero_object`

You can override them with environment variables:

```powershell
$env:CLIP_MODEL_PATH="D:\huggingface_models\openai_clip-vit-base-patch32"
$env:LIBERO_OBJECT_DATASET_DIR="D:\LIBERO\datasets\libero_object"
```

## Install dependencies

Use the existing `libero_cuda` conda environment:

```powershell
conda activate libero_cuda
pip install -r examples/libero_clip/requirements.txt
```

## Run

Load CLIP on CUDA:

```powershell
python examples/libero_clip/load_clip_hf_cuda.py
```

Encode sample LIBERO instructions:

```powershell
python examples/libero_clip/encode_libero_clip_instructions.py
```

Use the frozen text encoder wrapper:

```powershell
python examples/libero_clip/clip_text_encoder.py
```

Preview a LIBERO Object demo:

```powershell
python examples/libero_clip/preview_libero_object_demo.py
```

Generated files are written under `outputs/` by default and are ignored by git.
