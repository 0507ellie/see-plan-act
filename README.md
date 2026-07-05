# see-plan-act

Utilities for exploring vision-language-action research code with LIBERO demos
and CLIP language embeddings.

This branch currently contains a small LIBERO + CLIP CUDA toolkit under
`examples/libero_clip/`:

- Load a HuggingFace CLIP model on CUDA.
- Encode LIBERO-style language instructions into frozen 512-dimensional vectors.
- Preview a LIBERO Object HDF5 demonstration as images/video.

Large local artifacts are intentionally not committed:

- LIBERO datasets (`*.hdf5`)
- CLIP model weights
- generated embeddings (`*.npy`)
- generated preview videos/images under `outputs/`

See `examples/libero_clip/README.md` for setup and run commands.
