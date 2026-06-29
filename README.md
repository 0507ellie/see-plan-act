# See-Plan-Act

Behavioral cloning pipeline for LIBERO robotic manipulation with fine-tuned visual encoders, LSTM temporal memory, transformer action decoding, and action chunking.

## Architecture

```
Per step:
  camera images ──→ ResNet18 (layer3+4 fine-tuned) ──→ 2 visual tokens (512-d)
  camera images ──→ CLIP ViT-B/32 (frozen + LoRA) ──→ 2 visual tokens (512-d)
  proprio (9-d) ──→ linear projection ──→ 1 token
  language instr ──→ CLIP text (frozen + LoRA) ──→ 1 token

  all 6 tokens ──→ LSTM (2-layer, 512 hidden) ──→ compressed history token

  [history, vis x4, proprio, lang] ──→ transformer decoder ──→ 8-step action chunk
```

## Key features

- **Action chunking** (k=8): predicts 8 future actions per step for better trajectory planning
- **ResNet fine-tuning**: last 2 layers unfrozen with 10x lower LR to learn manipulation-relevant spatial features
- **LSTM temporal memory**: maintains hidden state across steps so the policy remembers what it's been doing
- **LoRA adapters**: lightweight adaptation of frozen CLIP visual/language features
- **L1 loss**: sharper action predictions vs MSE
- **Cosine LR with warmup**: 5-epoch warmup, cosine decay from 1e-4 to 1e-5
- **K-fold cross-validation**: task-level folds to evaluate cross-task generalization
- **Visual augmentation**: ColorJitter + RandomAffine on images before ResNet encoding
- **Auxiliary done prediction**: transformer predicts task completion as a secondary objective

## Project structure

```
see-plan-act/
  train_bc.py     # Training pipeline (BCPolicy, DemoDataset, k-fold CV)
  eval.py         # Evaluation with rollout GIF generation
  obs.py          # Data collection + visual encoding (random rollouts)
```

## Cameras

- **agentview** — Fixed third-person camera overlooking the workspace
- **eye_in_hand** — Wrist-mounted camera on the end-effector

## Encoders

| Encoder | Model | Dim | Status |
|---------|-------|-----|--------|
| ResNet  | ResNet18 (ImageNet init) | 512 | layer3+4 fine-tuned, layer1+2 frozen |
| CLIP image | ViT-B/32 | 512 | Frozen + LoRA adapter (rank=32) |
| CLIP text | ViT-B/32 | 512 | Frozen + LoRA adapter (rank=32) |

## Input/Output

**Inputs:**
- CLIP visual embeddings: agentview (512) + eye_in_hand (512) = 1024-d (pre-computed)
- Raw images: agentview + eye_in_hand (for ResNet, processed on-the-fly)
- Proprio (9-d): ee_pos (3) + ee_ori (3, axis-angle) + gripper_states (2) + gripper_open (1, binary)
- Language: CLIP text embedding of task instruction (512-d)

**Output:** 8-step action chunk, each action is 7-DoF (6 arm + 1 gripper)

## Usage

### Train

```bash
# Train on 8 tasks, hold out 2 for testing (skip cross-validation)
python train_bc.py \
  --demo_dir ../../libero/datasets/libero_object \
  --held_out_tasks 0 1 \
  --skip_folds \
  --epochs 200

# Full training with 4-fold cross-validation
python train_bc.py \
  --demo_dir ../../libero/datasets/libero_object \
  --held_out_tasks 0 1 \
  --num_folds 4 \
  --epochs 200
```

Saves best checkpoint to `checkpoints/bc_best.pt`.

### Evaluate

```bash
python eval.py \
  --policy bc \
  --checkpoint checkpoints/bc_best.pt \
  --suite libero_object \
  --task_id 0 \
  --num_episodes 20
```

Saves per-episode GIFs to `eval_gifs/` (e.g. `ep0_success.gif`, `ep1_fail.gif`).

### Task ID mapping (libero_object)

| Task ID | Task |
|---------|------|
| 0 | pick up the alphabet soup and place it in the basket |
| 1 | pick up the cream cheese and place it in the basket |
| 2 | pick up the salad dressing and place it in the basket |
| 3 | pick up the bbq sauce and place it in the basket |
| 4 | pick up the ketchup and place it in the basket |
| 5 | pick up the tomato sauce and place it in the basket |
| 6 | pick up the butter and place it in the basket |
| 7 | pick up the milk and place it in the basket |
| 8 | pick up the chocolate pudding and place it in the basket |
| 9 | pick up the orange juice and place it in the basket |

### Training arguments

| Arg | Default | Description |
|-----|---------|-------------|
| `--demo_dir` | required | Path to demo HDF5 directory |
| `--held_out_tasks` | 0 1 | File indices to hold out for testing |
| `--num_folds` | 4 | Number of cross-validation folds |
| `--skip_folds` | false | Skip CV, train final model only |
| `--chunk_size` | 8 | Number of future actions to predict |
| `--seq_len` | 10 | Temporal window length for LSTM |
| `--hidden_dim` | 256 | Transformer/projection hidden dimension |
| `--batch_size` | 64 | Batch size |
| `--lr` | 1e-4 | Base learning rate (ResNet uses lr*0.1) |
| `--epochs` | 200 | Max training epochs |
| `--patience` | 25 | Early stopping patience |
| `--aux_weight` | 0.1 | Weight for auxiliary done prediction loss |

## Dependencies

- libero, robosuite
- torch, torchvision
- clip (OpenAI)
- h5py, numpy, Pillow
