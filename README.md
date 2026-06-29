# See-Plan-Act

Observation extraction and policy learning pipeline for LIBERO robotic manipulation tasks.

## Project structure

```
see-plan-act/
  obs.py          # Data collection + visual encoding (random rollouts)
  train_bc.py     # Behavior cloning training
  eval.py         # Shared evaluation (works with any policy)
```

## Cameras

- **agentview** — Fixed third-person camera overlooking the workspace (global scene context)
- **eye_in_hand** — Wrist-mounted camera on the end-effector (close-up detail for grasping)

Both cameras capture RGBD (RGB + depth as 4th channel).

## Encoders

| Encoder | Model | Embedding Dim | Applied to |
|---------|-------|---------------|------------|
| ResNet  | ResNet18 (ImageNet pretrained, fc removed) | 512 | RGB only |
| CLIP    | ViT-B/32 | 512 | RGB only |

Depth is stored raw as the 4th channel of RGBD — intended for a learned encoder in the policy, not pretrained RGB encoders.

## Policies

### Behavior Cloning (BC)

3-layer MLP trained with MSE on expert demonstrations.

**Input:** CLIP agentview (512) + CLIP eye_in_hand (512) + proprio (8) = 1032-d
- Proprio: ee_pos (3) + ee_ori (3, axis-angle) + gripper (2)
- Actions are normalized (zero mean, unit std)

**Output:** 7-DoF action (6 arm + 1 gripper)

## Usage

### 1. Collect observations (random rollouts)

```bash
/workspace/envs/libero/bin/python3.10 obs.py
```

### 2. Train BC

```bash
/workspace/envs/libero/bin/python3.10 train_bc.py \
  --demo_path /workspace/LIBERO/libero/datasets/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5 \
  --epochs 100 \
  --batch_size 64 \
  --lr 1e-3
```

Saves best checkpoint to `checkpoints/bc_best.pt`.

### 3. Evaluate

```bash
/workspace/envs/libero/bin/python3.10 eval.py \
  --policy bc \
  --checkpoint checkpoints/bc_best.pt \
  --task_id 0
```

Saves per-episode GIFs to `eval_gifs/` (e.g. `ep0_success.gif`, `ep1_fail.gif`).

### Adding a new policy

1. Create `train_<name>.py` with a policy class (must have `forward(visual, proprio) -> action`)
2. Add an entry to `POLICY_REGISTRY` in `eval.py`
3. Eval works automatically

## obs.py outputs

- `rollout_data.hdf5` — Full rollout data (actions, rewards, dones, proprioception, RGBD)
- `agentview_embed.npy` / `eye_in_hand_embed.npy` — ResNet18 embeddings
- `agentview_clip_embed.npy` / `eye_in_hand_clip_embed.npy` — CLIP embeddings
- `rollout.gif` — Animated GIF of agentview rollout

### HDF5 structure

```
data/
  attrs: task, num_steps, resnet_encoder, resnet_embed_dim, clip_encoder, clip_embed_dim
  actions            (N, 7)
  rewards            (N,)
  dones              (N,)
  obs/
    ee_pos           (N, 3)
    ee_quat          (N, 4)
    joint_pos        (N, 7)
    joint_pos_cos    (N, 7)
    joint_pos_sin    (N, 7)
    joint_vel        (N, 7)
    gripper_qpos     (N, 2)
    gripper_qvel     (N, 2)
    agentview_rgbd   (N, 256, 256, 4)
    eye_in_hand_rgbd (N, 256, 256, 4)
```

## Dependencies

- libero
- torch, torchvision
- clip (OpenAI)
- robosuite
- h5py, numpy, Pillow
