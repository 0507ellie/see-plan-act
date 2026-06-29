import argparse
import math
import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import clip
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESNET_PREPROCESS = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

AUG_TRANSFORM = T.Compose([
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    T.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
])


class LoRAAdapter(nn.Module):
    def __init__(self, dim, rank=32):
        super().__init__()
        self.down = nn.Linear(dim, rank)
        self.up = nn.Linear(rank, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.up(torch.relu(self.down(x)))


class BCPolicy(nn.Module):
    def __init__(self, clip_dim=512, proprio_dim=9, lang_dim=512,
                 action_dim=7, chunk_size=8, hidden_dim=256,
                 num_layers=4, num_heads=4, dropout=0.1,
                 rnn_hidden=512, rnn_layers=2, lora_rank=32):
        super().__init__()
        self.chunk_size = chunk_size
        self.clip_dim = clip_dim

        # ResNet18 with last 2 layers unfrozen
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        resnet.fc = nn.Identity()
        for name, param in resnet.named_parameters():
            param.requires_grad = ('layer3' in name or 'layer4' in name)
        self.resnet = resnet
        resnet_dim = 512

        # LoRA adapter for frozen CLIP features
        self.clip_adapter = LoRAAdapter(clip_dim, rank=lora_rank)
        self.lang_adapter = LoRAAdapter(lang_dim, rank=lora_rank)

        num_visual_tokens = 4  # 2 clip + 2 resnet
        self.visual_proj = nn.Linear(clip_dim, hidden_dim)
        self.resnet_proj = nn.Linear(resnet_dim, hidden_dim)
        self.proprio_proj = nn.Linear(proprio_dim, hidden_dim)
        self.lang_proj = nn.Linear(lang_dim, hidden_dim)

        # LSTM: processes projected features over time
        num_input_tokens = num_visual_tokens + 2
        self.lstm = nn.LSTM(
            input_size=hidden_dim * num_input_tokens,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            dropout=dropout if rnn_layers > 1 else 0.0,
        )
        self.context_proj = nn.Linear(rnn_hidden, hidden_dim)

        # Transformer decoder
        num_memory_tokens = 1 + num_visual_tokens + 2
        self.memory_pos_embed = nn.Parameter(torch.randn(1, num_memory_tokens, hidden_dim) * 0.02)
        self.action_queries = nn.Parameter(torch.randn(1, chunk_size, hidden_dim) * 0.02)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.done_head = nn.Linear(hidden_dim, 1)

        self._h = None
        self._c = None

    def reset_hidden(self):
        self._h = None
        self._c = None

    def forward(self, clip_embed, images, proprio, lang, reset_hidden=True):
        # clip_embed: (B, T, 1024), images: (B, T, 2, 3, 224, 224)
        # proprio: (B, T, proprio_dim), lang: (B, 512)
        B, T = clip_embed.shape[0], clip_embed.shape[1]

        # ResNet on raw images
        img_flat = images.reshape(B * T * 2, *images.shape[-3:])
        resnet_out = self.resnet(img_flat)
        resnet_tokens = self.resnet_proj(resnet_out.reshape(B * T, 2, -1))

        # CLIP tokens (frozen + adapted)
        clip_tokens = clip_embed.reshape(B * T, 2, self.clip_dim)
        clip_tokens = self.visual_proj(self.clip_adapter(clip_tokens))

        # Combine: [clip_ag, clip_eih, resnet_ag, resnet_eih]
        vis_proj = torch.cat([clip_tokens, resnet_tokens], dim=1)

        pro_proj = self.proprio_proj(proprio.reshape(B * T, -1))
        lang_exp = lang.unsqueeze(1).expand(-1, T, -1).reshape(B * T, -1)
        lang_proj = self.lang_proj(self.lang_adapter(lang_exp))

        # LSTM
        lstm_in = torch.cat([
            vis_proj.reshape(B * T, -1), pro_proj, lang_proj,
        ], dim=-1).reshape(B, T, -1)

        if reset_hidden or self._h is None:
            lstm_out, (self._h, self._c) = self.lstm(lstm_in)
        else:
            lstm_out, (self._h, self._c) = self.lstm(lstm_in, (self._h, self._c))
        self._h = self._h.detach()
        self._c = self._c.detach()

        context = self.context_proj(lstm_out).reshape(B * T, 1, -1)

        memory = torch.cat([
            context, vis_proj, pro_proj.unsqueeze(1), lang_proj.unsqueeze(1),
        ], dim=1)
        memory = memory + self.memory_pos_embed

        queries = self.action_queries.expand(B * T, -1, -1)
        decoded = self.decoder(queries, memory)

        actions = self.action_head(decoded).reshape(B, T, self.chunk_size, -1)
        done_logits = self.done_head(decoded[:, -1, :]).reshape(B, T, 1)
        return actions, done_logits


class DemoDataset(Dataset):
    """CLIP features pre-computed, raw images stored for ResNet fine-tuning."""

    def __init__(self, demo_paths, clip_model, clip_preprocess,
                 chunk_size=8, seq_len=10, augment=False):
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        self.augment = augment
        self.chunk_size = chunk_size
        self.seq_len = seq_len

        demo_clip = []
        demo_agentview = []
        demo_eye_in_hand = []
        demo_proprio = []
        demo_actions = []
        demo_lang = []
        demo_dones = []

        for demo_path in demo_paths:
            with h5py.File(demo_path, "r") as f:
                task_name = f["data"].attrs.get("task_name", "")
                if not task_name:
                    task_name = Path(demo_path).stem.replace("_demo", "").replace("_", " ")
                lang_embed = self._encode_text(task_name)
                print(f"  Loading {Path(demo_path).stem} ...")

                demos = sorted([k for k in f["data"].keys() if k.startswith("demo")])
                for demo_key in demos:
                    demo = f[f"data/{demo_key}"]
                    agentview = demo["obs/agentview_rgb"][:]
                    eye_in_hand = demo["obs/eye_in_hand_rgb"][:]
                    ee_pos = demo["obs/ee_pos"][:]
                    ee_ori = demo["obs/ee_ori"][:]
                    gripper = demo["obs/gripper_states"][:]
                    act = demo["actions"][:]
                    dones = demo["dones"][:]

                    gripper_open = (gripper[:, 0] > 0.03).astype(np.float32)
                    T_len = len(act)

                    clip_list, pro_list, act_list, don_list = [], [], [], []
                    for i in range(T_len):
                        ag_clip = self._encode_clip(agentview[i])
                        eih_clip = self._encode_clip(eye_in_hand[i])
                        clip_list.append(np.concatenate([ag_clip, eih_clip]))

                        pro_list.append(np.concatenate([ee_pos[i], ee_ori[i], gripper[i], [gripper_open[i]]]))

                        chunk = act[i:i + chunk_size]
                        if len(chunk) < chunk_size:
                            chunk = np.concatenate([chunk, np.tile(chunk[-1:], (chunk_size - len(chunk), 1))])
                        act_list.append(chunk)

                        done_chunk = dones[i:i + chunk_size]
                        don_list.append(float(done_chunk.max()))

                    demo_clip.append(np.array(clip_list, dtype=np.float32))
                    demo_agentview.append(agentview)
                    demo_eye_in_hand.append(eye_in_hand)
                    demo_proprio.append(np.array(pro_list, dtype=np.float32))
                    demo_actions.append(np.array(act_list, dtype=np.float32))
                    demo_lang.append(lang_embed.astype(np.float32))
                    demo_dones.append(np.array(don_list, dtype=np.float32))

        all_actions_flat = np.concatenate([a.reshape(-1, act.shape[-1]) for a in demo_actions])
        self.action_mean = all_actions_flat.mean(axis=0)
        self.action_std = all_actions_flat.std(axis=0) + 1e-8
        for i in range(len(demo_actions)):
            demo_actions[i] = (demo_actions[i] - self.action_mean) / self.action_std

        self.demo_clip = demo_clip
        self.demo_agentview = demo_agentview
        self.demo_eye_in_hand = demo_eye_in_hand
        self.demo_proprio = demo_proprio
        self.demo_actions = demo_actions
        self.demo_lang = demo_lang
        self.demo_dones = demo_dones

        self.proprio = np.concatenate(demo_proprio)

        self.windows = []
        for d_idx in range(len(demo_clip)):
            for t in range(len(demo_clip[d_idx])):
                self.windows.append((d_idx, t))

    def _encode_text(self, text):
        tokens = clip.tokenize([text]).to(device)
        with torch.no_grad():
            return self.clip_model.encode_text(tokens).squeeze(0).float().cpu().numpy()

    def _encode_clip(self, image):
        x = self.clip_preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
        with torch.no_grad():
            return self.clip_model.encode_image(x).squeeze(0).float().cpu().numpy()

    def _preprocess_image(self, img_uint8):
        pil = Image.fromarray(img_uint8)
        if self.augment:
            pil = AUG_TRANSFORM(pil)
        return RESNET_PREPROCESS(pil)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        d_idx, start_t = self.windows[idx]
        T_demo = len(self.demo_clip[d_idx])
        end_t = min(start_t + self.seq_len, T_demo)
        actual = end_t - start_t

        w_clip = self.demo_clip[d_idx][start_t:end_t]
        w_ag = self.demo_agentview[d_idx][start_t:end_t]
        w_eih = self.demo_eye_in_hand[d_idx][start_t:end_t]
        w_pro = self.demo_proprio[d_idx][start_t:end_t]
        w_act = self.demo_actions[d_idx][start_t:end_t]
        w_don = self.demo_dones[d_idx][start_t:end_t]
        lng = self.demo_lang[d_idx]

        # Preprocess images for ResNet (with augmentation)
        img_tensors = []
        for t in range(actual):
            img_tensors.append(self._preprocess_image(w_ag[t]))
            img_tensors.append(self._preprocess_image(w_eih[t]))

        # Pad if needed
        if actual < self.seq_len:
            pad = self.seq_len - actual
            w_clip = np.pad(w_clip, ((0, pad), (0, 0)), mode='edge')
            w_pro = np.pad(w_pro, ((0, pad), (0, 0)), mode='edge')
            w_act = np.pad(w_act, ((0, pad), (0, 0), (0, 0)), mode='edge')
            w_don = np.pad(w_don, (0, pad), mode='edge')
            last_ag = img_tensors[-2]
            last_eih = img_tensors[-1]
            for _ in range(pad):
                img_tensors.append(last_ag)
                img_tensors.append(last_eih)

        images = torch.stack(img_tensors).reshape(self.seq_len, 2, 3, 224, 224)

        return (
            torch.from_numpy(w_clip),
            images,
            torch.from_numpy(w_pro),
            torch.from_numpy(lng.copy()),
            torch.from_numpy(w_act),
            torch.from_numpy(w_don),
        )


class DemoSubset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        self.proprio = dataset.proprio
        self.action_mean = dataset.action_mean
        self.action_std = dataset.action_std

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


def cosine_lr(epoch, warmup_epochs, total_epochs, base_lr, min_lr):
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def train_one(train_dataset, val_dataset, args, save_path):
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, pin_memory=True)

    proprio_dim = train_dataset.proprio.shape[1]
    policy = BCPolicy(
        proprio_dim=proprio_dim,
        chunk_size=args.chunk_size,
        hidden_dim=args.hidden_dim,
    ).to(device)

    resnet_params = [p for n, p in policy.named_parameters() if 'resnet' in n and p.requires_grad]
    other_params = [p for n, p in policy.named_parameters() if 'resnet' not in n]
    optimizer = torch.optim.AdamW([
        {'params': other_params, 'lr': args.lr},
        {'params': resnet_params, 'lr': args.lr * 0.1},
    ], weight_decay=1e-4)

    action_loss_fn = nn.L1Loss()
    done_loss_fn = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(args.epochs):
        lr = cosine_lr(epoch, 5, args.epochs, args.lr, 1e-5)
        for pg in optimizer.param_groups:
            pg['lr'] = lr if 'resnet' not in str(pg.get('params', '')) else lr * 0.1
        optimizer.param_groups[0]['lr'] = lr
        optimizer.param_groups[1]['lr'] = lr * 0.1

        policy.train()
        train_loss = 0
        for clip_embed, images, proprio, lang, actions, dones in train_loader:
            clip_embed, images, proprio, lang, actions, dones = (
                clip_embed.to(device), images.to(device), proprio.to(device),
                lang.to(device), actions.to(device), dones.to(device),
            )
            policy.reset_hidden()
            pred_actions, done_logits = policy(clip_embed, images, proprio, lang, reset_hidden=True)
            loss_action = action_loss_fn(pred_actions, actions)
            loss_done = done_loss_fn(done_logits.squeeze(-1), dones)
            loss = loss_action + args.aux_weight * loss_done
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(actions)
        train_loss /= len(train_dataset)

        policy.eval()
        val_loss = 0
        with torch.no_grad():
            for clip_embed, images, proprio, lang, actions, dones in val_loader:
                clip_embed, images, proprio, lang, actions, dones = (
                    clip_embed.to(device), images.to(device), proprio.to(device),
                    lang.to(device), actions.to(device), dones.to(device),
                )
                policy.reset_hidden()
                pred_actions, done_logits = policy(clip_embed, images, proprio, lang, reset_hidden=True)
                loss_action = action_loss_fn(pred_actions, actions)
                loss_done = done_loss_fn(done_logits.squeeze(-1), dones)
                val_loss += (loss_action + args.aux_weight * loss_done).item() * len(actions)
        val_loss /= len(val_dataset)

        print(f"  Epoch {epoch+1}/{args.epochs}  lr={lr:.2e}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "policy_state_dict": policy.state_dict(),
                "action_mean": train_dataset.action_mean,
                "action_std": train_dataset.action_std,
                "proprio_dim": proprio_dim,
                "chunk_size": args.chunk_size,
                "hidden_dim": args.hidden_dim,
            }, save_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch+1} (no improvement for {args.patience} epochs)")
                break

    return best_val_loss


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    print(f"Seed: {args.seed}")

    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

    demo_dir = Path(args.demo_dir)
    all_demo_files = sorted(demo_dir.glob("*_demo.hdf5"))
    held_out = set(args.held_out_tasks)
    train_files = [f for i, f in enumerate(all_demo_files) if i not in held_out]
    test_files = [f for i, f in enumerate(all_demo_files) if i in held_out]

    print(f"Total tasks: {len(all_demo_files)}, train: {len(train_files)}, test: {len(test_files)}")
    for f in test_files:
        print(f"  [test] {f.stem}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    ds_kwargs = dict(chunk_size=args.chunk_size, seq_len=args.seq_len)

    # --- K-fold cross-validation ---
    if not args.skip_folds:
        num_folds = args.num_folds
        np.random.seed(42)
        indices = np.random.permutation(len(train_files))
        folds = np.array_split(indices, num_folds)

        fold_results = []
        for fold_i in range(num_folds):
            val_indices = set(folds[fold_i].tolist())
            fold_train_files = [train_files[j] for j in range(len(train_files)) if j not in val_indices]
            fold_val_files = [train_files[j] for j in folds[fold_i]]

            print(f"\n{'='*60}")
            print(f"Fold {fold_i+1}/{num_folds}")
            for f in fold_val_files:
                print(f"    [val] {f.stem}")

            print("  Loading train set (with augmentation)...")
            fold_train_ds = DemoDataset(fold_train_files, clip_model, clip_preprocess, **ds_kwargs, augment=True)
            print(f"  {len(fold_train_ds)} train transitions")

            print("  Loading val set...")
            fold_val_ds = DemoDataset(fold_val_files, clip_model, clip_preprocess, **ds_kwargs, augment=False)
            print(f"  {len(fold_val_ds)} val transitions")

            fold_path = save_dir / f"bc_fold{fold_i}.pt"
            best_val = train_one(fold_train_ds, fold_val_ds, args, fold_path)
            fold_results.append(best_val)
            print(f"  Fold {fold_i+1} best val loss: {best_val:.6f}")

        print(f"\n{'='*60}")
        print(f"Cross-validation results:")
        for i, v in enumerate(fold_results):
            print(f"  Fold {i+1}: {v:.6f}")
        print(f"  Mean: {np.mean(fold_results):.6f}  Std: {np.std(fold_results):.6f}")

    # --- Final model ---
    print(f"\n{'='*60}")
    print(f"Training final model on all {len(train_files)} training tasks...")
    final_ds = DemoDataset(train_files, clip_model, clip_preprocess, **ds_kwargs, augment=True)
    print(f"Loaded {len(final_ds)} transitions")

    val_size = int(len(final_ds) * 0.1)
    train_size = len(final_ds) - val_size
    train_indices, val_indices = random_split(range(len(final_ds)), [train_size, val_size])
    train_split = DemoSubset(final_ds, train_indices.indices)
    val_split = DemoSubset(final_ds, val_indices.indices)

    final_path = save_dir / f"bc_best_seed{args.seed}.pt"
    best_final = train_one(train_split, val_split, args, final_path)
    print(f"Final model best val loss: {best_final:.6f}")
    print(f"Saved final checkpoint to {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo_dir", type=str, required=True)
    parser.add_argument("--held_out_tasks", type=int, nargs="*", default=[])
    parser.add_argument("--num_folds", type=int, default=4)
    parser.add_argument("--skip_folds", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=10)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--aux_weight", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=25)
    args = parser.parse_args()
    train(args)
