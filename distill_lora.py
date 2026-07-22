"""Distillation stage: pool all per-task SAC specialists' harvested rollouts and
fine-tune the frozen BC base's existing LoRA adapters on them. Folds what every
specialist learned back into one deployable checkpoint, saved in the same format
train_bc.py uses so it drops straight into eval.py / benchmark_eval.py.

Only the existing resnet_adapter/clip_adapter/lang_adapter LoRA modules get
gradients; everything else in BCPolicy stays frozen.

Usage:
    python distill_lora.py --bc_checkpoint checkpoints/bc_best_seed42.pt \
        --shard_dir rollout_shards --save_path checkpoints/bc_distilled.pt
"""
import argparse
import glob
from pathlib import Path

import clip
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from train_bc import BCPolicy, build_lang_embed, cosine_lr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def reopen_adapters(base: BCPolicy):
    """Freeze everything except the resnet_adapter/clip_adapter/lang_adapter params."""
    reopened = []
    for name, p in base.named_parameters():
        p.requires_grad = name.startswith(("resnet_adapter.", "clip_adapter.", "lang_adapter."))
        if p.requires_grad:
            reopened.append(name)
    return reopened


class RolloutDataset(Dataset):
    """Like train_bc.py's DemoDataset, but sourced from harvest_rollouts.py shards.
    Yields cached resnet_out/clip_embed (already pre-adapter) instead of raw images,
    so the frozen backbones never get re-run during distillation. Only successful
    episodes are kept by default (only_success=True)."""

    def __init__(self, shard_paths, clip_model, chunk_size=8, seq_len=10, only_success=True):
        self.chunk_size = chunk_size
        self.seq_len = seq_len

        self.ep_resnet, self.ep_clip, self.ep_proprio, self.ep_action, self.ep_done = [], [], [], [], []
        self.ep_lang = []
        lang_cache = {}
        n_total, n_kept = 0, 0

        for shard_path in shard_paths:
            shard = torch.load(shard_path, map_location="cpu", weights_only=False)
            task_id = shard["task_id"]
            if task_id not in lang_cache:
                lang_cache[task_id] = build_lang_embed(clip_model, shard["language"], device).squeeze(0).cpu().numpy()
            for ep in shard["episodes"]:
                n_total += 1
                if only_success and not ep["success"]:
                    continue
                n_kept += 1
                T_ep = len(ep["done"])

                act_list, don_list = [], []
                for i in range(T_ep):
                    chunk = ep["action"][i:i + chunk_size]
                    if len(chunk) < chunk_size:
                        chunk = np.concatenate([chunk, np.tile(chunk[-1:], (chunk_size - len(chunk), 1))])
                    act_list.append(chunk)
                    don_list.append(float(ep["done"][i:i + chunk_size].max()))

                self.ep_resnet.append(ep["resnet_out"])
                self.ep_clip.append(ep["clip_embed"])
                self.ep_proprio.append(ep["proprio"])
                self.ep_action.append(np.array(act_list, dtype=np.float32))
                self.ep_done.append(np.array(don_list, dtype=np.float32))
                self.ep_lang.append(lang_cache[task_id])

        print(f"RolloutDataset: kept {n_kept}/{n_total} episodes"
              f"{' (successful only)' if only_success else ''} from {len(shard_paths)} shard(s)")

        self.proprio = np.concatenate(self.ep_proprio) if self.ep_proprio else np.zeros((0, 9))

        # Same windowing convention as DemoDataset: every start_t (including negative,
        # front-padded) start position within each episode.
        self.windows = []
        for ep_idx in range(len(self.ep_resnet)):
            T_ep = len(self.ep_resnet[ep_idx])
            for t in range(-(self.seq_len - 1), T_ep):
                self.windows.append((ep_idx, t))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        ep_idx, start_t = self.windows[idx]
        T_ep = len(self.ep_resnet[ep_idx])
        real_start = max(start_t, 0)
        real_end = min(start_t + self.seq_len, T_ep)
        front_pad = real_start - start_t
        back_pad = (start_t + self.seq_len) - real_end

        w_resnet = self.ep_resnet[ep_idx][real_start:real_end]
        w_clip = self.ep_clip[ep_idx][real_start:real_end]
        w_pro = self.ep_proprio[ep_idx][real_start:real_end]
        w_act = self.ep_action[ep_idx][real_start:real_end]
        w_don = self.ep_done[ep_idx][real_start:real_end]
        lng = self.ep_lang[ep_idx]

        if front_pad > 0:
            w_resnet = np.pad(w_resnet, ((front_pad, 0), (0, 0), (0, 0)), mode="edge")
            w_clip = np.pad(w_clip, ((front_pad, 0), (0, 0)), mode="edge")
            w_pro = np.pad(w_pro, ((front_pad, 0), (0, 0)), mode="edge")
            w_act = np.pad(w_act, ((front_pad, 0), (0, 0), (0, 0)), mode="edge")
            w_don = np.pad(w_don, (front_pad, 0), mode="edge")
        if back_pad > 0:
            w_resnet = np.pad(w_resnet, ((0, back_pad), (0, 0), (0, 0)), mode="edge")
            w_clip = np.pad(w_clip, ((0, back_pad), (0, 0)), mode="edge")
            w_pro = np.pad(w_pro, ((0, back_pad), (0, 0)), mode="edge")
            w_act = np.pad(w_act, ((0, back_pad), (0, 0), (0, 0)), mode="edge")
            w_don = np.pad(w_don, (0, back_pad), mode="edge")

        return (
            torch.from_numpy(w_resnet.copy()),
            torch.from_numpy(w_clip.copy()),
            torch.from_numpy(w_pro.copy()),
            torch.from_numpy(lng.copy()),
            torch.from_numpy(w_act.copy()),
            torch.from_numpy(w_don.copy()),
        )


def distill(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    clip_model, _ = clip.load("ViT-B/32", device=device)

    bc_ckpt = torch.load(args.bc_checkpoint, map_location=device, weights_only=False)
    ckpt_lora_rank = bc_ckpt.get("lora_rank", 32)
    policy = BCPolicy(
        proprio_dim=bc_ckpt["proprio_dim"],
        chunk_size=bc_ckpt.get("chunk_size", 8),
        hidden_dim=bc_ckpt.get("hidden_dim", 256),
        lora_rank=args.lora_rank,
    ).to(device)

    if args.lora_rank == ckpt_lora_rank:
        policy.load_state_dict(bc_ckpt["policy_state_dict"])
    else:
        # Different rank means the checkpoint's adapter tensors won't fit. Load
        # everything else as-is and leave fresh zero-init adapters at the new rank.
        filtered = {k: v for k, v in bc_ckpt["policy_state_dict"].items()
                    if not k.startswith(("resnet_adapter.", "clip_adapter.", "lang_adapter."))}
        missing, unexpected = policy.load_state_dict(filtered, strict=False)
        assert not unexpected, f"unexpected keys: {unexpected}"
        assert all(k.startswith(("resnet_adapter.", "clip_adapter.", "lang_adapter.")) for k in missing), \
            f"unexpected missing keys: {missing}"
        print(f"lora_rank changed {ckpt_lora_rank} -> {args.lora_rank}: "
              f"re-initialized {len(missing)} adapter tensors fresh (zero-init up-projection)")

    for p in policy.parameters():
        p.requires_grad = False
    reopened = reopen_adapters(policy)
    print(f"Reopened {len(reopened)} parameter tensors for distillation: "
          f"{sorted(set(n.split('.')[0] for n in reopened))}")

    shard_paths = sorted(glob.glob(str(Path(args.shard_dir) / "*_shard.pt")))
    assert shard_paths, f"no *_shard.pt files found in {args.shard_dir}"
    print(f"Found {len(shard_paths)} shard file(s): {[Path(p).name for p in shard_paths]}")

    full_dataset = RolloutDataset(
        shard_paths, clip_model, chunk_size=bc_ckpt.get("chunk_size", 8), seq_len=args.seq_len,
    )
    n_val = max(1, int(0.1 * len(full_dataset)))
    n_train = len(full_dataset) - n_val
    train_dataset, val_dataset = random_split(
        full_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed),
    )
    print(f"Train/val windows: {n_train}/{n_val}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4,
    )
    action_loss_fn = nn.L1Loss()
    done_loss_fn = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    patience_counter = 0
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        lr = cosine_lr(epoch, 5, args.epochs, args.lr, 1e-5)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        policy.train()
        train_loss = 0
        for resnet_out, clip_embed, proprio, lang, actions, dones in train_loader:
            resnet_out, clip_embed, proprio = resnet_out.to(device), clip_embed.to(device), proprio.to(device)
            lang, actions, dones = lang.to(device), actions.to(device), dones.to(device)
            pred_actions, done_logits = policy.forward_from_cached_features(resnet_out, clip_embed, proprio, lang)
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
            for resnet_out, clip_embed, proprio, lang, actions, dones in val_loader:
                resnet_out, clip_embed, proprio = resnet_out.to(device), clip_embed.to(device), proprio.to(device)
                lang, actions, dones = lang.to(device), actions.to(device), dones.to(device)
                pred_actions, done_logits = policy.forward_from_cached_features(resnet_out, clip_embed, proprio, lang)
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
                "action_mean": bc_ckpt["action_mean"],
                "action_std": bc_ckpt["action_std"],
                "proprio_dim": bc_ckpt["proprio_dim"],
                "chunk_size": bc_ckpt.get("chunk_size", 8),
                "hidden_dim": bc_ckpt.get("hidden_dim", 256),
                "clip_dim": bc_ckpt.get("clip_dim", 512),
                "lang_dim": bc_ckpt.get("lang_dim", 512),
                "action_dim": bc_ckpt.get("action_dim", 7),
                "num_layers": bc_ckpt.get("num_layers", 4),
                "num_heads": bc_ckpt.get("num_heads", 4),
                "dropout": bc_ckpt.get("dropout", 0.1),
                "rnn_hidden": bc_ckpt.get("rnn_hidden", 512),
                "rnn_layers": bc_ckpt.get("rnn_layers", 2),
                "lora_rank": args.lora_rank,
                "distilled_from": args.bc_checkpoint,
                "distilled_shards": shard_paths,
            }, save_path)
            print(f"  saved {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch+1} (no improvement for {args.patience} epochs)")
                break

    return best_val_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bc_checkpoint", type=str, required=True)
    parser.add_argument("--shard_dir", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--seq_len", type=int, default=10, help="must match BC training seq_len")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--aux_weight", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lora_rank", type=int, default=32,
                         help="if different from the bc_checkpoint's own lora_rank, adapters "
                              "are rebuilt fresh at this rank (zero-init up-projection) instead "
                              "of loading the checkpoint's adapter weights")
    args = parser.parse_args()
    distill(args)
