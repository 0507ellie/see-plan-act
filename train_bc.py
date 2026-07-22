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
from robosuite.utils.transform_utils import quat2axisangle

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

        # ResNet18, fully frozen; adapted via LoRA (same scheme as CLIP below)
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        resnet.fc = nn.Identity()
        for param in resnet.parameters():
            param.requires_grad = False
        self.resnet = resnet
        resnet_dim = 512
        self.resnet_adapter = LoRAAdapter(resnet_dim, rank=lora_rank)

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

    def train(self, mode=True):
        super().train(mode)
        self.resnet.eval()  # frozen backbone: keep BatchNorm stats fixed
        return self

    def forward(self, clip_embed, images, proprio, lang, return_features=False):
        # clip_embed: (B, T, 1024), images: (B, T, 2, 3, 224, 224)
        # proprio: (B, T, proprio_dim), lang: (B, 512)
        B, T = clip_embed.shape[0], clip_embed.shape[1]

        # ResNet on raw images (frozen; only the LoRA adapter is trainable)
        img_flat = images.reshape(B * T * 2, *images.shape[-3:])
        with torch.no_grad():
            resnet_out = self.resnet(img_flat)
        resnet_out = resnet_out.reshape(B, T, 2, -1)

        return self._forward_from_features(resnet_out, clip_embed, proprio, lang, return_features)

    def forward_from_cached_features(self, resnet_out, clip_embed, proprio, lang, return_features=False):
        """Alternate entry point for when resnet_out/clip_embed (the frozen backbones'
        pre-adapter outputs) are already available, e.g. cached by harvest_rollouts.py
        and reused across distillation epochs, so the frozen backbones never need to
        be re-run. Produces identical output to forward() for the same observation.
        resnet_out: (B, T, 2, resnet_dim), clip_embed: (B, T, 1024)."""
        return self._forward_from_features(resnet_out, clip_embed, proprio, lang, return_features)

    def _forward_from_features(self, resnet_out, clip_embed, proprio, lang, return_features):
        # resnet_out: (B, T, 2, resnet_dim), already computed. Shared by forward() and
        # forward_from_cached_features() so both stay consistent.
        B, T = clip_embed.shape[0], clip_embed.shape[1]

        resnet_out = self.resnet_adapter(resnet_out.reshape(B * T, 2, -1))
        resnet_tokens = self.resnet_proj(resnet_out)

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

        lstm_out, _ = self.lstm(lstm_in)

        context = self.context_proj(lstm_out).reshape(B * T, 1, -1)

        memory = torch.cat([
            context, vis_proj, pro_proj.unsqueeze(1), lang_proj.unsqueeze(1),
        ], dim=1)
        memory = memory + self.memory_pos_embed

        queries = self.action_queries.expand(B * T, -1, -1)
        decoded = self.decoder(queries, memory)

        actions = self.action_head(decoded).reshape(B, T, self.chunk_size, -1)
        done_logits = self.done_head(decoded[:, -1, :]).reshape(B, T, 1)
        if return_features:
            # decoded: (B*T, chunk_size, hidden_dim), pre-action-head tokens,
            # exposed so a downstream module (e.g. a residual policy head) can
            # condition on the same fused multimodal representation without
            # redoing the CLIP/ResNet/LSTM/decoder work.
            return actions, done_logits, decoded
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

        all_actions_flat = np.concatenate([a.reshape(-1, a.shape[-1]) for a in demo_actions])
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

        # Windows include start_t < 0 (front-padded by repeating frame 0), matching
        # rollout_episode's sliding window at the start of an episode, otherwise the
        # model never trains on the all-frames-identical input it sees at rollout step 0.
        self.windows = []
        for d_idx in range(len(demo_clip)):
            T_demo = len(demo_clip[d_idx])
            for t in range(-(self.seq_len - 1), T_demo):
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
        real_start = max(start_t, 0)
        real_end = min(start_t + self.seq_len, T_demo)
        front_pad = real_start - start_t
        back_pad = (start_t + self.seq_len) - real_end

        w_clip = self.demo_clip[d_idx][real_start:real_end]
        w_ag = self.demo_agentview[d_idx][real_start:real_end]
        w_eih = self.demo_eye_in_hand[d_idx][real_start:real_end]
        w_pro = self.demo_proprio[d_idx][real_start:real_end]
        w_act = self.demo_actions[d_idx][real_start:real_end]
        w_don = self.demo_dones[d_idx][real_start:real_end]
        lng = self.demo_lang[d_idx]

        # Preprocess images for ResNet (with augmentation)
        img_tensors = []
        for t in range(real_end - real_start):
            img_tensors.append(self._preprocess_image(w_ag[t]))
            img_tensors.append(self._preprocess_image(w_eih[t]))

        if front_pad > 0:
            w_clip = np.pad(w_clip, ((front_pad, 0), (0, 0)), mode='edge')
            w_pro = np.pad(w_pro, ((front_pad, 0), (0, 0)), mode='edge')
            w_act = np.pad(w_act, ((front_pad, 0), (0, 0), (0, 0)), mode='edge')
            w_don = np.pad(w_don, (front_pad, 0), mode='edge')
            first_ag, first_eih = img_tensors[0], img_tensors[1]
            img_tensors = [first_ag, first_eih] * front_pad + img_tensors

        if back_pad > 0:
            w_clip = np.pad(w_clip, ((0, back_pad), (0, 0)), mode='edge')
            w_pro = np.pad(w_pro, ((0, back_pad), (0, 0)), mode='edge')
            w_act = np.pad(w_act, ((0, back_pad), (0, 0), (0, 0)), mode='edge')
            w_don = np.pad(w_don, (0, back_pad), mode='edge')
            last_ag, last_eih = img_tensors[-2], img_tensors[-1]
            img_tensors = img_tensors + [last_ag, last_eih] * back_pad

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


def encode_clip_image(clip_model, clip_preprocess, image, device):
    x = clip_preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
    with torch.no_grad():
        return clip_model.encode_image(x).squeeze(0).float()


def build_lang_embed(clip_model, text, device):
    tokens = clip.tokenize([text]).to(device)
    with torch.no_grad():
        return clip_model.encode_text(tokens).float()


@torch.no_grad()
def rollout_episode(policy, env, clip_model, clip_preprocess, lang_embed,
                     action_mean, action_std, device, max_steps=300, seq_len=10,
                     temporal_ensemble=True, ensemble_decay=0.1, collect_frames=False,
                     init_state=None):
    """Runs one closed-loop episode with a fixed seq_len sliding window plus optional
    ACT-style temporal ensembling over predicted action chunks. Returns (success,
    frames or None).

    If init_state is given (a flattened sim state), the episode starts from that exact
    state via env.set_init_state instead of a randomized env.reset(), so different
    checkpoints/runs can be compared on identical initial conditions, matching LIBERO's
    own eval protocol.

    Always calls reset() first even when init_state is given: set_init_state() only
    teleports sim state, it doesn't reset robosuite's internal timestep/horizon
    bookkeeping, so reusing one env across many fixed-state trials without a reset in
    between would eventually exceed the env's horizon and robosuite would refuse
    further steps."""
    if init_state is not None:
        env.reset()
        obs = env.set_init_state(init_state)
        for _ in range(5):
            obs, _, _, _ = env.step(np.zeros(7))
    else:
        obs = env.reset()
    frames = [] if collect_frames else None
    done = False
    clip_hist, img_hist, proprio_hist = [], [], []
    chunk_history = {}
    for step in range(max_steps):
        ag_clip = encode_clip_image(clip_model, clip_preprocess, obs["agentview_image"], device)
        eih_clip = encode_clip_image(clip_model, clip_preprocess, obs["robot0_eye_in_hand_image"], device)
        clip_embed = torch.cat([ag_clip, eih_clip])

        ag_img = RESNET_PREPROCESS(Image.fromarray(obs["agentview_image"]))
        eih_img = RESNET_PREPROCESS(Image.fromarray(obs["robot0_eye_in_hand_image"]))
        images = torch.stack([ag_img, eih_img]).to(device)

        gripper_open = np.array([1.0 if obs["robot0_gripper_qpos"][0] > 0.03 else 0.0])
        proprio = np.concatenate([
            obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"], gripper_open,
        ])
        proprio = torch.tensor(proprio, dtype=torch.float32).to(device)

        clip_hist.append(clip_embed)
        img_hist.append(images)
        proprio_hist.append(proprio)
        clip_hist = clip_hist[-seq_len:]
        img_hist = img_hist[-seq_len:]
        proprio_hist = proprio_hist[-seq_len:]
        pad = seq_len - len(clip_hist)

        clip_seq = torch.stack([clip_hist[0]] * pad + clip_hist).unsqueeze(0)
        img_seq = torch.stack([img_hist[0]] * pad + img_hist).unsqueeze(0)
        proprio_seq = torch.stack([proprio_hist[0]] * pad + proprio_hist).unsqueeze(0)

        action_chunk, _ = policy(clip_seq, img_seq, proprio_seq, lang_embed)
        chunk_norm = action_chunk[0, -1]
        chunk_denorm = (chunk_norm * action_std + action_mean).cpu().numpy()

        if temporal_ensemble:
            chunk_history[step] = chunk_denorm
            oldest = max(0, step - policy.chunk_size + 1)
            for s in list(chunk_history):
                if s < oldest:
                    del chunk_history[s]
            candidates, weights = [], []
            for s, chunk in chunk_history.items():
                offset = step - s
                if offset < len(chunk):
                    candidates.append(chunk[offset])
                    weights.append(np.exp(-ensemble_decay * offset))
            weights = np.array(weights) / np.sum(weights)
            action = np.sum(np.stack(candidates) * weights[:, None], axis=0)
        else:
            action = chunk_denorm[0]

        obs, reward, done, info = env.step(action)
        if collect_frames:
            frames.append(Image.fromarray(obs["agentview_image"][::-1]))
        if done:
            return True, frames
    return False, frames


def quick_eval(policy, env, clip_model, clip_preprocess, lang_embed, action_mean, action_std,
               device, num_episodes=3, max_steps=150, seq_len=10, gif_dir=None, epoch=None):
    was_training = policy.training
    policy.eval()
    successes = 0
    for ep in range(num_episodes):
        success, frames = rollout_episode(
            policy, env, clip_model, clip_preprocess, lang_embed,
            action_mean, action_std, device, max_steps=max_steps, seq_len=seq_len,
            collect_frames=gif_dir is not None,
        )
        successes += int(success)
        if gif_dir is not None and frames:
            epoch_dir = Path(gif_dir) / f"epoch_{epoch:04d}"
            epoch_dir.mkdir(parents=True, exist_ok=True)
            result = "success" if success else "fail"
            gif_path = epoch_dir / f"ep{ep}_{result}.gif"
            frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=100, loop=0)
    if was_training:
        policy.train()
    return successes / num_episodes


def cosine_lr(epoch, warmup_epochs, total_epochs, base_lr, min_lr):
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def train_one(train_dataset, val_dataset, args, save_path, eval_ctx=None):
    loader_kwargs = dict(num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
                          pin_memory=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, **loader_kwargs)

    proprio_dim = train_dataset.proprio.shape[1]
    policy = BCPolicy(
        proprio_dim=proprio_dim,
        chunk_size=args.chunk_size,
        hidden_dim=args.hidden_dim,
    ).to(device)

    # ResNet backbone is fully frozen (LoRA-adapted like CLIP), so every
    # trainable param shares one LR group, no more backbone/head split.
    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-4,
    )

    action_loss_fn = nn.L1Loss()
    done_loss_fn = nn.BCEWithLogitsLoss()

    action_mean_t = torch.tensor(train_dataset.action_mean, dtype=torch.float32).to(device)
    action_std_t = torch.tensor(train_dataset.action_std, dtype=torch.float32).to(device)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(args.epochs):
        lr = cosine_lr(epoch, 5, args.epochs, args.lr, 1e-5)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        policy.train()
        train_loss = 0
        for clip_embed, images, proprio, lang, actions, dones in train_loader:
            clip_embed, images, proprio, lang, actions, dones = (
                clip_embed.to(device), images.to(device), proprio.to(device),
                lang.to(device), actions.to(device), dones.to(device),
            )
            pred_actions, done_logits = policy(clip_embed, images, proprio, lang)
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
                pred_actions, done_logits = policy(clip_embed, images, proprio, lang)
                loss_action = action_loss_fn(pred_actions, actions)
                loss_done = done_loss_fn(done_logits.squeeze(-1), dones)
                val_loss += (loss_action + args.aux_weight * loss_done).item() * len(actions)
        val_loss /= len(val_dataset)

        print(f"  Epoch {epoch+1}/{args.epochs}  lr={lr:.2e}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

        if eval_ctx is not None and args.eval_every > 0 and (epoch + 1) % args.eval_every == 0:
            success_rate = quick_eval(
                policy, eval_ctx["env"], eval_ctx["clip_model"], eval_ctx["clip_preprocess"],
                eval_ctx["lang_embed"], action_mean_t, action_std_t, device,
                num_episodes=args.eval_episodes, max_steps=args.eval_max_steps, seq_len=args.seq_len,
                gif_dir=args.eval_gif_dir or None, epoch=epoch + 1,
            )
            print(f"  [rollout eval] epoch {epoch+1}: success {success_rate:.1%} ({args.eval_episodes} episodes, task {args.eval_task_id})")

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
                # Not currently CLI-configurable, train_one() always constructs BCPolicy with
                # these left at class defaults, but saving them explicitly means a downstream
                # loader (eval.py, train_ppo.py, train_sac.py) never has to silently assume that
                # stays true. If any of these ever become CLI args, update this dict to match.
                "clip_dim": policy.clip_dim,
                "lang_dim": policy.lang_proj.in_features,
                "action_dim": policy.action_head.out_features,
                "num_layers": 4,
                "num_heads": 4,
                "dropout": 0.1,
                "rnn_hidden": policy.lstm.hidden_size,
                "rnn_layers": policy.lstm.num_layers,
                "lora_rank": policy.resnet_adapter.down.out_features,
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

    eval_ctx = None
    if args.eval_every > 0:
        from libero.libero import benchmark as libero_benchmark
        from libero.libero.envs import OffScreenRenderEnv

        eval_task_suite = libero_benchmark.get_benchmark_dict()[args.eval_suite]()
        eval_task = eval_task_suite.get_task(args.eval_task_id)
        eval_env = OffScreenRenderEnv(
            bddl_file_name=eval_task_suite.get_task_bddl_file_path(args.eval_task_id),
            camera_heights=128, camera_widths=128,
            hard_reset=False,
        )
        eval_ctx = {
            "env": eval_env,
            "clip_model": clip_model,
            "clip_preprocess": clip_preprocess,
            "lang_embed": build_lang_embed(clip_model, eval_task.language, device),
        }
        print(f"Rollout eval every {args.eval_every} epochs on task {args.eval_task_id}: {eval_task.language!r}")

    demo_dir = Path(args.demo_dir)
    train_files = sorted(demo_dir.glob("*_demo.hdf5"))
    print(f"Total tasks: {len(train_files)}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    ds_kwargs = dict(chunk_size=args.chunk_size, seq_len=args.seq_len)

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
    best_final = train_one(train_split, val_split, args, final_path, eval_ctx=eval_ctx)
    print(f"Final model best val loss: {best_final:.6f}")
    print(f"Saved final checkpoint to {final_path}")

    if eval_ctx is not None:
        eval_ctx["env"].close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo_dir", type=str, required=True)
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
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader worker processes for image preprocessing")
    parser.add_argument("--eval_every", type=int, default=3, help="rollout-eval every N epochs, 0 disables")
    parser.add_argument("--eval_episodes", type=int, default=3)
    parser.add_argument("--eval_max_steps", type=int, default=150)
    parser.add_argument("--eval_suite", type=str, default="libero_object")
    parser.add_argument("--eval_task_id", type=int, default=0)
    parser.add_argument("--eval_gif_dir", type=str, default="eval_gifs",
                         help="dir to save rollout-eval GIFs during training, pass '' to disable")
    args = parser.parse_args()
    train(args)
