import argparse
import importlib
import numpy as np
import torch
import torch.nn as nn
import clip
import torchvision.transforms as T
from PIL import Image
from pathlib import Path
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from robosuite.utils.transform_utils import quat2axisangle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESNET_PREPROCESS = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

POLICY_REGISTRY = {
    "bc": ("train_bc", "BCPolicy"),
}


def load_policy(policy_type, checkpoint_path):
    module_name, class_name = POLICY_REGISTRY[policy_type]
    module = importlib.import_module(module_name)
    policy_class = getattr(module, class_name)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    policy = policy_class(
        proprio_dim=ckpt["proprio_dim"],
        chunk_size=ckpt.get("chunk_size", 8),
        hidden_dim=ckpt.get("hidden_dim", 256),
    ).to(device)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()
    return policy, ckpt


def evaluate(args):
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

    def encode_clip(image):
        x = clip_preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
        with torch.no_grad():
            return clip_model.encode_image(x).squeeze(0).float()

    policy, ckpt = load_policy(args.policy, args.checkpoint)
    action_mean = torch.tensor(ckpt["action_mean"], dtype=torch.float32).to(device)
    action_std = torch.tensor(ckpt["action_std"], dtype=torch.float32).to(device)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.suite]()
    task_id = args.task_id
    task = task_suite.get_task(task_id)

    lang_tokens = clip.tokenize([task.language]).to(device)
    with torch.no_grad():
        lang_embed = clip_model.encode_text(lang_tokens).float()

    env = OffScreenRenderEnv(**{
        "bddl_file_name": task_suite.get_task_bddl_file_path(task_id),
        "camera_heights": 256,
        "camera_widths": 256,
    })

    gif_dir = Path(args.gif_dir)
    gif_dir.mkdir(parents=True, exist_ok=True)

    successes = 0
    for ep in range(args.num_episodes):
        obs = env.reset()
        frames = []
        done = False
        policy.reset_hidden()
        for step in range(args.max_steps):
            # CLIP features (frozen)
            ag_clip = encode_clip(obs["agentview_image"])
            eih_clip = encode_clip(obs["robot0_eye_in_hand_image"])
            clip_embed = torch.cat([ag_clip, eih_clip]).reshape(1, 1, -1)

            # Raw images for ResNet (inside model)
            ag_img = RESNET_PREPROCESS(Image.fromarray(obs["agentview_image"]))
            eih_img = RESNET_PREPROCESS(Image.fromarray(obs["robot0_eye_in_hand_image"]))
            images = torch.stack([ag_img, eih_img]).reshape(1, 1, 2, 3, 224, 224).to(device)

            gripper_open = np.array([1.0 if obs["robot0_gripper_qpos"][0] > 0.03 else 0.0])
            proprio = np.concatenate([
                obs["robot0_eef_pos"],
                quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
                gripper_open,
            ])
            proprio = torch.tensor(proprio, dtype=torch.float32).reshape(1, 1, -1).to(device)

            with torch.no_grad():
                action_chunk, _ = policy(clip_embed, images, proprio, lang_embed, reset_hidden=False)
            action_norm = action_chunk[0, 0, 0]
            action = (action_norm * action_std + action_mean).cpu().numpy()

            obs, reward, done, info = env.step(action)
            frames.append(Image.fromarray(obs["agentview_image"][::-1]))
            if done:
                successes += 1
                break

        result = "success" if done else "fail"
        gif_path = gif_dir / f"ep{ep}_{result}.gif"
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=100, loop=0)
        print(f"Episode {ep+1}/{args.num_episodes}: {result.upper()}  -> {gif_path}")

    success_rate = successes / args.num_episodes
    print(f"\nTask: {task.language}")
    print(f"Success rate: {successes}/{args.num_episodes} ({success_rate:.1%})")
    env.close()
    return success_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, required=True, choices=list(POLICY_REGISTRY.keys()))
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--suite", type=str, default="libero_object")
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--num_episodes", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--gif_dir", type=str, default="eval_gifs")
    args = parser.parse_args()
    evaluate(args)
