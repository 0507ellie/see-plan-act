from __future__ import annotations

import os
from pathlib import Path

import h5py
import imageio.v2 as imageio
from PIL import Image, ImageDraw


DEFAULT_DATASET_DIR = Path(r"D:\LIBERO\datasets\libero_object")
DEFAULT_DEMO_NAME = "pick_up_the_ketchup_and_place_it_in_the_basket_demo.hdf5"


def get_demo_file() -> Path:
    dataset_dir = Path(os.environ.get("LIBERO_OBJECT_DATASET_DIR", DEFAULT_DATASET_DIR))
    demo_name = os.environ.get("LIBERO_OBJECT_DEMO_NAME", DEFAULT_DEMO_NAME)
    return dataset_dir / demo_name


def get_output_dir() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return Path(os.environ.get("LIBERO_PREVIEW_OUT", repo_root / "outputs" / "libero_preview"))


def label_frame(frame, text: str) -> Image.Image:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 18), fill=(0, 0, 0))
    draw.text((4, 3), text, fill=(255, 255, 255))
    return image


def main() -> None:
    demo_file = get_demo_file()
    out_dir = get_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(demo_file, "r") as f:
        demo = f["data"]["demo_0"]
        agent = demo["obs"]["agentview_rgb"][:]
        wrist = demo["obs"]["eye_in_hand_rgb"][:]
        actions = demo["actions"][:]

    first_agent = out_dir / "ketchup_agentview_first.png"
    first_wrist = out_dir / "ketchup_eye_in_hand_first.png"
    video_path = out_dir / "ketchup_demo_side_by_side.mp4"

    Image.fromarray(agent[0]).save(first_agent)
    Image.fromarray(wrist[0]).save(first_wrist)

    frames = []
    for t, (agent_frame, wrist_frame) in enumerate(zip(agent, wrist)):
        left = label_frame(agent_frame, f"agentview t={t}")
        right = label_frame(wrist_frame, f"eye_in_hand t={t}")
        canvas = Image.new("RGB", (left.width + right.width, left.height))
        canvas.paste(left, (0, 0))
        canvas.paste(right, (left.width, 0))
        frames.append(canvas)

    imageio.mimsave(video_path, frames, fps=20)

    print("demo:", demo_file)
    print("timesteps:", len(frames))
    print("action shape:", actions.shape)
    print("saved first agentview:", first_agent)
    print("saved first eye_in_hand:", first_wrist)
    print("saved video:", video_path)


if __name__ == "__main__":
    main()
