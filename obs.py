from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
import h5py
import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
resnet.fc = torch.nn.Identity()
resnet = resnet.to(device).eval()

preprocess = T.Compose([
    T.ToTensor(),
    T.Resize((224, 224)),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def encode(image):
    x = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        return resnet(x).squeeze(0).cpu().numpy()

benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict["libero_object"]()
task = task_suite.get_task(0)

env = OffScreenRenderEnv(**{
    "bddl_file_name": task_suite.get_task_bddl_file_path(0),
    "camera_heights": 256,
    "camera_widths": 256,
})

obs = env.reset()
frames = []
actions_list = []
ee_pos_list = []
ee_quat_list = []
joint_pos_list = []
gripper_list = []
agentview_list = []
eye_in_hand_list = []
agentview_embed_list = []
eye_in_hand_embed_list = []
rewards_list = []
dones_list = []

for i in range(50):
    action = env.env.action_spec[1] * (2 * np.random.rand(7) - 1)
    obs, reward, done, info = env.step(action)

    actions_list.append(action)
    ee_pos_list.append(obs["robot0_eef_pos"])
    ee_quat_list.append(obs["robot0_eef_quat"])
    joint_pos_list.append(obs["robot0_joint_pos"])
    gripper_list.append(obs["robot0_gripper_qpos"])
    agentview_list.append(obs["agentview_image"])
    eye_in_hand_list.append(obs["robot0_eye_in_hand_image"])
    agentview_embed_list.append(encode(obs["agentview_image"]))
    eye_in_hand_embed_list.append(encode(obs["robot0_eye_in_hand_image"]))
    rewards_list.append(reward)
    dones_list.append(done)

    frames.append(Image.fromarray(obs["agentview_image"]))
    if done:
        break

out_path = "/workspace/rollout_data.hdf5"
with h5py.File(out_path, "w") as f:
    grp = f.create_group("data")
    grp.attrs["task"] = task.language
    grp.attrs["num_steps"] = len(actions_list)
    grp.attrs["encoder"] = "resnet18-imagenet"
    grp.attrs["embed_dim"] = 512

    grp.create_dataset("actions", data=np.array(actions_list))
    grp.create_dataset("rewards", data=np.array(rewards_list))
    grp.create_dataset("dones", data=np.array(dones_list))

    obs_grp = grp.create_group("obs")
    obs_grp.create_dataset("ee_pos", data=np.array(ee_pos_list))
    obs_grp.create_dataset("ee_quat", data=np.array(ee_quat_list))
    obs_grp.create_dataset("joint_pos", data=np.array(joint_pos_list))
    obs_grp.create_dataset("gripper_qpos", data=np.array(gripper_list))
    obs_grp.create_dataset("agentview_rgb", data=np.array(agentview_list))
    obs_grp.create_dataset("eye_in_hand_rgb", data=np.array(eye_in_hand_list))
print(f"saved rollout data to {out_path}")

embed_dir = "/workspace/LIBERO/project"
np.save(f"{embed_dir}/agentview_embed.npy", np.array(agentview_embed_list))
np.save(f"{embed_dir}/eye_in_hand_embed.npy", np.array(eye_in_hand_embed_list))
print(f"saved embeddings to {embed_dir}/agentview_embed.npy and eye_in_hand_embed.npy")

frames[0].save(
    "/workspace/rollout.gif",
    save_all=True,
    append_images=frames[1:],
    duration=100,
    loop=0,
)
print("saved gif to /workspace/rollout.gif")
env.close()