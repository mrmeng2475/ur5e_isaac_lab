# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to an environment with random action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import ur5e.tasks  # noqa: F401


def main():
    """Sequential target action agent with Isaac Lab environment."""
    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    env.reset()

    # =========================================================================
    # 👉 核心修改：四阶段动作控制 (预备 -> 接近 -> 抓紧 -> 抬起)
    # =========================================================================
    num_envs = env.unwrapped.num_envs
    device = env.unwrapped.device

    # 阶段 1：抬起手腕等待 (0 ~ 1秒)
    phase1_actions_list = [
        0.0, 0.0, 0.0, -0.87, 0.0, 0.6,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0 
    ]
    # 阶段 2：机械臂移动到目标点，灵巧手张开 (1 ~ 3秒)
    phase2_actions_list = [
        -0.01, 0.26, 0.26, -0.87, 0.0, 0.6,
        -0.44, 0.0, 0.0, 0.0, 0.0, 0.0
    ]
    # 阶段 3：机械臂保持不动，灵巧手闭合抓取 (3 ~ 4秒)
    phase3_actions_list = [
        -0.01, 0.26, 0.26, -0.87, 0.0, 0.6,   
        0.0, 0.5, 1.2, 1.4, 1.4, 1.2     
    ]
    # 阶段 4：灵巧手保持抓紧，机械臂抬起回到安全位置 (4秒之后)
    phase4_actions_list = [
        0.0, 0.0, 0.0, -0.82, 0.0, 0.5,   # 👈 机械臂执行新的抬起动作
        0.44, 1.38, 1.2, 1.4, 1.4, 1.2      # 👈 灵巧手保持阶段 3 的抓紧状态！
    ]
    
    # 转换为 Tensor 并扩充到所有并行环境
    phase1_actions = torch.tensor(phase1_actions_list, device=device, dtype=torch.float32).repeat(num_envs, 1)
    phase2_actions = torch.tensor(phase2_actions_list, device=device, dtype=torch.float32).repeat(num_envs, 1)
    phase3_actions = torch.tensor(phase3_actions_list, device=device, dtype=torch.float32).repeat(num_envs, 1)
    phase4_actions = torch.tensor(phase4_actions_list, device=device, dtype=torch.float32).repeat(num_envs, 1) # 新增

    # ⌚ 独立计时器
    step_counts = torch.zeros(num_envs, dtype=torch.long, device=device)
    
    # 时间节点配置 (你可以根据需要微调这些秒数)
    wait_steps_1 = int(1.0 / env.unwrapped.step_dt) # 第 1 阶段：0~1秒
    wait_steps_2 = int(3.0 / env.unwrapped.step_dt) # 第 2 阶段：1~3秒
    wait_steps_3 = int(5.0 / env.unwrapped.step_dt) # 第 3 阶段：3~4秒 (给灵巧手1秒钟的时间抓牢)

    # =========================================================================

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            
            # 👇 1. 动态生成 Mask
            mask_phase1 = (step_counts < wait_steps_1).unsqueeze(1)
            mask_phase2 = (step_counts < wait_steps_2).unsqueeze(1)
            mask_phase3 = (step_counts < wait_steps_3).unsqueeze(1) # 新增
            
            # 👇 2. GPU 高效并行分发动作 (嵌套逻辑，像洋葱一样一层层剥开)
            active_actions = torch.where(
                mask_phase1, 
                phase1_actions, 
                torch.where(
                    mask_phase2, 
                    phase2_actions, 
                    torch.where(
                        mask_phase3, 
                        phase3_actions, 
                        phase4_actions # 如果都大于 4 秒，就执行抬起！
                    )
                )
            )
                
            # 👇 3. 物理步进
            obs, rewards, terminated, truncated, info = env.step(active_actions)
            
            # 👇 4. 步数全面 +1
            step_counts += 1
            
            # 👇 5. 环境重生检测
            dones = terminated | truncated
            step_counts[dones] = 0

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
