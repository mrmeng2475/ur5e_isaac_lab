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


def get_sequence_action(step_counts, phases, dt, device):
    """
    根据当前步数和定义的阶段序列，计算当前应执行的动作。
    
    参数:
        step_counts: (num_envs,) 每个环境当前的运行步数
        phases: 列表，每个元素为 ([关节角列表], 持续时间_秒)
        dt: 环境的时间步长 (env.unwrapped.step_dt)
        device: 计算设备 (cuda/cpu)
        
    返回:
        current_actions: (num_envs, action_dim) 发送给环境的动作 Tensor
    """
    num_envs = step_counts.shape[0]
    action_dim = len(phases[0][0])
    
    # 预计算每个阶段的截止步数
    phase_end_steps = []
    cumulative_steps = 0
    for _, duration in phases:
        cumulative_steps += int(duration / dt)
        phase_end_steps.append(cumulative_steps)

    # 默认执行最后一个阶段的动作（当超过总时长时）
    last_action = torch.tensor(phases[-1][0], device=device).repeat(num_envs, 1)
    current_actions = last_action

    # 从倒数第二个阶段开始向前判断，使用 torch.where 嵌套逻辑（像洋葱一样包裹）
    # 只要 step_counts 小于该阶段的截止步数，就覆盖为该阶段的动作
    for i in range(len(phases) - 2, -1, -1):
        phase_action = torch.tensor(phases[i][0], device=device).repeat(num_envs, 1)
        mask = (step_counts < phase_end_steps[i]).unsqueeze(1)
        current_actions = torch.where(mask, phase_action, current_actions)

    return current_actions

# =========================================================================
# 3. 主程序
# =========================================================================

def main():
    # 创建环境配置
    env_cfg = parse_env_cfg(
        args_cli.task, 
        device=args_cli.device, 
        num_envs=args_cli.num_envs, 
        use_fabric=not args_cli.disable_fabric
    )
    # 创建环境
    env = gym.make(args_cli.task, cfg=env_cfg)
    
    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs
    dt = env.unwrapped.step_dt # 获取环境的步长

    print(f"[INFO]: 成功创建环境，环境数量: {num_envs}")
    
    # --- 定义动作序列 ---
    # 格式: ([关节角], 持续时间秒)
    # 假设前 6 位是 UR5e 机械臂，后 6 位是灵巧手
    task_phases = [
        ([0.0, 0.0, 0.0, -0.87, 0.0, 0.6,   0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1.0),   # 1. 预备
        ([-0.01, 0.26, 0.26, -0.87, 0.0, 0.6, -0.44, 0.0, 0.0, 0.0, 0.0, 0.0], 2.0), # 2. 接近
        ([-0.01, 0.26, 0.26, -0.87, 0.0, 0.6, 0.0, 0.5, 1.2, 1.4, 1.4, 1.2], 1.0),   # 3. 抓紧
        ([0.0, 0.1, 0.0, -0.82, -0.2, 0.5,   0.44, 1.38, 1.2, 1.4, 1.4, 1.2], 3.0),    # 4. 抬起
        ([0.2, 0.1, 0.0, -0.82, -0.2, 0.5,   0.44, 1.38, 1.2, 1.4, 1.4, 1.2], 4.0),    # 5. 保持
        ([0.2, 0.15, 0.15, -0.6, -0.2, 0.4,   0.44, 1.38, 1.2, 1.4, 1.4, 1.2], 4.0),    # 5. 保持
        ([0.2, 0.15, 0.15, -0.6, -0.2, 0.4,   -0.44, 0.0, 0.0, 0.0, 0.0, 0.0], 4.0),    # 5. 保持
    ]

    # 初始化计时器
    step_counts = torch.zeros(num_envs, dtype=torch.long, device=device)
    env.reset()

    while simulation_app.is_running():
        with torch.inference_mode():
            # 1. 调用新函数获取当前应执行的动作
            actions = get_sequence_action(step_counts, task_phases, dt, device)
            
            # 2. 物理步进
            obs, rewards, terminated, truncated, info = env.step(actions)
            
            # 3. 统计步数
            step_counts += 1
            
            # 4. 环境重生检测 (如果某个环境结束了，重置该环境的计时器)
            dones = terminated | truncated
            step_counts[dones] = 0

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
