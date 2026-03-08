# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

import argparse
from isaaclab.app import AppLauncher

# 1. 配置命令行参数
parser = argparse.ArgumentParser(description="IK Debug agent with Position and Orientation control.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--task", type=str, default="Template-Ur5e-v0", help="Name of the task.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 2. 启动仿真器
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""其余导入在 App 启动后进行"""
import gymnasium as gym
import torch
import numpy as np

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import (
    quat_from_euler_xyz, 
    quat_mul, 
    quat_inv, 
    axis_angle_from_quat,
    quat_rotate_inverse
)

# 导入自定义任务包
import ur5e.tasks  # noqa: F401

def main():
    # 创建环境配置
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # 创建环境
    env = gym.make(args_cli.task, cfg=env_cfg)
    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs

    print(f"[INFO]: Gym action space: {env.action_space.shape}")

    # 重置环境
    env.reset()

    # --- 目标设置 ---
    
    # 1. 位置目标 (相对于每个机器人的底座)
    local_target_pos = torch.tensor([[0.625, 0.325, 0.93]], device=device).repeat(num_envs, 1)
    env_origins = env.unwrapped.scene.env_origins
    target_pos_w = local_target_pos + env_origins

    # 2. 姿态目标 (绕自身 X 轴旋转 -30 度)
    # 将角度转为弧度: -30 * pi / 180
    roll_rad = -0.0 * np.pi / 180.0
    # 生成局部目标四元数 (w, x, y, z)
    target_quat_local = quat_from_euler_xyz(
        torch.tensor([roll_rad], device=device), 
        torch.tensor([0.0], device=device), 
        torch.tensor([0.0], device=device)
    )
    # 假设目标是相对于初始朝向 (Identity)，如果需要相对于世界坐标，可在此调整
    target_quat_w = target_quat_local.repeat(num_envs, 1)

    # --- 控制器增益 ---
    p_gain_pos = 3.0
    p_gain_ori = 2.0

    # 预分配 action 张量
    actions = torch.zeros((num_envs, env.action_space.shape[1]), device=device)

    step_count = 0

    # 模拟主循环
    while simulation_app.is_running():
        with torch.inference_mode():
            # 获取当前末端状态
            # 注意：某些配置中 ee_frame 路径可能不同，请根据你的场景结构确认
            ee_state = env.unwrapped.scene["ee_frame"].data
            ee_pos_w = ee_state.target_pos_w[:, 0, :]
            ee_quat_w = ee_state.target_quat_w[:, 0, :]

            # --- [A] 位置控制 ---
            # 计算世界系下的位移偏差
            pos_error_w = target_pos_w - ee_pos_w
            actions[:, 0:3] = pos_error_w * p_gain_pos

            # --- [B] 姿态控制 ---
            # 计算旋转偏差: q_error = q_target * inv(q_current)
            # 这代表了从当前朝向转到目标朝向所需的旋转
            q_error = quat_mul(target_quat_w, quat_inv(ee_quat_w))
            
            # 将四元数误差转换为轴角矢量 (axis-angle error)
            # 结果是一个 [num_envs, 3] 的张量，代表旋转轴 * 旋转弧度
            ori_error_w = axis_angle_from_quat(q_error)
            actions[:, 3:6] = ori_error_w * p_gain_ori

            # --- [C] 安全处理与执行 ---
            # 限制最大速度，防止机器人瞬间起飞
            actions[:, 0:3] = torch.clamp(actions[:, 0:3], min=-1.5, max=1.5)
            actions[:, 3:6] = torch.clamp(actions[:, 3:6], min=-1.0, max=1.0)

            # 打印调试信息
            if step_count % 50 == 0:
                print(f"Step: {step_count}")
                print(f"  Pos Error Norm: {torch.norm(pos_error_w[0]).item():.4f} m")
                print(f"  Ori Error Norm: {torch.norm(ori_error_w[0]).item():.4f} rad")

            # 发送动作
            env.step(actions)
            step_count += 1

if __name__ == "__main__":
    main()
    simulation_app.close()