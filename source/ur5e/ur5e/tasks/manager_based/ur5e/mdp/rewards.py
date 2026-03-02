# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi
# 👉 补充导入坐标系传感器类型，用于类型提示
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv



# ==============================================================================
# 自定义抓取与放置 (Pick and Place) 奖励函数
# ==============================================================================

# def ee_to_part2_pose_reward(env: ManagerBasedRLEnv, pos_std: float, rot_std: float) -> torch.Tensor:
#     """
#     奖励机械臂末端 (TCP) 靠近并对齐目标零件 2。
#     同时计算位置的欧氏距离和姿态的轴角 (Axis-Angle) 偏差。
#     """
#     # 提取传感器数据
#     ee_tf: FrameTransformer = env.scene["ee_frame"]
#     parts_tf: FrameTransformer = env.scene["parts_frame"]

#     # ==========================================
#     # 1. 位置计算 (Position Reward)
#     # ==========================================
#     ee_pos_w = ee_tf.data.target_pos_w[:, 0, :]
#     part2_pos_w = parts_tf.data.target_pos_w[:, 1, :]
    
#     pos_distance = torch.norm(ee_pos_w - part2_pos_w, dim=-1)
#     pos_reward = torch.exp(-torch.square(pos_distance) / pos_std)

#     # ==========================================
#     # 2. 姿态计算 (Orientation / Axis-Angle Reward)
#     # ==========================================
#     # 提取四元数 (w, x, y, z)
#     ee_quat_w = ee_tf.data.target_quat_w[:, 0, :]
#     part2_quat_w = parts_tf.data.target_quat_w[:, 1, :]
    
#     # 计算两个四元数的点积
#     quat_dot = torch.sum(ee_quat_w * part2_quat_w, dim=-1)
    
#     # 取绝对值以确保走最短的旋转路径 (因为 q 和 -q 在物理上代表完全相同的姿态)
#     quat_dot_abs = torch.abs(quat_dot)
    
#     # 截断以防止由于浮点精度问题导致点积略大于 1.0，从而使 acos 产生 NaN
#     quat_dot_clamped = torch.clamp(quat_dot_abs, max=1.0 - 1e-6)
    
#     # 根据点积反推旋转角度差 (也就是 Axis-Angle 中的 Angle)
#     angle_diff = 2.0 * torch.acos(quat_dot_clamped)
    
#     # 将角度差映射为 0~1 的奖励
#     rot_reward = torch.exp(-torch.square(angle_diff) / rot_std)

#     # ==========================================
#     # 3. 综合奖励输出 (Multiplicative Reward)
#     # ==========================================
#     # 使用乘法结合：只有当位置和姿态同时对齐时，才能获得高分。
#     # 这会逼迫 Agent 不能只顾一头。
#     return pos_reward * rot_reward
# ==========================================
# 1. 独立的位置靠近奖励 (Position Reward)
# ==========================================
def ee_to_part2_pos_reward(env: ManagerBasedRLEnv, std: float) -> torch.Tensor:
    """
    仅奖励机械臂末端 (TCP) 靠近目标零件 2 的位置。
    """
    ee_tf: FrameTransformer = env.scene["ee_frame"]
    parts_tf: FrameTransformer = env.scene["parts_frame"]

    ee_pos_w = ee_tf.data.target_pos_w[:, 0, :]
    part2_pos_w = parts_tf.data.target_pos_w[:, 1, :]
    
    pos_distance = torch.norm(ee_pos_w - part2_pos_w, dim=-1)
    
    return torch.exp(-torch.square(pos_distance) / std)


# ==========================================
# 2. 独立的姿态对齐奖励 (Orientation / Axis-Angle Reward)
# ==========================================
def ee_to_part2_rot_reward(env: ManagerBasedRLEnv, std: float) -> torch.Tensor:
    """
    仅奖励机械臂末端 (TCP) 姿态与目标零件 2 对齐 (基于轴角差)。
    """
    ee_tf: FrameTransformer = env.scene["ee_frame"]
    parts_tf: FrameTransformer = env.scene["parts_frame"]

    ee_quat_w = ee_tf.data.target_quat_w[:, 0, :]
    part2_quat_w = parts_tf.data.target_quat_w[:, 1, :]
    
    quat_dot = torch.sum(ee_quat_w * part2_quat_w, dim=-1)
    quat_dot_abs = torch.abs(quat_dot)
    quat_dot_clamped = torch.clamp(quat_dot_abs, max=1.0 - 1e-6)
    
    angle_diff = 2.0 * torch.acos(quat_dot_clamped)
    
    return torch.exp(-torch.square(angle_diff) / std)


def part2_continuous_lift_with_grasp_reward(env: ManagerBasedRLEnv, rest_height: float, dist_threshold: float = 0.01) -> torch.Tensor:
    """
    条件连续性举起奖励：
    只有当机械臂末端与零件的距离小于 dist_threshold 时，零件离开桌面才给予举起奖励。
    """
    ee_tf: FrameTransformer = env.scene["ee_frame"]
    parts_tf: FrameTransformer = env.scene["parts_frame"]
    
    # ==========================================
    # 1. 抓取距离约束检测
    # ==========================================
    ee_pos_w = ee_tf.data.target_pos_w[:, 0, :]
    part2_pos_w = parts_tf.data.target_pos_w[:, 1, :]
    
    # 计算末端与零件抓取点的欧式距离
    dist = torch.norm(ee_pos_w - part2_pos_w, dim=-1)
    
    # 只有距离小于阈值时，is_grasped 才为 1.0，否则为 0.0
    is_grasped = (dist < dist_threshold).float()
    
    # ==========================================
    # 2. 举起高度计算
    # ==========================================
    part2_z_w = parts_tf.data.target_pos_w[:, 1, 2]
    
    # 计算抬起高度（加入 1cm 的防抖动缓冲）
    lift_diff = part2_z_w - (rest_height + 0.001)
    
    # 限制下限为 0，没举起来就不给分
    actual_lift = torch.clamp(lift_diff, min=0.0)
    
    # ==========================================
    # 3. 组合奖励 (掩码相乘)
    # ==========================================
    # 只有当 is_grasped 为 1.0 时，才会把 actual_lift 的分数发出去
    return actual_lift * is_grasped

def finger_part2_contact_reward(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, force_threshold: float) -> torch.Tensor:
    """
    奖励食指与目标零件 2 的物理接触。
    """
    # 1. 从环境中获取刚刚配置的接触传感器
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    
    # 2. 获取传感器检测到的三维受力大小，并计算向量的模长 (也就是受力大小)
    # net_forces_w 的形状是 (num_envs, num_bodies, 3)
    forces = torch.norm(contact_sensor.data.net_forces_w, dim=-1)
    
    # 3. 因为食指可能由多个碰撞体组成，我们取最大受力
    max_force = torch.max(forces, dim=-1)[0]
    
    # 4. 判断受力是否超过了我们设定的有效接触阈值 (返回布尔张量)
    is_contact = max_force > force_threshold
    
    # 5. 将 True/False 转换为 1.0/0.0 作为奖励发给 AI
    return is_contact.float()

def ee_to_part2_fine_pos_reward(env: ManagerBasedRLEnv, threshold: float) -> torch.Tensor:
    """
    当机械臂末端 (TCP) 与目标零件 2 的距离严格小于 threshold 时，给予持续奖励。
    """
    # 获取末端和零件的姿态框架
    ee_tf = env.scene["ee_frame"]
    parts_tf = env.scene["parts_frame"]

    # 提取坐标 (注意：同样是取 0 和 1)
    ee_pos_w = ee_tf.data.target_pos_w[:, 0, :]
    part2_pos_w = parts_tf.data.target_pos_w[:, 1, :]
    
    # 计算欧式距离
    pos_distance = torch.norm(ee_pos_w - part2_pos_w, dim=-1)
    
    # 距离小于 threshold 返回 1.0 (触发奖励)，否则返回 0.0
    return (pos_distance < threshold).float()

def maximize_negative_joint_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, dist_threshold: float) -> torch.Tensor:
    """
    分段大拇指奖励：
    - 距离目标 > dist_threshold 时：鼓励大拇指张开（角度越负越好）。
    - 距离目标 <= dist_threshold 时：鼓励大拇指夹紧（角度越正越好）。
    """
    # 1. 获取末端和零件的位置，计算距离
    ee_tf = env.scene["ee_frame"]
    parts_tf = env.scene["parts_frame"]
    ee_pos_w = ee_tf.data.target_pos_w[:, 0, :]
    part2_pos_w = parts_tf.data.target_pos_w[:, 1, :]
    
    dist = torch.norm(ee_pos_w - part2_pos_w, dim=-1)
    
    # 2. 获取大拇指关节角度
    robot = env.scene[asset_cfg.name]
    joint_indices, _ = robot.find_joints(asset_cfg.joint_names)
    thumb_pos = robot.data.joint_pos[:, joint_indices[0]]
    
    # 3. 动态切换奖励逻辑
    # 当距离大于阈值时，返回 -thumb_pos (张开得分)
    # 当距离小于等于阈值时，返回 thumb_pos (夹紧得分)
    reward = torch.where(dist > dist_threshold, -thumb_pos, thumb_pos)
    
    return reward

def conditional_posture_tracking_reward(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg, 
    target_angles: list[float], 
    dist_threshold: float, 
    active_inside: bool,
    std: float = 1.0
) -> torch.Tensor:
    """
    按距离条件激活的姿态追踪奖励：
    - 当 active_inside=True 时，仅在距离 < dist_threshold 时给予奖励（适合抓取动作）
    - 当 active_inside=False 时，仅在距离 >= dist_threshold 时给予奖励（适合张开/预备动作）
    """
    # 1. 计算末端与零件的距离
    ee_tf = env.scene["ee_frame"]
    parts_tf = env.scene["parts_frame"]
    ee_pos_w = ee_tf.data.target_pos_w[:, 0, :]
    part2_pos_w = parts_tf.data.target_pos_w[:, 1, :]
    
    dist = torch.norm(ee_pos_w - part2_pos_w, dim=-1)
    
    # 2. 根据 active_inside 参数生成判定掩码
    if active_inside:
        mask = (dist < dist_threshold).float()
    else:
        mask = (dist >= dist_threshold).float()
        
    # 3. 获取当前机器人的手指关节角度
    robot = env.scene[asset_cfg.name]
    joint_indices, _ = robot.find_joints(asset_cfg.joint_names)
    finger_pos = robot.data.joint_pos[:, joint_indices]
    
    # 4. 将目标角度转换为 GPU 张量
    target_tensor = torch.tensor(target_angles, device=robot.device, dtype=torch.float32)
    
    # 5. 计算误差与指数奖励
    error = torch.sum(torch.abs(finger_pos - target_tensor), dim=-1)
    reward = torch.exp(-error / std)
    
    # 6. 核心：只有满足距离条件的机器人才会获得该阶段的分数
    return reward * mask

def penalize_body_lin_acc_near_target(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, dist_threshold: float) -> torch.Tensor:
    """
    惩罚特定连杆在靠近目标零件时的线加速度过大（末端防抖）。
    只有当末端执行器距离目标零件小于 dist_threshold 时，才会输出加速度惩罚。
    """
    # ==========================================
    # 1. 距离计算与掩码生成 (复用抓取逻辑)
    # ==========================================
    ee_tf = env.scene["ee_frame"]
    parts_tf = env.scene["parts_frame"]
    
    # 提取世界坐标
    ee_pos_w = ee_tf.data.target_pos_w[:, 0, :]
    part2_pos_w = parts_tf.data.target_pos_w[:, 1, :]
    
    # 计算末端与零件的欧氏距离
    dist = torch.norm(ee_pos_w - part2_pos_w, dim=-1)
    
    # 生成掩码：进入阈值范围记为 1.0，在外面瞎晃悠记为 0.0
    is_close = (dist < dist_threshold).float()
    
    # ==========================================
    # 2. 加速度大小计算
    # ==========================================
    robot = env.scene[asset_cfg.name]
    body_indices = robot.find_bodies(asset_cfg.body_names)[0]
    
    # 获取加速度张量并计算 L2 范数
    acc = robot.data.body_lin_acc_w[:, body_indices[0], :]
    acc_magnitude = torch.norm(acc, dim=-1)
    
    # ==========================================
    # 3. 掩码相乘：只惩罚“核心区”的抖动
    # ==========================================
    # 当 is_close 为 0 时，惩罚值瞬间归零
    return acc_magnitude * is_close