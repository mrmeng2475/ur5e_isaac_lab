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
#  补充导入坐标系传感器类型，用于类型提示
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
    lift_diff = part2_z_w - (rest_height + 0.0001)
    
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

def conditional_grasp_normalized_reward(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg, 
    dist_threshold: float, 
    max_angles: list[float],
    joint_weights: list[float]  #  新增参数：各个关节的独立权重
) -> torch.Tensor:
    """
    当机械臂末端靠近零件时，鼓励灵巧手的指定关节向正方向（闭合）运动。
    加入归一化处理，并允许为不同关节分配不同的逼近权重。
    """
    ee_tf = env.scene["ee_frame"]
    parts_tf = env.scene["parts_frame"]
    ee_pos_w = ee_tf.data.target_pos_w[:, 0, :]
    part2_pos_w = parts_tf.data.target_pos_w[:, 1, :]
    
    # 计算末端与零件抓取点的距离
    dist = torch.norm(ee_pos_w - part2_pos_w, dim=-1)
    
    # 判断是否进入抓取阈值
    is_close = (dist < dist_threshold).float()
    
    # 获取指定的 8 个手指关节角度
    robot = env.scene[asset_cfg.name]
    joint_indices, _ = robot.find_joints(asset_cfg.joint_names)
    finger_pos = robot.data.joint_pos[:, joint_indices]
    
    # 将传入的列表转换为 PyTorch 张量
    max_angles_tensor = torch.tensor(max_angles, device=robot.device, dtype=torch.float32)
    weights_tensor = torch.tensor(joint_weights, device=robot.device, dtype=torch.float32) #  转换权重列表
    
    # 核心：归一化 (当前角度 / 最大极限角度)
    normalized_finger_pos = torch.clamp(finger_pos / max_angles_tensor, min=0.0, max=1.0)
    
    #  将归一化后的值乘以对应关节的权重，再相加
    weighted_finger_pos = normalized_finger_pos * weights_tensor
    sum_flexion = torch.sum(weighted_finger_pos, dim=-1)
    
    # 距离小于阈值时才发放分数
    return sum_flexion * is_close

def object_is_lifted(env: ManagerBasedRLEnv, rest_height: float, threshold: float, dist_threshold: float = 0.02) -> torch.Tensor:
    ee_tf: FrameTransformer = env.scene["ee_frame"]
    parts_tf: FrameTransformer = env.scene["parts_frame"]
    
    # 提取坐标
    ee_pos_w = ee_tf.data.target_pos_w[:, 0, :]
    part2_pos_w = parts_tf.data.target_pos_w[:, 1, :]
    part2_z_w = part2_pos_w[:, 2]

    # 判断高度是否达标
    is_lifted = (part2_z_w > (rest_height + threshold)).float()
    
    # 判断是否在抓取范围内
    dist = torch.norm(ee_pos_w - part2_pos_w, dim=-1)
    is_grasped = (dist < dist_threshold).float()

    # 必须同时满足“抓紧”和“离地”才能拿分
    return is_lifted * is_grasped

def part2_assemble_pos_reward(
    env: ManagerBasedRLEnv, 
    std: float, 
    dist_threshold: float = 0.02,
    min_height: float = 0.94,         # 基础离地高度
    xy_active_height: float = 0.95     #  新增：激活 XY 奖励的高度
) -> torch.Tensor:
    """
    分阶段位置装配奖励：
    1. Z <= 0.931 : 没举起来，0分。
    2. 0.931 < Z <= 0.94 : 只奖励 Z 轴方向靠近 Part1 (引导继续往上举)，不考虑 XY 偏差。
    3. Z > 0.94 : 完全激活，奖励 3D 空间逼近 (Z 轴与 XY 轴同时靠近)。
    """
    ee_tf: FrameTransformer = env.scene["ee_frame"]
    parts_tf: FrameTransformer = env.scene["parts_frame"]
    
    # 提取坐标
    ee_pos_w = ee_tf.data.target_pos_w[:, 0, :]
    part1_pos_w = parts_tf.data.target_pos_w[:, 0, :]
    part2_pos_w = parts_tf.data.target_pos_w[:, 1, :]
    
    # ==========================================
    # 1. 状态掩码检测
    # ==========================================
    # 抓取掩码
    grasp_dist = torch.norm(ee_pos_w - part2_pos_w, dim=-1)
    is_grasped = (grasp_dist < dist_threshold).float()
    
    # 阶段高度掩码
    part2_z = part2_pos_w[:, 2]
    is_lifted_base = (part2_z > min_height).float()         # 是否离地
    is_lifted_high = (part2_z > xy_active_height).float()   # 是否达到平移高度
    
    # ==========================================
    # 2. 轴向距离拆解与奖励计算
    # ==========================================
    # 1. 计算 Z 轴高度差奖励
    z_dist = torch.abs(part2_pos_w[:, 2] - part1_pos_w[:, 2])
    z_reward = torch.exp(-torch.square(z_dist) / std)
    
    # 2. 计算 XY 水平面距离奖励
    xy_dist = torch.norm(part2_pos_w[:, :2] - part1_pos_w[:, :2], dim=-1)
    xy_reward = torch.exp(-torch.square(xy_dist) / std)
    
    # ==========================================
    # 3. 组合逻辑 (核心)
    # ==========================================
    # 如果达到 0.94，乘以 xy_reward；
    # 如果没达到 0.94，相当于不对 XY 进行惩罚（将其视为 1.0 满分），只按 Z 轴给分。
    # 这样在 0.931 ~ 0.94 阶段，AI 只要往上举就能持续得分，不会因为 XY 没对齐而失分。
    staged_xy_reward = torch.where(is_lifted_high > 0.5, xy_reward, torch.ones_like(xy_reward))
    
    pos_reward = z_reward * staged_xy_reward
    
    # 必须抓紧并且离地才发分数
    return pos_reward * is_grasped * is_lifted_base


def part2_assemble_rot_reward(
    env: ManagerBasedRLEnv, 
    std: float, 
    dist_threshold: float = 0.02,
    rot_active_height: float = 0.95  #  使用 0.94 作为姿态奖励的激活阈值
) -> torch.Tensor:
    """
    条件装配姿态奖励：
    只有当机械臂将零件举起超过 0.94 时，才开始奖励 Z 轴朝向对齐。
    """
    ee_tf: FrameTransformer = env.scene["ee_frame"]
    parts_tf: FrameTransformer = env.scene["parts_frame"]
    
    # 提取坐标与四元数
    ee_pos_w = ee_tf.data.target_pos_w[:, 0, :]
    part2_pos_w = parts_tf.data.target_pos_w[:, 1, :]
    
    part1_quat_w = parts_tf.data.target_quat_w[:, 0, :]
    part2_quat_w = parts_tf.data.target_quat_w[:, 1, :]
    
    # ==========================================
    # 1. 抓取与举起状态约束检测
    # ==========================================
    grasp_dist = torch.norm(ee_pos_w - part2_pos_w, dim=-1)
    is_grasped = (grasp_dist < dist_threshold).float()
    
    #  高度约束直接改为 0.94 (rot_active_height)
    part2_z = part2_pos_w[:, 2]
    is_lifted_high = (part2_z > rot_active_height).float()
    
    # ==========================================
    # 2. Z轴重合奖励
    # ==========================================
    def get_z_axis(quat):
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        zx = 2.0 * (x * z + w * y)
        zy = 2.0 * (y * z - w * x)
        zz = 1.0 - 2.0 * (x * x + y * y)
        return torch.stack([zx, zy, zz], dim=-1)
        
    part1_z_axis = get_z_axis(part1_quat_w)
    part2_z_axis = get_z_axis(part2_quat_w)
    
    z_dot = torch.sum(part1_z_axis * part2_z_axis, dim=-1)
    rot_reward = torch.exp(-(1.0 - z_dot) / std)
    
    #  只有举高到 0.94 且抓稳时，才发放姿态对齐分
    return rot_reward * is_grasped * is_lifted_high