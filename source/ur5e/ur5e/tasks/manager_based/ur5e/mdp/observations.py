import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sensors import FrameTransformer

# ==============================================================================
# 自定义观测函数 (Observations)
# ==============================================================================

def ee_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """获取夹爪 TCP 的三维空间位置 (x, y, z)"""
    ee_tf: FrameTransformer = env.scene["ee_frame"]
    return ee_tf.data.target_pos_w[:, 0, :]

def ee_quat(env: ManagerBasedRLEnv) -> torch.Tensor:
    """获取夹爪 TCP 的四元数姿态 (w, x, y, z)"""
    ee_tf: FrameTransformer = env.scene["ee_frame"]
    return ee_tf.data.target_quat_w[:, 0, :]

def part2_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """获取 Part 2 抓取点的三维空间位置 (x, y, z)"""
    parts_tf: FrameTransformer = env.scene["parts_frame"]
    return parts_tf.data.target_pos_w[:, 1, :]

def part2_quat(env: ManagerBasedRLEnv) -> torch.Tensor:
    """获取 Part 2 抓取点的四元数姿态 (w, x, y, z)"""
    parts_tf: FrameTransformer = env.scene["parts_frame"]
    return parts_tf.data.target_quat_w[:, 1, :]

def ee_to_part2_vec(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    [黄金特征] 获取从夹爪 TCP 指向 Part 2 抓取点的相对向量 (dx, dy, dz)。
    这能极大地加速神经网络的收敛！
    """
    ee_tf: FrameTransformer = env.scene["ee_frame"]
    parts_tf: FrameTransformer = env.scene["parts_frame"]
    
    ee_pos_w = ee_tf.data.target_pos_w[:, 0, :]
    part2_pos_w = parts_tf.data.target_pos_w[:, 1, :]
    
    return part2_pos_w - ee_pos_w