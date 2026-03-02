import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sensors import FrameTransformer
from isaaclab.managers import SceneEntityCfg

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
def link_lin_acc(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    直接获取机器人特定连杆 (Link) 在世界坐标系下的线加速度。
    """
    # 1. 从场景中获取机器人对象
    robot = env.scene[asset_cfg.name]
    
    # 2. 找到你想获取的那个连杆的内部索引
    # find_bodies 会根据正则匹配返回 (body_indices, body_names)
    body_indices = robot.find_bodies(asset_cfg.body_names)[0]
    
    # 3. 直接读取底层物理引擎的加速度张量
    # body_lin_acc_w 的形状是 [num_envs, num_bodies, 3]
    # 我们取出所有环境 (:), 特定的那个连杆 (body_indices[0]), 的 XYZ 坐标 (:)
    acc = robot.data.body_lin_acc_w[:, body_indices[0], :]
    
    return acc