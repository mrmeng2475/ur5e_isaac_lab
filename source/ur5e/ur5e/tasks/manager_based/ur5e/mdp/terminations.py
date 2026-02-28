import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation

def part_toppled(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, threshold: float = 0.1736) -> torch.Tensor:
    """
    如果零件倒下了（即其上方向量与世界坐标系 Z 轴偏离太大），则终止。
    threshold: 阈值，通常取 Z 轴分量小于 0.5 (约倒下 60 度)
    """
    # 获取零件对象
    part: RigidObject = env.scene[asset_cfg.name]
    
    # 获取零件在世界坐标系下的上方向量（通常是局部坐标系的 Z 轴 [0,0,1]）
    # 我们通过四元数变换 [0,0,1] 向量得到它在世界系下的朝向
    # 这里为了简单，直接看旋转四元数变换后的 Z 轴投影
    part_quat_w = part.data.root_quat_w
    
    # 计算物体局部 Z 轴在世界坐标系 Z 轴上的投影 (取自旋转矩阵的 R[2,2])
    # 公式：z_axis_w = 1 - 2 * (x^2 + y^2)
    q_x = part_quat_w[:, 1]
    q_y = part_quat_w[:, 2]
    up_projection = 1.0 - 2.0 * (q_x**2 + q_y**2)
    
    # 如果上方向量在世界 Z 轴上的投影小于阈值，说明倒了
    return up_projection < threshold

def robot_table_contact(env: ManagerBasedRLEnv, threshold: float = 1.0) -> torch.Tensor:
    """
    如果机械臂末端 TCP 撞击桌面 (高度低于阈值)，则终止。
    """
    ee_tf = env.scene["ee_frame"]
    # 获取末端 Z 轴高度
    ee_z = ee_tf.data.target_pos_w[:, 0, 2]
    
    return ee_z < threshold

def joint_velocity_limit(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, max_velocity: float) -> torch.Tensor:
    """
    如果机器人任何一个关节的角速度绝对值超过了最大允许速度，则终止回合。
    """
    # 提取机器人本体对象
    asset: Articulation = env.scene[asset_cfg.name]
    
    # 获取当前所有被监控关节的角速度，维度通常为 [num_envs, num_joints]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    
    # 取绝对值，并判断是否超过阈值
    # torch.any(..., dim=1) 表示只要这台机器人的哪怕一个关节超速，就返回 True
    return torch.any(torch.abs(joint_vel) > max_velocity, dim=1)