# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.markers.config import FRAME_MARKER_CFG
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer import OffsetCfg 
from isaaclab.markers.config import FRAME_MARKER_CFG     
from isaaclab.sim import SimulationCfg


from . import mdp
from .ur5e_scene import Ur5eSceneCfg

##
# Pre-defined configs
##

from ur5e.tasks.manager_based.assets.ur5e_configs import UR5E_CFG, UR5E_ACTION_SCALES


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    
    # 1. 机械臂前 6 个轴
    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot", 
        joint_names=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"], 
        scale={
            "shoulder.*": 3.14,  
            "elbow_joint": 3.14, 
            "wrist_1_joint": 3.14,
            "wrist_2_joint": 3.14,
            "wrist_3_joint": 3.14,
        },  
        use_default_offset=True,
        # 👇 修复：改为字典格式，".*" 匹配上述 6 个关节
        # clip={".*": (-1.0, 1.0)} 
    )
    
    # 2. 灵巧手
    hand_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["l_f_joint.*"],  
        scale=1.5,  
        use_default_offset=True,
        # 👇 修复：改为字典格式，".*" 匹配灵巧手的关节
        # clip={".*": (-1.0, 1.0)}
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # 1. 机器人本体状态 (Proprioception)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        # 2. 夹爪 TCP (末端执行器) 的绝对状态
        ee_pos = ObsTerm(func=mdp.ee_pos)
        ee_quat = ObsTerm(func=mdp.ee_quat)

        # 3. 目标 Part 2 的绝对状态
        part2_pos = ObsTerm(func=mdp.part2_pos)
        part2_quat = ObsTerm(func=mdp.part2_quat)

        # 4. [强烈推荐] 相对位置偏差 (让网络知道目标在哪里)
        ee_to_part2_vec = ObsTerm(func=mdp.ee_to_part2_vec)

        def __post_init__(self) -> None:
            self.enable_corruption = False # 如果需要加噪声，后续可以设为 True
            self.concatenate_terms = True  # 将所有特征拼接成一个一维向量输入给神经网络

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""
    
    # 1. 回合重置时，将机器人关节恢复到默认位置
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "position_range": (-0.0, 0.0),
            "velocity_range": (-0.0, 0.0),
        },
    )

    # 👉 2. 重置 Part 1 的位置和状态
    reset_part_1 = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("assemble_part_1"),
            # 这里设为 0 表示严格回到 ur5e_scene.py 里定义的 init_state
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}, 
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), 
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)
            },
        },
    )

    # 👉 3. 重置 Part 2 的位置和状态 (我们主要抓它，加一点随机化)
    reset_part_2 = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("assemble_part_2"),
            # 让每次重置时，Part 2 在 XY 平面上随机偏移 ±2cm，偏航角随机转动一定角度
            # 这能大幅提高策略在真机部署时的鲁棒性！
            "pose_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "yaw": (-0.0, 0.0)}, 
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), 
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)
            },
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    
    # 👉 1. 独立的位置靠近奖励
    reach_part_2_pos = RewTerm(
        func=mdp.ee_to_part2_pos_reward,
        weight=10.0,   # 你可以独立调整这个权重
        params={"std": 0.2}
    )
    # 👉 1厘米高精度区域奖励
    fine_reach_part_2_pos = RewTerm(
        func=mdp.ee_to_part2_fine_pos_reward,  # 调用刚才写的阶跃函数
        weight=50.0,   # 权重给高一点！只要待在这 1cm 的球体里，每步都拿 50 分
        params={"threshold": 0.015}  # 0.01 米 = 1 厘米
    )

    # 👉 2. 独立的姿态对齐奖励
    reach_part_2_rot = RewTerm(
        func=mdp.ee_to_part2_rot_reward,
        weight=3.0,   # 你可以独立调整这个权重
        params={"std": 0.5}  # 这里的 std 控制角度宽容度
    )

    continuous_lift = RewTerm(
        func=mdp.part2_continuous_lift_with_grasp_reward, # 👉 使用带抓取约束的新函数
        weight=1000.0,  
        params={
            "rest_height": 0.93,       # 根据之前的计算，静止时的绝对高度
            "dist_threshold": 0.025     # 👉 必须保持在 1 厘米内才算抓紧
        }
    )

    grasp_when_close = RewTerm(
        func=mdp.conditional_grasp_normalized_reward,  # 👉 使用新的归一化函数
        weight=50.0,  
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[
                "l_f_joint1_2", 
                "l_f_joint1_3", 
                "l_f_joint1_4", 

                "l_f_joint2_2", 
                "l_f_joint2_3", 

                "l_f_joint3_2", 
                "l_f_joint3_3", 
            ]),
            "dist_threshold": 0.01,  # 只要进入 1 厘米的核心区，就开始闭合
            # 👉 新增：严格对应上面 7 个关节名称的最大闭合角度
            "max_angles": [
                1.38, 0.45, 1.26,  # 食指 (1)
                1.40, 1.20,        # 中指 (2)
                1.40, 1.20         # 无名指 (3)
            ] 
        }
    )

    # finger_touch_part2 = RewTerm(
    #     func=mdp.finger_part2_contact_reward,  # 调用我们刚写的函数
    #     weight=20.0,  # 只要碰到了，每一步都给 20 分！鼓励它一直摸着
    #     params={
    #         "sensor_cfg": SceneEntityCfg("index_finger_contact"), # 绑定步骤 1 的传感器
    #         "force_threshold": 0.1  # 力度大于 0.1 牛顿就算有效接触，过滤掉偶尔的计算噪声
    #     }
    # )
    thumb_open_wide = RewTerm(
        func=mdp.maximize_negative_joint_pos, # 调用我们刚写的函数
        weight=2.0,  # 权重设置：建议先从 1.0 或 2.0 开始试
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["l_f_joint_1"]),
            # 设定阈值：例如进入目标点 1 厘米 (0.01米) 范围内时，瞬间切换为夹紧模式
            "dist_threshold": 0.01
        }
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    # 1. 超时终止 (5秒)
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    # 2. 零件倒下终止
    part_2_fall = DoneTerm(
        func=mdp.part_toppled,
        params={
            "asset_cfg": SceneEntityCfg("assemble_part_2"),
            "threshold": 0.1736  # cos(80度) ≈ 0.1736，倾斜超过 80 度就算倒了 
        }
    )
    
    # 👉 3. 真实的撞击桌面终止 (物理接触力检测)
    bad_collision = DoneTerm(
        func=mdp.illegal_contact, # 直接使用官方自带的碰撞检测函数
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces"), # 绑定刚才在 Scene 里加的传感器
            "threshold": 1.0  # 受力阈值：超过 1.0 牛顿就视为撞击并终止
        }
    )
    # velocity_limit = DoneTerm(
    #     func=mdp.joint_velocity_limit,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]), # 监控机器人的所有关节
    #         # UR5e 的官方标称最大速度：前三个关节 3.14 rad/s，手腕关节 6.28 rad/s。
    #         # 这里设置 3.0 rad/s (约 170度/秒) 作为安全红线。
    #         "max_velocity": 9.0 
    #     }
    # )

# 👉 3. 机器人自碰撞终止
    # self_collision = DoneTerm(
    #     func=mdp.illegal_contact, 
    #     params={
    #         # 绑定刚才在 SceneCfg 里加的自碰撞专属传感器
    #         "sensor_cfg": SceneEntityCfg("robot_self_contact"), 
    #         # 同样设置受力阈值：内部挤压力超过 1.0 牛顿就视为自碰撞并终止
    #         "threshold": 1.0  
    #     }
    # )

##
# Environment configuration
##


@configclass
class Ur5eEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: Ur5eSceneCfg = Ur5eSceneCfg(num_envs=4096, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=20.0,       # 极高的静摩擦
            dynamic_friction=15.5,      
            restitution=0.0,           # 毫无弹性
            friction_combine_mode="max", 
            restitution_combine_mode="min"
        )
    )

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5.0
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation