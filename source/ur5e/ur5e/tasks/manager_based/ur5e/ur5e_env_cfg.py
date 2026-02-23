# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp

##
# Pre-defined configs
##

from ur5e.tasks.manager_based.assets.ur5e_configs import UR5E_CFG  # isort:skip


##
# Scene definition
##


@configclass
class Ur5eSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # table (cube)
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(1.5, 1.2, 0.8),  # width, depth, height
            collision_props=sim_utils.CollisionPropertiesCfg(), 
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.3, 0.3, 0.4)),  
    )

    assemble_part_1 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/AssemblePart_1",
        spawn=sim_utils.UsdFileCfg( 
            usd_path=__file__.replace("ur5e_env_cfg.py", "../usd/assemble_part/1.usd"), 
            scale=(0.001, 0.001, 0.001), 
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.3, 0.4, 0.9)),
    )

    assemble_part_2 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/AssemblePart_2",
        spawn=sim_utils.UsdFileCfg( 
            usd_path=__file__.replace("ur5e_env_cfg.py", "../usd/assemble_part/2.usd"), 
            scale=(0.001, 0.001, 0.001),  
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.6, 0.45, 0.9)), 
    )

    # robot
    robot: ArticulationCfg = UR5E_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot", 
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.8))  # 👉 注意这里前缀的改变
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=[".*"], scale=10.0)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""
    # 回合重置时，将机器人关节恢复到默认位置，并加上一点微小的随机偏移
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "position_range": (-0.1, 0.1),
            "velocity_range": (-0.05, 0.05),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    # (1) 存活奖励 (让环境能跑起来的基础奖励)
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # (2) 动作惩罚 (防止机械臂乱挥)
    action_penalty = RewTerm(func=mdp.action_l2, weight=-0.01)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    # 超时终止回合
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


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

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation