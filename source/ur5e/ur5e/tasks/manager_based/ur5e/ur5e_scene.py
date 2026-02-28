import os
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer import OffsetCfg 
from isaaclab.markers.config import FRAME_MARKER_CFG     
from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg

from ur5e.tasks.manager_based.assets.ur5e_configs import UR5E_CFG 

# 获取当前文件所在目录，方便后续稳健地拼接路径
CURRENT_DIR = os.path.dirname(__file__)

@configclass
class Ur5eSceneCfg(InteractiveSceneCfg):
    """Configuration for a UR5e grasping scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # table (cube)
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(1.5, 1.2, 0.8),
            collision_props=sim_utils.CollisionPropertiesCfg(), 
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.3, 0.3, 0.4)),  
    )

    assemble_part_1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/AssemblePart_1",
        spawn=sim_utils.UsdFileCfg( 
            # 👉 优化了路径拼接方式，再也不怕改文件名了
            usd_path=os.path.join(CURRENT_DIR, "../usd/assemble_part/1.usd"), 
            scale=(0.001, 0.001, 0.001), 
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.1, 0.4, 0.82)),
        debug_vis=False,
    )

    assemble_part_2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/AssemblePart_2",
        spawn=sim_utils.UsdFileCfg( 
            usd_path=os.path.join(CURRENT_DIR, "../usd/assemble_part/2.usd"), 
            scale=(0.001, 0.001, 0.001),  
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.6, 0.3, 0.82)), 
        debug_vis=False,
    )

    # robot
    robot: ArticulationCfg = UR5E_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )

    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link", 
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/.*wrist_3_link", 
                name="end_effector",
                offset=OffsetCfg(pos=(-0.035, 0.24, 0.05),rot=(0.7071, 0.0, -0.7071, 0.0)),
            ),
        ],
        
        debug_vis=True,
        visualizer_cfg=FRAME_MARKER_CFG.replace(
            prim_path="/Visuals/ee_frame",
            markers={
                "frame": sim_utils.UsdFileCfg(
                    usd_path=FRAME_MARKER_CFG.markers["frame"].usd_path,
                    scale=(0.04, 0.04, 0.04),
                )
            }
        ),
    )

    parts_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link", 
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/AssemblePart_1", 
                name="part_1",
                offset=OffsetCfg(
                    pos=(0.075, 0.075, 0.15),  
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/AssemblePart_2", 
                name="part_2",
                offset=OffsetCfg(
                    pos=(0.025, 0.025, 0.13),  
                ),
            ),
        ],
        debug_vis=True,
        visualizer_cfg=FRAME_MARKER_CFG.replace(
            prim_path="/Visuals/parts_frame",
            markers={
                "frame": sim_utils.UsdFileCfg(
                    usd_path=FRAME_MARKER_CFG.markers["frame"].usd_path,
                    scale=(0.02, 0.02, 0.02),
                )
            }
        ),
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*(shoulder|elbow|wrist).*",
        history_length=3,
        track_air_time=False,
        # 👉 核心：只记录和桌子 (Table) 发生的物理碰撞力！
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Table"], 
    )

    robot_self_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",  
        update_period=0.0,
        history_length=3,
        # 👈 核心过滤条件：只记录目标路径也在机器人本体内的接触受力
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/.*"], 
        track_air_time=False,
    )

    index_finger_contact = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/.*l_f_link2_.*", 
    update_period=0.0,
    history_length=3,
    filter_prim_paths_expr=["{ENV_REGEX_NS}/AssemblePart_2"], 
    track_air_time=False,
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )