import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from pathlib import Path

# Define UR5E_ROOT_DIR as the root directory of the ur5e package
UR5E_ROOT_DIR = str(Path(__file__).parent.parent)
# 定义您的ur5e配置
UR5E_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{UR5E_ROOT_DIR}/usd/ur5e_zhiyuan/urdf/ur5e/ur5e.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": 0.0,
            "elbow_joint": 0.0,
            "wrist_1_joint": 0.0,
            "wrist_2_joint": 0.0,
            "wrist_3_joint": 0.0,

            "l_f_joint_1": 0.0,
            "l_f_joint1_2": 0.0,
            "l_f_joint1_3": 0.0,
            "l_f_joint1_4": 0.0,

            "l_f_joint2_1": 0.0,
            "l_f_joint2_2": 0.0,
            "l_f_joint2_3": 0.0,

            "l_f_joint3_1": 0.0,
            "l_f_joint3_2": 0.0,
            "l_f_joint3_3": 0.0,
        },
    ),
    actuators={
        "ur5e": ImplicitActuatorCfg(
            joint_names_expr=[
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
            velocity_limit_sim={
                "shoulder_pan_joint": 2.175,
                "shoulder_lift_joint": 2.175,
                "elbow_joint": 2.175,
                "wrist_1_joint": 2.61,
                "wrist_2_joint": 2.61,
                "wrist_3_joint": 2.61,
            },
            effort_limit_sim={
                "shoulder_pan_joint": 40.0,
                "shoulder_lift_joint": 27.0,
                "elbow_joint": 7.0,
                "wrist_1_joint": 40.0,
                "wrist_2_joint": 27.0,
                "wrist_3_joint": 7.0,
            },
            stiffness=80.0,
            damping=4.0,
        ),
        "dexhand": ImplicitActuatorCfg(
            joint_names_expr=[
                "l_f_joint_1",
                "l_f_joint1_2",
                "l_f_joint1_3",
                "l_f_joint1_4",
                "l_f_joint2_1",
                "l_f_joint2_2",
                "l_f_joint2_3",
                "l_f_joint3_1",
                "l_f_joint3_2",
                "l_f_joint3_3",
            ],
            velocity_limit_sim=0.2,
            effort_limit_sim=5.0,
            stiffness=80,
            damping=4,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
