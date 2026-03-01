import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from pathlib import Path

UR5E_JOINT_LIMITS = {
    # 机械臂 (举例：限制大臂只能在一个安全的小范围内活动)
    "shoulder_pan_joint": [-1.57, 1.57],
    "shoulder_lift_joint": [-2.0, -1.0], # 这是一个典型的不对称范围！
    "elbow_joint": [1.0, 2.0],
    "wrist_1_joint": [-1.57, 1.57],
    "wrist_2_joint": [-1.57, 1.57],
    "wrist_3_joint": [-3.14, 3.14],
    
    # 灵巧手 (举例：限制在 0 到 1.5 弧度之间开合)
    "l_f_joint_1": [0.0, 1.5],
    "l_f_joint1_2": [0.0, 1.5],
    "l_f_joint1_3": [0.0, 1.5],
    "l_f_joint1_4": [0.0, 1.5],
    "l_f_joint2_1": [0.0, 1.5],
    "l_f_joint2_2": [0.0, 1.5],
    "l_f_joint2_3": [0.0, 1.5],
    "l_f_joint3_1": [0.0, 1.5],
    "l_f_joint3_2": [0.0, 1.5],
    "l_f_joint3_3": [0.0, 1.5],
}

ur5e_init_pos = {}
UR5E_ACTION_SCALES = {}

for joint_name, limits in UR5E_JOINT_LIMITS.items():
    min_val, max_val = limits
    # 中心点：(max + min) / 2
    ur5e_init_pos[joint_name] = (max_val + min_val) / 2.0
    # 半跨度：(max - min) / 2
    UR5E_ACTION_SCALES[joint_name] = (max_val - min_val) / 2.0

# Define UR5E_ROOT_DIR as the root directory of the ur5e package
UR5E_ROOT_DIR = str(Path(__file__).parent.parent)
# 定义您的ur5e配置
UR5E_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{UR5E_ROOT_DIR}/usd/ur5e_zhiyuan/urdf/ur5e/ur5e.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),

    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),
        joint_pos={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.5708,
            "elbow_joint": 1.5708,
            "wrist_1_joint": 0.0,
            "wrist_2_joint": 0.0,
            "wrist_3_joint": -1.5708,

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
                "shoulder_pan_joint": 200.0,
                "shoulder_lift_joint": 200.0,
                "elbow_joint": 200.0,
                "wrist_1_joint": 80.0,
                "wrist_2_joint": 80.0,
                "wrist_3_joint": 40.0,
            },
            stiffness=50.0,
            damping=5.0,
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
            velocity_limit_sim=0.8,
            effort_limit_sim=10,
            stiffness=30,
            damping=3,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
