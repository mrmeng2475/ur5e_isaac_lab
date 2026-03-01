#!/usr/bin/env python3

# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Script to convert OBJ files to USD and visualize them in Isaac Sim."""

import argparse
import os
from pathlib import Path

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Convert OBJ to USD and visualize in Isaac Sim.")
# 👉 修改为 obj_path
parser.add_argument("--obj_path", type=str, required=True, help="Path to the OBJ file to convert.")
parser.add_argument(
    "--output_dir", type=str, default=None, help="Output directory for USD file. If not specified, uses the same directory as OBJ."
)
parser.add_argument(
    "--visualize", type=bool, default=False, help="Whether to visualize the converted USD file in Isaac Sim. Default: False."
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

# Import after app is launched
import omni.client
from pxr import Usd, UsdGeom, Sdf
import omni.usd


def convert_obj_to_usd(obj_path: str, output_path: str) -> bool:
    """
    Convert OBJ file to USD format with baked physics properties.
    """
    from pxr import UsdPhysics # 引入物理引擎 API
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Create a new USD stage
        stage = Usd.Stage.CreateNew(output_path)

        # Set up the stage with default parameters
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)

        # Create a root Xform prim and set it as Default Prim
        root_prim = UsdGeom.Xform.Define(stage, "/Root")
        stage.SetDefaultPrim(root_prim.GetPrim())

        # 把根节点注册为刚体，并永久写入 0.1kg 的质量
        UsdPhysics.RigidBodyAPI.Apply(root_prim.GetPrim())
        mass_api = UsdPhysics.MassAPI.Apply(root_prim.GetPrim())
        mass_api.CreateMassAttr().Set(0.1)

        # Create a mesh prim and import the OBJ
        mesh_prim_path = "/Root/Mesh"
        mesh = UsdGeom.Mesh.Define(stage, mesh_prim_path)

        # 👉 调用新的 OBJ 读取函数
        vertices, faces, face_vertex_counts = read_obj_file(obj_path)

        # Set mesh properties
        mesh.GetPointsAttr().Set(vertices)
        mesh.GetFaceVertexIndicesAttr().Set(faces)
        # 👉 动态设置每个面的顶点数（因为 OBJ 可能包含四边形等）
        mesh.GetFaceVertexCountsAttr().Set(face_vertex_counts)

        # Add smooth shading
        mesh.CreateDisplayColorAttr().Set([(0.8, 0.8, 0.8)] * len(vertices))

        # 给网格加上碰撞属性
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
        # 👉 强烈推荐：使用凸分解 (Convex Decomposition) 完美贴合凹陷零件
        mesh_collision.CreateApproximationAttr().Set(UsdPhysics.Tokens.convexDecomposition)

        # Save the stage
        stage.GetRootLayer().Save()

        print(f"✓ Successfully converted {obj_path} to {output_path}")
        return True

    except Exception as e:
        print(f"✗ Error converting OBJ to USD: {e}")
        return False

def read_obj_file(obj_path: str):
    """
    👉 新增：读取 OBJ 文件并提取顶点和面数据。
    
    Args:
        obj_path: Path to the OBJ file

    Returns:
        Tuple of (vertices, faces, face_vertex_counts)
    """
    vertices = []
    faces = []
    face_vertex_counts = []

    with open(obj_path, "r", encoding="utf-8") as f:
        for line in f:
            # 去除首尾空白字符
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            
            # 解析顶点 (Vertex)
            if parts[0] == 'v':
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            
            # 解析面 (Face)
            elif parts[0] == 'f':
                # OBJ 的面可以有 3 个或更多顶点
                face_vertices = parts[1:]
                face_vertex_counts.append(len(face_vertices))
                
                for p in face_vertices:
                    # OBJ 格式通常为 顶点索引/纹理索引/法线索引 (如 1/1/1)
                    # 我们只需要提取第一个（即顶点索引）
                    v_idx_str = p.split('/')[0]
                    v_idx = int(v_idx_str)
                    
                    # OBJ 的索引是从 1 开始的，USD 和 Python 是从 0 开始的
                    if v_idx > 0:
                        faces.append(v_idx - 1)
                    else:
                        # 处理相对索引（罕见但符合标准）
                        faces.append(len(vertices) + v_idx)

    return vertices, faces, face_vertex_counts


def main():
    """Main function to convert and visualize OBJ."""
    obj_path = args_cli.obj_path
    
    # Validate OBJ file exists
    if not os.path.exists(obj_path):
        print(f"✗ OBJ file not found: {obj_path}")
        simulation_app.close()
        return

    # Determine output path
    if args_cli.output_dir:
        output_dir = args_cli.output_dir
    else:
        output_dir = os.path.dirname(obj_path)

    usd_filename = os.path.splitext(os.path.basename(obj_path))[0] + ".usd"
    usd_path = os.path.abspath(os.path.join(output_dir, usd_filename))
    obj_path = os.path.abspath(obj_path)

    print(f"Converting OBJ to USD...")
    print(f"  Input:  {obj_path}")
    print(f"  Output: {usd_path}")

    # Convert OBJ to USD
    if not convert_obj_to_usd(obj_path, usd_path):
        simulation_app.close()
        return

    # Check if visualization is enabled
    if not args_cli.visualize:
        print(f"\n✓ Conversion completed. Visualization disabled.")
        print(f"  USD file saved at: {usd_path}")
        simulation_app.close()
        return

    print(f"\nLaunching Isaac Sim visualization...")

    # Get the stage
    from isaacsim.core.utils.stage import create_new_stage
    create_new_stage()
    
    # Get stage reference
    stage = omni.usd.get_context().get_stage()
    
    # Create root prim if it doesn't exist
    if not stage.GetPrimAtPath("/World"):
        stage.DefinePrim("/World", "Xform")
    
    # Add reference to the converted USD file
    try:
        prim_path = "/World/ImportedModel"
        prim = stage.DefinePrim(prim_path, "Xform")
        prim.GetReferences().AddReference(usd_path)
        print(f"✓ Added USD model to stage: {usd_path}")
    except Exception as e:
        print(f"✗ Error adding USD to stage: {e}")
        simulation_app.close()
        return

    # Simulation loop
    print("\nSimulation running. Close the window to exit.")
    print(f"Visualizing: {usd_path}")

    while simulation_app.is_running():
        simulation_app.update()

    simulation_app.close()


if __name__ == "__main__":
    main()