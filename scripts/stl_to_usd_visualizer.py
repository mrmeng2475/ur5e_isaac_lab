#!/usr/bin/env python3

# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to convert STL files to USD and visualize them in Isaac Sim."""

import argparse
import os
from pathlib import Path

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Convert STL to USD and visualize in Isaac Sim.")
parser.add_argument("--stl_path", type=str, required=True, help="Path to the STL file to convert.")
parser.add_argument(
    "--output_dir", type=str, default=None, help="Output directory for USD file. If not specified, uses the same directory as STL."
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


def convert_stl_to_usd(stl_path: str, output_path: str) -> bool:
    """
    Convert STL file to USD format with baked physics properties.
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

        # 👉 核心新增 1：把根节点注册为刚体，并永久写入 0.5kg 的质量
        UsdPhysics.RigidBodyAPI.Apply(root_prim.GetPrim())
        mass_api = UsdPhysics.MassAPI.Apply(root_prim.GetPrim())
        mass_api.CreateMassAttr().Set(0.5)

        # Create a mesh prim and import the STL
        mesh_prim_path = "/Root/Mesh"
        mesh = UsdGeom.Mesh.Define(stage, mesh_prim_path)

        # Read STL file and extract vertices and faces
        vertices, faces = read_stl_file(stl_path)

        # Set mesh properties
        mesh.GetPointsAttr().Set(vertices)
        mesh.GetFaceVertexIndicesAttr().Set(faces)
        mesh.GetFaceVertexCountsAttr().Set([3] * (len(faces) // 3))

        # Add smooth shading
        mesh.CreateDisplayColorAttr().Set([(0.8, 0.8, 0.8)] * len(vertices))

        # 👉 核心新增 2：给网格加上凸包碰撞属性
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
        mesh_collision.CreateApproximationAttr().Set(UsdPhysics.Tokens.convexHull)

        # Save the stage
        stage.GetRootLayer().Save()

        print(f"✓ Successfully converted {stl_path} to {output_path}")
        return True

    except Exception as e:
        print(f"✗ Error converting STL to USD: {e}")
        return False

def read_stl_file(stl_path: str):
    """
    Read STL file and extract vertices and faces.

    Args:
        stl_path: Path to the STL file

    Returns:
        Tuple of (vertices, faces)
    """
    import struct
    import numpy as np

    with open(stl_path, "rb") as f:
        # Skip 80-byte header
        f.read(80)

        # Read number of triangles
        num_triangles = struct.unpack("I", f.read(4))[0]

        vertices = []
        faces = []
        vertex_map = {}
        vertex_index = 0

        for _ in range(num_triangles):
            # Skip normal vector (3 floats)
            f.read(12)

            # Read triangle vertices (3 vertices × 3 floats each)
            triangle_vertices = []
            for _ in range(3):
                v = struct.unpack("fff", f.read(12))
                v_tuple = tuple(np.round(v, 6))  # Round to avoid floating point errors

                if v_tuple not in vertex_map:
                    vertex_map[v_tuple] = vertex_index
                    vertices.append(list(v))
                    vertex_index += 1

                triangle_vertices.append(vertex_map[v_tuple])

            faces.extend(triangle_vertices)

            # Skip attribute byte count
            f.read(2)

    return vertices, faces


def main():
    """Main function to convert and visualize STL."""
    stl_path = args_cli.stl_path
    
    # Validate STL file exists
    if not os.path.exists(stl_path):
        print(f"✗ STL file not found: {stl_path}")
        simulation_app.close()
        return

    # Determine output path
    if args_cli.output_dir:
        output_dir = args_cli.output_dir
    else:
        output_dir = os.path.dirname(stl_path)

    usd_filename = os.path.splitext(os.path.basename(stl_path))[0] + ".usd"
    # 👉 核心修复：强制转换为绝对路径，防止跨目录引用失败
    usd_path = os.path.abspath(os.path.join(output_dir, usd_filename))
    
    # 顺手把 stl_path 也转成绝对路径，更安全
    stl_path = os.path.abspath(stl_path)

    print(f"Converting STL to USD...")
    print(f"  Input:  {stl_path}")
    print(f"  Output: {usd_path}")

    # Convert STL to USD
    if not convert_stl_to_usd(stl_path, usd_path):
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
        world_prim = stage.DefinePrim("/World", "Xform")
    
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
