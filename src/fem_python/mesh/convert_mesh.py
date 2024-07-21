import os

import meshio
import numpy as np
import trimesh

from fem_python.mesh import generate_mesh


def obj2msh(obj_path: str) -> str:
    # OBJファイルを読み込む
    mesh = trimesh.load(obj_path)

    # 頂点と面を取得
    vertices = mesh.vertices
    faces = mesh.faces

    # Meshioでメッシュデータを作成
    mesh_data = meshio.Mesh(points=vertices, cells=[("triangle", faces)])

    path_wo_ext = os.path.splitext(obj_path)
    msh_path = f"{path_wo_ext}.msh"

    # MSHファイルとして保存
    mesh_data.write(msh_path)

    return msh_path


def o3dmesh2msh():
    mesh = generate_mesh.create_2d_band_ring_mesh(draw=True)

    vertices = np.array([mesh.vertices])[0, :, :2]
    faces = np.array([mesh.triangles])[0]

    # Meshioでメッシュデータを作成
    mesh_data = meshio.Mesh(points=vertices, cells=[("triangle", faces)])
    # MSHファイルとして保存
    mesh_data.write("./data/2d_band_ring.msh")


if __name__ == "__main__":
    o3dmesh2msh()
