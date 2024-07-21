import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def create_2d_band_ring_points(
    r0: float = 5, r1: float = 10, num_points: int = 10000, draw: bool = False
) -> np.ndarray:
    if r0 < 0 or r1 < 0:
        raise RuntimeError(f"Invalid arguments: {r0=}, {r1=}")

    angle = np.random.uniform(0, 2 * np.pi, num_points)
    radius = np.random.uniform(r0, r1, num_points)

    x = radius * np.cos(angle)
    y = radius * np.sin(angle)

    points = np.vstack([x, y]).T

    if draw:
        plt.axes().set_aspect("equal")
        plt.scatter(points[:, 0], points[:, 1])
        plt.show()

    return points


def create_2d_band_ring_mesh(draw: bool = False) -> o3d.geometry.TriangleMesh:
    points = create_2d_band_ring_points(draw=False)
    points = np.hstack([points, np.zeros(len(points))[:, None]])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=10)
    )
    pcd.orient_normals_to_align_with_direction()
    pcd = pcd.voxel_down_sample(voxel_size=0.3)

    if draw:
        o3d.visualization.draw_geometries([pcd])

    radii = [0.05, 0.1, 0.2, 0.4]
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii)
    )

    mesh = mesh.simplify_vertex_clustering(
        voxel_size=0.5,
        contraction=o3d.geometry.SimplificationContraction.Average,
    )

    if draw:
        pcd.paint_uniform_color([1, 0, 0])
        o3d.visualization.draw_geometries([mesh, pcd])

    return mesh


if __name__ == "__main__":
    create_2d_band_ring_mesh()
