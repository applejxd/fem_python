from pathlib import Path
from typing import Iterator, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib.animation import FuncAnimation
from scipy.sparse.linalg import splu
from skfem import *
from skfem.models.poisson import laplace, mass
from skfem.visuals.matplotlib import draw, plot


def create_basis(mesh):
    element = ElementTriP1()
    basis = Basis(mesh, element)
    return basis


class HeatTransfer:
    def __init__(
        self,
        mesh_path: str = "./data/2d_band_ring.msh",
        diffusivity: float = 5,
        dt: float = 0.02,
        # Crank–Nicolson 法では 0.5
        theta=0.5,
    ):
        self.mesh = MeshTri.load(mesh_path)
        # 重み関数（基底関数）を構築
        self.basis = create_basis(self.mesh)

        self.dt = dt

        self.u_init = self.create_init_state()
        self.A, self.B = self.create_coefficients(diffusivity=diffusivity, theta=theta)

    def create_init_state(self) -> np.ndarray:
        """初期状態を作成.
        中心からの距離の逆数.

        :return: 初期状態
        """
        # basis.doflocs は各節点の座標
        u_init = 1 / np.linalg.norm(self.basis.doflocs, axis=0)
        return u_init

    def create_coefficients(
        self, diffusivity: float, theta: float
    ) -> List[scipy.sparse.csr_matrix]:
        # ΔT・u -> -Tr(∇T・(∇u)^T)
        L = diffusivity * asm(laplace, self.basis)
        # T・u（現時刻の解）
        M = asm(mass, self.basis)

        # そのままなら自然境界条件（FEM の性質）
        if False:
            # 指定すればディリクレ境界条件. get_dofs() は境界の自由度.
            # see https://github.com/kinnala/scikit-fem/blob/40e4f4e05b0127909325c8dba0891cced7adb02e/skfem/utils.py#L40
            L, M = penalize(L, M, D=self.basis.get_dofs())

        # see https://itpass.scitec.kobe-u.ac.jp/~fourtran/FB-kobe/2011-FB-kobe/seminar-9/text/seminar-9.pdf
        # see https://www.astr.tohoku.ac.jp/~chinone/pdf/Diffusion_equation.pdf
        A = M + theta * L * self.dt
        B = M - (1 - theta) * L * self.dt
        return A, B

    def evolve(
        self, t: float, u: np.ndarray, t_max=2
    ) -> Iterator[Tuple[float, np.ndarray]]:
        # t_max まで計算
        while t < t_max:
            t = t + self.dt

            # 直接法
            # Au_t=Bu_{t-1} & A=LU -> Au_t=LUu_t=Bu_{t-1} -> Uu_t=y & Ly=Bu_{t-1}
            u = splu(self.A.T).solve(self.B @ u)
            yield t, u


class AnimCreator:
    def __init__(self):
        self.simulator = HeatTransfer()

        mesh_ax = draw(self.simulator.mesh)
        ax = plot(
            self.simulator.mesh,
            self.simulator.u_init[self.simulator.basis.nodal_dofs.flatten()],
            shading="gouraud",
            colorbar=True,
            # メッシュ表示と重畳
            ax=mesh_ax,
        )

        self.title = ax.set_title("t = 0.00")
        # mesh_ax に ax を重ねているため 1 番目を選択
        # vertex-based temperature-colour
        self.field = ax.get_children()[1]
        self.fig = ax.get_figure()

        # 値をチェックする地点 (x,y)=(5,5)
        self.probe = self.simulator.basis.probes(x=np.array([[6, 6]]).T)

    def update(self, event: Tuple[float, np.ndarray]) -> None:
        t, u = event
        # 指定した地点の値
        u_point = (self.probe @ u)[0]
        print(f"{t=:4.2f}, {u_point=:5.3f}")

        self.title.set_text(f"$t$ = {t:.2f}")
        self.field.set_array(u[self.simulator.basis.nodal_dofs.flatten()])

    def create(self):
        animation = FuncAnimation(
            self.fig,
            self.update,
            self.simulator.evolve(t=0.0, u=self.simulator.u_init),
            repeat=False,
            # 10 ms おきにプロット
            interval=10,
        )

        if False:
            animation.save(Path(__file__).with_suffix(".gif"), "imagemagick")
        else:
            plt.show()


if __name__ == "__main__":
    AnimCreator().create()
