"""
see https://github.com/kinnala/scikit-fem/blob/9.1.1/docs/examples/ex18.py
"""

from os.path import splitext
from pathlib import Path
from sys import argv

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.tri import Triangulation
from scipy.integrate import solve_ivp
from skfem import *
from skfem.io.json import from_file
from skfem.models.general import divergence, rot
from skfem.models.poisson import laplace, mass, vector_laplace
from skfem.visuals.matplotlib import draw, plot, savefig


def create_basis(mesh):
    element = {"u": ElementVector(ElementTriP2()), "p": ElementTriP1()}
    basis = {variable: Basis(mesh, e, intorder=3) for variable, e in element.items()}
    basis["psi"] = basis["u"].with_element(ElementTriP2())
    return basis


@LinearForm
def body_force(v, w):
    return w.x[0] * v[1]


class Stokes:
    def __init__(self, mesh, dt: float = 0.02):
        self.dt = dt
        self.basis = create_basis(mesh)

        self.K, self.f = self.coeff()

    def coeff(self):
        """
        定常的ストークス流れ

        :return: 係数行列, 外力ベクトル
        """
        A = asm(vector_laplace, self.basis["u"])
        B = asm(divergence, self.basis["u"], self.basis["p"])
        C = asm(mass, self.basis["p"])

        # (Δu-∇p, -∇・u)
        K = bmat([[A, -B.T], [-B, 1e-6 * C]], "csr")
        # (f, 0)
        f = np.concatenate([asm(body_force, self.basis["u"]), self.basis["p"].zeros()])

        return K, f

    def solve_uvp(self):
        uvp = solve(*condense(self.K, self.f, D=self.basis["u"].get_dofs()))
        velocity, pressure = np.split(uvp, self.K.blocks)
        return velocity, pressure

    def solve_psi(self, velocity):
        A = asm(laplace, self.basis["psi"])
        vorticity = asm(rot, self.basis["psi"], w=self.basis["u"].interpolate(velocity))
        psi = solve(*condense(A, vorticity, D=self.basis["psi"].get_dofs()))
        return psi

    def evolve(self, t: float):
        velocity, pressure = self.solve_uvp()
        psi = self.solve_psi(velocity)

        u_sol = solve_ivp(
            diff_operator,
            t_span=(0, self.dt),
            y0=np.hstack([u_start.reshape(-1), np.zeros(mat_num)]),
            method="RK45",
            dense_output=True,
            rtol=1e-8,
        )


class AnimCreator:
    def __init__(self):
        mesh = MeshTri.init_circle(4)
        self.simulator = Stokes(mesh)

        velocity, pressure = self.simulator.solve_uvp()
        velocity1 = velocity[self.simulator.basis["u"].nodal_dofs]
        psi = self.simulator.solve_psi(velocity)

        p_ax = plot(self.simulator.basis["p"], pressure, ax=draw(mesh))

        v_ax = draw(mesh)
        v_ax.quiver(*mesh.p, *velocity1, mesh.p[0])  # colour by buoyancy

        psi_ax = draw(mesh)
        psi_ax.tricontour(
            Triangulation(*mesh.p, mesh.t.T),
            psi[self.simulator.basis["psi"].nodal_dofs.flatten()],
        )

        plt.show()

    def update(self, event: Tuple[float, np.ndarray]) -> None:
        t, velocity = event

    def create(self):
        anim = FuncAnimation()


if __name__ == "__main__":
    AnimCreator()
