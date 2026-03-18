import numpy as np
from numpy import sin, cos
import numba as nb
from numba_progress import ProgressBar
from tqdm import tqdm
from copy import deepcopy
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import sympy as sp
from sympy import Symbol, Abs, I, exp, diff


class LatticeLatticeStructure:
    def __init__(self, m_1, m_2, c_1, c_2, c_12, d_1, d_2, cnt_x, cnt_y, a):
        if cnt_x % 2 == 0 or cnt_y % 2 == 0:
            raise ValueError("Количество частиц вдоль каждой оси должно быть нечётным")
        self.a = a

        self.indices_x = np.tile(np.arange(-(cnt_x // 2), cnt_x // 2 + 1, 1), (cnt_y, 1))
        self.indices_y = np.tile(np.arange(-(cnt_y // 2), cnt_y // 2 + 1, 1)[::-1, None], (1, cnt_x))
        self.coords_x = a * self.indices_x
        self.coords_y = a * self.indices_y
        self.masses = m_1 * (self.indices_x < 0) + m_2 * (self.indices_x >= 0)
        self.stiffnesses = c_1 * (self.indices_x < -1) + c_12 * (self.indices_x == -1) + c_2 * (self.indices_x > -1)
        self.foundation_stiffnesses = d_1 * (self.indices_x < 0) + d_2 * (self.indices_x >= 0)

        self.disp = np.zeros(shape=(cnt_y, cnt_x))
        self.vel = np.zeros(shape=(cnt_y, cnt_x))

    def specify_initial_and_boundary(self, gamma, beta_x, beta_y,
                                     u_0, shift_x=None, shift_y=None, omega=None, omega_undim=None):
        setattr(self, "gamma", gamma)

        if omega_undim is not None:
            omega = np.sqrt(self.omega_low ** 2 + omega_undim ** 2 * (self.omega_high ** 2 - self.omega_low ** 2))
        if shift_x is None:
            shift_x = -3 / beta_x
        if shift_y is None:
            shift_y = -3 / beta_y

        setattr(self, "omega", omega)
        setattr(self, "u_0", u_0)
        setattr(self, "beta_x", beta_x)
        setattr(self, "shift_x", shift_x)

        k_1 = fsolve(lambda k: self.masses[0, 0] * omega ** 2 - self.foundation_stiffnesses[0, 0] -
                     4 * self.stiffnesses[0, 0] *
                     (sin(cos(gamma) * k * self.a / 2) ** 2 + sin(sin(gamma) * k * self.a / 2) ** 2), np.ones(1))[0]
        g_1 = 4 * self.stiffnesses * \
            (cos(gamma) * self.a / 2 * sin(k_1 * cos(gamma) * self.a / 2) * cos(k_1 * cos(gamma) * self.a / 2) +
             sin(gamma) * self.a / 2 * sin(k_1 * sin(gamma) * self.a / 2) * cos(k_1 * sin(gamma) * self.a / 2)) / \
            (self.masses * np.sqrt((4 * self.stiffnesses * (sin(k_1 * cos(gamma) * self.a / 2)) ** 2 +
                                   4 * self.stiffnesses * (sin(k_1 * sin(gamma) * self.a / 2)) ** 2 +
                                   self.foundation_stiffnesses) / self.masses))

        setattr(self, "g_1", g_1)
        print(self.omega_low)
        print(self.omega_high)
        print("Текущий k:", k_1)

        self.disp = u_0 * np.exp(-beta_x ** 2 / 2 * (self.coords_x * cos(gamma) + self.coords_y * sin(gamma) -
                                                     shift_x * cos(gamma) - shift_y * sin(gamma)) ** 2)
        self.disp *= np.exp(-beta_y ** 2 / 2 * (-self.coords_x * sin(gamma) + self.coords_y * cos(gamma) +
                                                shift_x * sin(gamma) - shift_y * cos(gamma)) ** 2)
        self.disp *= sin(k_1 * cos(gamma) * self.coords_x + k_1 * sin(gamma) * self.coords_y)

        self.vel = -u_0 * np.exp(-beta_x ** 2 / 2 * (self.coords_x * cos(gamma) + self.coords_y * sin(gamma) -
                                                     shift_x * cos(gamma) - shift_y * sin(gamma)) ** 2)
        self.vel *= np.exp(-beta_y ** 2 / 2 * (-self.coords_x * sin(gamma) + self.coords_y * cos(gamma) +
                                               shift_x * sin(gamma) - shift_y * cos(gamma)) ** 2)
        self.vel *= (omega * cos(k_1 * cos(gamma) * self.coords_x + k_1 * sin(gamma) * self.coords_y) -
                     beta_x ** 2 * g_1 / self.a * (self.coords_x * cos(gamma) + self.coords_y * sin(gamma) -
                                                   shift_x * cos(gamma) - shift_y * sin(gamma)) *
                     sin(k_1 * cos(gamma) * self.coords_x + k_1 * sin(gamma) * self.coords_y))
        self.disp[np.where(self.indices_x >= -1)] = 0
        self.vel[np.where(self.indices_x >= -1)] = 0

    def solve(self, dt=None, t_max=None, save_time=None, auto_stop=True, accelerate=False):
        if dt is None:
            # dt = 0.05 / self.omega_high
            dt = 0.05
        if t_max is None:
            t_max = 3 * abs(getattr(self, "shift_x")) * self.a / \
                    (getattr(self, "g_1")[0, 0] * cos(getattr(self, "gamma")))
        if save_time is None:
            save_time = 15

        time_steps = np.arange(0, t_max, dt)
        if accelerate:
            with ProgressBar(total=len(time_steps)) as progress:
                self.disp, self.vel = numba_accelerate_2(dt, time_steps, self.masses, self.disp, self.vel,
                                                         self.stiffnesses, self.foundation_stiffnesses, progress)
        else:
            for t in tqdm(time_steps):
                # leapfrog synchronized form
                acc1 = (self.stiffnesses / self.masses) * (np.roll(self.disp, -1, axis=1) +
                                                           np.roll(self.disp, 1, axis=0) -
                                                           2 * self.disp) + \
                       (np.roll(self.stiffnesses, 1, axis=1) / self.masses) * (np.roll(self.disp, 1, axis=1) +
                                                                               np.roll(self.disp, -1, axis=0) -
                                                                               2 * self.disp)
                acc1 -= self.foundation_stiffnesses / self.masses * self.disp
                self.disp += self.vel * dt + 1 / 2 * acc1 * dt ** 2
                acc2 = (self.stiffnesses / self.masses) * (np.roll(self.disp, -1, axis=1) +
                                                           np.roll(self.disp, 1, axis=0) -
                                                           2 * self.disp) + \
                       (np.roll(self.stiffnesses, 1, axis=1) / self.masses) * (np.roll(self.disp, 1, axis=1) +
                                                                               np.roll(self.disp, -1, axis=0) -
                                                                               2 * self.disp)
                acc2 -= self.foundation_stiffnesses / self.masses * self.disp
                self.vel += 1 / 2 * (acc1 + acc2) * dt

                # save results
                if t % save_time == 0:
                    self.save_history(t)

                if auto_stop:
                    interface_energy = getattr(self, "energy_interface_undim_frames", None)
                    if interface_energy and interface_energy[-1] < max(interface_energy) / 1e3:
                        break

    @property
    def energy_field(self):
        e = self.masses / 2 * self.vel ** 2 + \
            self.stiffnesses / 4 * ((np.roll(self.disp, -1, axis=1) - self.disp) ** 2 +
                                    (np.roll(self.disp, 1, axis=0) - self.disp) ** 2) + \
            np.roll(self.stiffnesses, 1, axis=1) / 4 * ((np.roll(self.disp, 1, axis=1) - self.disp) ** 2 +
                                                        (np.roll(self.disp, -1, axis=0) - self.disp) ** 2) +\
            self.foundation_stiffnesses / 2 * self.disp ** 2
        return e

    @property
    def disp_undim(self):
        u_0 = getattr(self, "u_0")
        return self.disp / u_0

    @property
    def vel_undim(self):
        omega = getattr(self, "omega")
        return self.vel / (self.a * omega)

    @property
    def energy_field_undim(self):
        m_1 = self.masses[0, 0]
        u_0 = getattr(self, "u_0")
        omega = getattr(self, "omega")
        return 2 * self.energy_field / (m_1 * u_0 ** 2 * omega ** 2)

    @property
    def energy_both_undim(self):
        return np.sum(self.energy_field_undim)

    @property
    def energy_left_undim(self):
        return np.sum(self.energy_field_undim * (self.indices_x < 0))

    @property
    def energy_interface_undim(self):
        return np.sum(self.energy_field_undim * (self.indices_x == 0))

    @property
    def energy_right_undim(self):
        return np.sum(self.energy_field_undim * (self.indices_x >= 0))

    @property
    def transmission_coeff_numerical(self):
        return self.energy_right_undim / self.energy_both_undim

    @property
    def transmission_coeff_analytical(self):
        gamma = getattr(self, "gamma")
        theta, k_1, k_2 = self.theta
        k_1_x = k_1 * cos(gamma)
        k_2_x = k_2 * cos(theta)

        k = Symbol("k")
        g_1 = diff(sp.sqrt((4 * self.stiffnesses[0, 0] * (sp.sin(k * cos(gamma) * self.a / 2) ** 2 +
                            sp.sin(k * sin(gamma) * self.a / 2) ** 2) +
                            self.foundation_stiffnesses[0, 0]) / self.masses[0, 0]), k).evalf(subs={k: k_1})
        g_1_x = g_1 * np.cos(gamma)
        g_1_y = g_1 * np.sin(gamma)
        g_2 = diff(sp.sqrt((4 * self.stiffnesses[0, -1] * (sp.sin(k * cos(theta) * self.a / 2) ** 2 +
                            sp.sin(k * sin(theta) * self.a / 2) ** 2) +
                            self.foundation_stiffnesses[0, -1]) / self.masses[0, -1]), k).evalf(subs={k: k_2})
        g_2_x = g_2 * np.cos(theta)
        g_2_y = g_2 * np.sin(theta)

        amp_frac = (exp(I * k_1_x * self.a) - exp(-I * k_1_x * self.a)) / \
                   (exp(I * k_2_x * self.a) - exp(-I * k_1_x * self.a))
        amp_frac = amp_frac.evalf()

        trans_coeff = ((self.masses[0, -1] * g_2_x) /
                       (self.masses[0, 0] * g_1_x)) * (Abs(amp_frac)) ** 2

        # return Abs(amp_frac)
        return trans_coeff

    @property
    def adjustment_coeff(self):
        gamma = getattr(self, "gamma")
        theta, _, _ = self.theta
        m_1, m_2 = self.masses[0, 0], self.masses[0, -1]
        c_1, c_2 = self.stiffnesses[0, 0], self.stiffnesses[0, -1]
        return ((np.sqrt(m_1 * c_1) * cos(gamma) + np.sqrt(m_2 * c_2) * cos(theta)) /
                (np.sqrt(m_2 * c_2) * cos(gamma) + np.sqrt(m_1 * c_1) * cos(theta))) ** 2

    @property
    def transmission_coeff_analytical_adjustment(self):
        return self.transmission_coeff_analytical * self.adjustment_coeff

    @property
    def transmission_coeff_continuum(self):
        gamma = getattr(self, "gamma")
        m_1 = self.masses[0, 0]
        m_2 = self.masses[0, -1]
        c_1 = self.stiffnesses[0, 0]
        c_2 = self.stiffnesses[0, -1]
        theta = np.arcsin(np.sqrt(m_1 / m_2) * sin(gamma))
        trans_coeff = (4 * cos(gamma) * cos(theta) / (np.sqrt(m_2) * np.sqrt(m_1))) / \
                      (cos(gamma) / np.sqrt(m_1) + cos(theta) / np.sqrt(m_2)) ** 2
        # return Abs(2 * (m_2 / m_1) * cos(gamma) / ((m_2 / m_1) * cos(gamma) + np.sqrt(m_2 / m_1) * cos(theta)))
        return trans_coeff

    @property
    def omega_low(self):
        gamma = getattr(self, "gamma")
        c_1, c_2 = self.stiffnesses[0, 0], self.stiffnesses[0, -1]
        m_1, m_2 = self.masses[0, 0], self.masses[0, -1]
        d_1, d_2 = self.foundation_stiffnesses[0, 0], self.foundation_stiffnesses[0, -1]
        #lst = [(sin(cos(gamma) * var)) ** 2 + (sin(sin(gamma) * var)) ** 2 for var in np.arange(0, 2 * np.pi, 0.001)]
        lst = [0]
        return np.sqrt(max((4 * c_1 * min(lst) + d_1) / m_1,
                           (4 * c_2 * min(lst) + d_2) / m_2))

    @property
    def omega_high(self):
        gamma = getattr(self, "gamma")
        c_1, c_2 = self.stiffnesses[0, 0], self.stiffnesses[0, -1]
        m_1, m_2 = self.masses[0, 0], self.masses[0, -1]
        d_1, d_2 = self.foundation_stiffnesses[0, 0], self.foundation_stiffnesses[0, -1]
        #lst = [(sin(cos(gamma) * var)) ** 2 + (sin(sin(gamma) * var)) ** 2 for var in np.arange(0, 2 * np.pi, 0.001)]
        lst = [1]
        return np.sqrt(min((4 * c_1 * max(lst) + d_1) / m_1,
                           (4 * c_2 * max(lst) + d_2) / m_2))

    @property
    def theta(self):
        gamma = getattr(self, "gamma")
        omega = getattr(self, "omega")
        k_1 = fsolve(lambda k: self.masses[0, 0] * omega ** 2 - self.foundation_stiffnesses[0, 0] -
                     4 * self.stiffnesses[0, 0] * (sin(cos(gamma) * k * self.a / 2) ** 2 +
                                                   sin(sin(gamma) * k * self.a / 2) ** 2), np.ones(1))[0]
        k_1_y = k_1 * sin(gamma)

        k_2_y = k_1_y
        k_2_x = fsolve(lambda k_x: self.masses[0, -1] * omega ** 2 - self.foundation_stiffnesses[0, -1] -
                       4 * self.stiffnesses[0, -1] * (sin(k_x * self.a / 2) ** 2 + sin(k_2_y * self.a / 2) ** 2),
                       np.array([0.5]))[0]
        k_2 = np.sqrt(k_2_x ** 2 + k_2_y ** 2)
        theta = np.arctan(k_2_y / k_2_x)
        return theta, k_1, k_2

    def plot_field(self, field="energy_field_undim", title="Энергия",
                   x_label="n", y_label="m", cbar_label=r"$2e_{n,m} \;/\; \left(m_1U_0^2\Omega^2\right)$"):
        cur_field = getattr(self, field)
        #levels = np.linspace(cur_field.min(), cur_field.max(), 100)
        levels = np.linspace(0, 0.01, 10)
        fig, ax = plt.subplots()
        ax.plot([0] * self.coords_y.shape[0], self.coords_y[:, 0], linestyle="dashed", color="red", linewidth=1)
        cf = ax.contourf(self.coords_x, self.coords_y, cur_field, levels=levels)
        cbar = fig.colorbar(cf, ticks=np.linspace(0, cur_field.max(), 10), label=cbar_label, ax=ax)
        ax = plt.gca()
        plt.title(f"{title} {cbar_label}")
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        ax.set_aspect("equal", adjustable="box")

        # ax.plot(self.coords_x[0][np.where(self.coords_x[0] >= 0)],
        #        np.add(np.tan(self.theta[0]) * self.coords_x[0][np.where(self.coords_x[0] >= 0)], -20),
        #        linestyle="dashed", color="orange", linewidth=1)

        plt.show()

    frames_containers = ["time_undim_frames", "disp_undim_frames", "vel_undim_frames", "energy_field_undim_frames",
                         "energy_both_undim_frames", "energy_left_undim_frames",
                         "energy_right_undim_frames", "energy_interface_undim_frames",
                         "transmission_coeff_numerical_frames"]
    frames_container_names = list(map(lambda s: s.replace("_frames", ""), frames_containers))

    def save_history(self, t):
        setattr(self, "time_undim", t * getattr(self, "g_1")[0, 0] * cos(getattr(self, "gamma")) /
                (abs(getattr(self, "shift_x")) * self.a))
        for i, frames_container in enumerate(self.frames_containers):
            if not hasattr(self, frames_container):
                setattr(self, frames_container, [])
            getattr(self, frames_container).append(deepcopy(getattr(self, self.frames_container_names[i])))


@nb.jit(nopython=True, nogil=True, inline="always", parallel=True)
def disp_neighbors(disp, stiff):
    disp_m1_1 = np.concatenate((disp[:, 1:], disp[:, :1]), axis=1)
    disp_p1_0 = np.concatenate((disp[-1:, :], disp[:-1, :]), axis=0)
    disp_p1_1 = np.concatenate((disp[:, -1:], disp[:, :-1]), axis=1)
    disp_m1_0 = np.concatenate((disp[1:, :], disp[:1, :]), axis=0)
    stiff_p1_1 = np.concatenate((stiff[:, -1:], stiff[:, :-1]), axis=1)
    return disp_m1_1, disp_p1_0, disp_p1_1, disp_m1_0, stiff_p1_1


@nb.jit(nopython=True, nogil=True, parallel=True)
def numba_accelerate_1(dt, time_steps, masses, disp, vel, stiffnesses, foundation_stiffnesses, progress_proxy):
    for t in time_steps:
        disp_m1_1, disp_p1_0, disp_p1_1, disp_m1_0, stiff_p1_1 = disp_neighbors(disp, stiffnesses)
        # leapfrog synchronized form
        acc1 = (stiffnesses / masses) * (disp_m1_1 + disp_p1_0 - 2 * disp) + \
               (stiff_p1_1 / masses) * (disp_p1_1 + disp_m1_0 - 2 * disp)
        acc1 -= foundation_stiffnesses / masses * disp
        disp += vel * dt + 1 / 2 * acc1 * dt ** 2

        disp_m1_1, disp_p1_0, disp_p1_1, disp_m1_0, stiff_p1_1 = disp_neighbors(disp, stiffnesses)

        acc2 = (stiffnesses / masses) * (disp_m1_1 + disp_p1_0 - 2 * disp) + \
               (stiff_p1_1 / masses) * (disp_p1_1 + disp_m1_0 - 2 * disp)
        acc2 -= foundation_stiffnesses / masses * disp
        vel += 1 / 2 * (acc1 + acc2) * dt
        progress_proxy.update(1)
    return disp, vel


@nb.jit(nopython=True, nogil=True)
def numba_accelerate_2(dt, time_steps, masses, disp, vel,
                       stiff, foundation_stiffnesses, progress_proxy):
    acc1 = np.zeros_like(disp)
    acc2 = np.zeros_like(disp)
    n = len(disp[0])
    m = len(disp)
    for t in time_steps:
        for i in range(m):
            for j in range(n):
                acc1[i, j] = (stiff[i, j] / masses[i, j]) * (disp[i, (j + 1) % n] + disp[i - 1, j] - 2 * disp[i, j]) + \
                             (stiff[i, j - 1] / masses[i, j]) * (disp[i, j - 1] + disp[(i + 1) % m, j] - 2 * disp[i, j])
                acc1[i, j] -= foundation_stiffnesses[i, j] / masses[i, j] * disp[i, j]

        for i in range(m):
            for j in range(n):
                disp[i, j] += vel[i, j] * dt + 1 / 2 * acc1[i, j] * dt ** 2

        for i in range(m):
            for j in range(n):
                acc2[i, j] = (stiff[i, j] / masses[i, j]) * (disp[i, (j + 1) % n] + disp[i - 1, j] - 2 * disp[i, j]) + \
                             (stiff[i, j - 1] / masses[i, j]) * (disp[i, j - 1] + disp[(i + 1) % m, j] - 2 * disp[i, j])
                acc2[i, j] -= foundation_stiffnesses[i, j] / masses[i, j] * disp[i, j]
                vel[i, j] += 1 / 2 * (acc1[i, j] + acc2[i, j]) * dt
        progress_proxy.update(1)
    return disp, vel


if __name__ == "__main__":
    lattice_lattice = LatticeLatticeStructure(m_1=0.5, m_2=1.0,
                                              c_1=0.1, c_2=0.1, c_12=0.1,
                                              d_1=0.0, d_2=0.2,
                                              cnt_x=601, cnt_y=601, a=1)
    lattice_lattice.specify_initial_and_boundary(gamma=np.radians(0), beta_x=0.02, beta_y=0.02,
                                                 u_0=1, omega_undim=np.sqrt(0.5))
    lattice_lattice.plot_field()
    lattice_lattice.solve(auto_stop=False, accelerate=True)
    lattice_lattice.plot_field()
