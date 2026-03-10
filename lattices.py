import numpy as np
from numpy import sin, cos
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

    def specify_initial_and_boundary(self, gamma, beta_x, beta_y, n_0, v_0, u_0, omega=None, omega_undim=None):
        setattr(self, "gamma", gamma)

        if omega_undim is not None:
            omega = np.sqrt(self.omega_low ** 2 + omega_undim ** 2 * (self.omega_high ** 2 - self.omega_low ** 2))

        setattr(self, "omega", omega)
        setattr(self, "u_0", u_0)

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

        self.disp = u_0 * np.exp(-beta_x ** 2 / 2 * (self.coords_x * cos(gamma) + self.coords_y * sin(gamma) -
                                                     n_0 * cos(gamma) - v_0 * sin(gamma)) ** 2)
        self.disp *= np.exp(-beta_y ** 2 / 2 * (-self.coords_x * sin(gamma) + self.coords_y * cos(gamma) +
                                                n_0 * sin(gamma) - v_0 * cos(gamma)) ** 2)
        self.disp *= sin(k_1 * cos(gamma) * self.coords_x + k_1 * sin(gamma) * self.coords_y)

        self.vel = -u_0 * np.exp(-beta_x ** 2 / 2 * (self.coords_x * cos(gamma) + self.coords_y * sin(gamma) -
                                                     n_0 * cos(gamma) - v_0 * sin(gamma)) ** 2)
        self.vel *= np.exp(-beta_y ** 2 / 2 * (-self.coords_x * sin(gamma) + self.coords_y * cos(gamma) +
                                               n_0 * sin(gamma) - v_0 * cos(gamma)) ** 2)
        self.vel *= (omega * cos(k_1 * cos(gamma) * self.coords_x + k_1 * sin(gamma) * self.coords_y) -
                     beta_x ** 2 * g_1 / self.a * (self.coords_x * cos(gamma) + self.coords_y * sin(gamma) -
                                                   n_0 * cos(gamma)) *
                     sin(k_1 * cos(gamma) * self.coords_x + k_1 * sin(gamma) * self.coords_y))

    def solve(self, dt, t_max, save_time, auto_stop=True):
        time_steps = np.arange(0, t_max, dt)
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
            if auto_stop and len(getattr(self, "energy_right_sum_frames", [])) > 1:
                e_left = getattr(self, "energy_left_undi_frames")
                e_right = getattr(self, "energy_right_undi_frames")
                if abs(e_right[-1]) > 0.5 * e_left[0] and (e_right[-1] - e_right[-2]) < 0.01 * e_left[0]:
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
    def energy_right_undim(self):
        return np.sum(self.energy_field_undim * (self.indices_x >= 0))

    @property
    def transmission_coeff_numerical(self):
        return self.energy_right_undim / self.energy_both_undim

    @property
    def transmission_coeff_analytical(self):
        gamma = getattr(self, "gamma")
        omega = getattr(self, "omega")
        k_1 = fsolve(lambda k: self.masses[0, 0] * omega ** 2 - 4 * self.stiffnesses[0, 0] *
                     (sin(cos(gamma) * k * self.a / 2) ** 2 +
                      sin(sin(gamma) * k * self.a / 2) ** 2), np.ones(1))[0]
        k_1_x = k_1 * cos(gamma)
        k_1_y = k_1 * sin(gamma)

        k_2_y = k_1_y
        k_2_x = fsolve(lambda k_x: self.masses[0, -1] * omega ** 2 - 4 * self.stiffnesses[0, -1] *
                       (sin(k_x * self.a / 2) ** 2 + sin(k_2_y * self.a / 2) ** 2), np.array([0.5]))[0]
        k_2 = np.sqrt(k_2_x ** 2 + k_2_y ** 2)
        zeta = np.arctan(k_2_y / k_2_x)

        k = Symbol("k")
        g_1 = diff(2 * np.sqrt(self.stiffnesses[0, 0] / self.masses[0, 0]) *
                   sp.sqrt(sp.sin(k * np.cos(gamma) * self.a / 2) ** 2 +
                           sp.sin(k * np.sin(gamma) * self.a / 2) ** 2), k).evalf(subs={k: k_1})
        g_1_x = g_1 * np.cos(gamma)
        g_1_y = g_1 * np.sin(gamma)
        g_2 = diff(2 * np.sqrt(self.stiffnesses[0, -1] / self.masses[0, -1]) *
                   sp.sqrt(sp.sin(k * np.cos(zeta) * self.a / 2) ** 2 +
                           sp.sin(k * np.sin(zeta) * self.a / 2) ** 2), k).evalf(subs={k: k_2})
        g_2_x = g_2 * np.cos(zeta)
        g_2_y = g_2 * np.sin(zeta)

        amp_frac = (exp(I * k_1_x * self.a) - exp(-I * k_1_x * self.a)) / \
                   (exp(I * k_2_x * self.a) - exp(-I * k_1_x * self.a))
        amp_frac = amp_frac.evalf()
        trans_coeff = self.masses[0, -1] * g_2_x / (self.masses[0, 0] * g_1_x) * (Abs(amp_frac)) ** 2

        return trans_coeff

    @property
    def omega_low(self):
        gamma = getattr(self, "gamma")
        c_1, c_2 = self.stiffnesses[0, 0], self.stiffnesses[0, -1]
        m_1, m_2 = self.masses[0, 0], self.masses[0, -1]
        d_1, d_2 = self.foundation_stiffnesses[0, 0], self.foundation_stiffnesses[0, -1]
        arr = [(sin(cos(gamma) * var)) ** 2 + (sin(sin(gamma) * var)) ** 2 for var in np.arange(0, 2 * np.pi, 0.001)]
        return np.sqrt(max((4 * c_1 * min(arr) + d_1) / m_1,
                           (4 * c_2 * min(arr) + d_2) / m_2))

    @property
    def omega_high(self):
        gamma = getattr(self, "gamma")
        c_1, c_2 = self.stiffnesses[0, 0], self.stiffnesses[0, -1]
        m_1, m_2 = self.masses[0, 0], self.masses[0, -1]
        d_1, d_2 = self.foundation_stiffnesses[0, 0], self.foundation_stiffnesses[0, -1]
        arr = [(sin(cos(gamma) * var)) ** 2 + (sin(sin(gamma) * var)) ** 2 for var in np.arange(0, 2 * np.pi, 0.001)]
        return np.sqrt(min((4 * c_1 * max(arr) + d_1) / m_1,
                           (4 * c_2 * max(arr) + d_2) / m_2))

    def plot_field(self, field="disp_undim", title="", x_label="", y_label="", cbar_label=""):
        cur_field = getattr(self, field)
        levels = np.linspace(cur_field.min(), cur_field.max(), 100)
        fig, ax = plt.subplots()
        ax.plot([0] * self.coords_y.shape[0], self.coords_y[:, 0], linestyle="dashed", color="red", linewidth=1)
        cs = ax.contourf(self.coords_x, self.coords_y, cur_field, levels=levels)
        cbar = fig.colorbar(cs, ticks=np.linspace(0, cur_field.max(), 10), label=cbar_label, ax=ax)
        ax = plt.gca()
        plt.title(f"{title} {cbar_label}")
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        ax.set_aspect("equal", adjustable="box")
        plt.show()

    frames_containers = ["time_undim_frames", "disp_undim_frames", "vel_undim_frames", "energy_field_undim_frames",
                         "energy_both_undim_frames", "energy_left_undim_frames",
                         "energy_right_undim_frames", "transmission_coeff_numerical_frames",
                         "transmission_coeff_analytical_frames"]
    frames_container_names = list(map(lambda s: s.replace("_frames", ""), frames_containers))

    def save_history(self, t):
        setattr(self, "time_undim", t * getattr(self, "g_1")[0, 0] / self.a)
        for i, frames_container in enumerate(self.frames_containers):
            if not hasattr(self, frames_container):
                setattr(self, frames_container, [])
            getattr(self, frames_container).append(deepcopy(getattr(self, self.frames_container_names[i])))


if __name__ == "__main__":
    lattice_lattice = LatticeLatticeStructure(m_1=0.5, m_2=1.0,
                                              c_1=0.1, c_2=0.1, c_12=0.1,
                                              d_1=0.0, d_2=0.2,
                                              cnt_x=301, cnt_y=301, a=1)
    lattice_lattice.specify_initial_and_boundary(gamma=np.radians(5), beta_x=0.035, beta_y=0.035,
                                                 n_0=-70, v_0=-35, u_0=1, omega=0.7)
    lattice_lattice.plot_field(field="energy_field_undim", title="Энергия",
                               x_label="n", y_label="m", cbar_label=r"$2e_{n,m} \;/\; \left(m_1U_0^2\Omega^2\right)$")
    lattice_lattice.solve(dt=0.05, t_max=800, save_time=15)
    lattice_lattice.plot_field(field="energy_field_undim", title="Энергия",
                               x_label="n", y_label="m", cbar_label=r"$2e_{n,m} \;/\; \left(m_1U_0^2\Omega^2\right)$")
