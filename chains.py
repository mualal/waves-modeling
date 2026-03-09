import numpy as np
from tqdm import tqdm
from copy import deepcopy
import matplotlib.pyplot as plt
from sympy import symbols, Abs, I, exp


class ChainChainStructure:
    def __init__(self, m_1, m_2, c_1, c_2, c_12, d_1, d_2, cnt, a):
        if cnt % 2 == 0:
            raise ValueError("Количество частиц должно быть нечётным")
        self.a = a

        self.indices = np.arange(-(cnt // 2), cnt // 2 + 1, 1)
        self.coords = a * self.indices
        self.masses = m_1 * (self.indices < 0) + m_2 * (self.indices >= 0)
        self.stiffnesses = c_1 * (self.indices < -1) + c_12 * (self.indices == -1) + c_2 * (self.indices > -1)
        self.foundation_stiffnesses = d_1 * (self.indices < 0) + d_2 * (self.indices >= 0)

        self.disp = np.zeros(cnt)
        self.vel = np.zeros(cnt)

    def specify_initial_and_boundary(self, beta, n_0, u_0, omega):
        for i in (0, -1):
            omega_min = np.sqrt(self.foundation_stiffnesses[i] / self.masses[i])
            omega_max = np.sqrt((4 * self.stiffnesses[i] + self.foundation_stiffnesses[i]) / self.masses[i])

            if not omega_min < omega < omega_max:
                label = "левой" if i == 0 else "правой"
                raise ValueError(f"Не выполнены условия: {omega_min} < {omega} < {omega_max} для {label} цепочки")
            print("Выполнено условие: ", f"{omega_min} < {omega} < {omega_max}")

        setattr(self, "omega", omega)
        setattr(self, "u_0", u_0)

        expr = (omega ** 2 - self.foundation_stiffnesses / self.masses) / (4 * self.stiffnesses)
        k_1 = np.arcsin(np.sqrt(self.masses * expr)) * 2 / self.a
        expr = ((4 * self.stiffnesses + self.foundation_stiffnesses) / self.masses - omega ** 2)
        g_1 = self.a / (2 * omega) * np.sqrt((omega ** 2 - self.foundation_stiffnesses / self.masses) * expr)

        self.disp = u_0 * np.exp(-beta ** 2 / 2 * (self.coords - n_0) ** 2) * np.sin(self.coords * k_1)
        self.vel = -u_0 * np.exp(-beta ** 2 / 2 * (self.coords - n_0) ** 2)
        self.vel *= (omega * np.cos(k_1 * self.coords) -
                     beta ** 2 * g_1 / self.a * (self.coords - n_0) * np.sin(self.coords * k_1))
        self.disp[np.where(self.indices >= -1)] = 0
        self.vel[np.where(self.indices >= -1)] = 0

    def solve(self, dt, t_max, save_time):
        time_steps = np.arange(0, t_max, dt)
        for t in tqdm(time_steps):
            # leapfrog synchronized form
            acc1 = (self.stiffnesses / self.masses) * (np.roll(self.disp, -1) - self.disp) + \
                (np.roll(self.stiffnesses, 1) / self.masses) * (np.roll(self.disp, 1) - self.disp) - \
                self.foundation_stiffnesses / self.masses * self.disp
            self.disp += self.vel * dt + 1 / 2 * acc1 * dt ** 2
            acc2 = (self.stiffnesses / self.masses) * (np.roll(self.disp, -1) - self.disp) + \
                (np.roll(self.stiffnesses, 1) / self.masses) * (np.roll(self.disp, 1) - self.disp) - \
                self.foundation_stiffnesses / self.masses * self.disp
            self.vel += 1 / 2 * (acc1 + acc2) * dt

            # save results
            if t % save_time == 0:
                self.save_history()

    @property
    def energy_field(self):
        e = self.masses / 2 * self.vel ** 2 + self.stiffnesses / 4 * (np.roll(self.disp, -1) - self.disp) ** 2 + \
            np.roll(self.stiffnesses, 1) / 4 * (np.roll(self.disp, 1) - self.disp) ** 2 + \
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
        m_1 = self.masses[0]
        u_0 = getattr(self, "u_0")
        omega = getattr(self, "omega")
        return 2 * self.energy_field / (m_1 * u_0 ** 2 * omega ** 2)

    @property
    def energy_both_undim(self):
        return np.sum(self.energy_field_undim)

    @property
    def energy_left_undim(self):
        return np.sum(self.energy_field_undim * (self.indices < 0))

    @property
    def energy_right_undim(self):
        return np.sum(self.energy_field_undim * (self.indices >= 0))

    @property
    def transmission_coeff_numerical(self):
        return self.energy_right_undim / self.energy_both_undim

    @property
    def transmission_coeff_analytical(self):
        m_1, m_2 = self.masses[0], self.masses[-1]
        c_1, c_2 = self.stiffnesses[0], self.stiffnesses[-1]
        d_1, d_2 = self.foundation_stiffnesses[0], self.foundation_stiffnesses[-1]
        omega = getattr(self, "omega")

        # c12 = symbols("c12")
        c12 = self.stiffnesses[np.where(self.indices == -1)][0]

        k_1 = 2 / self.a * np.arcsin(np.sqrt((m_1 * omega ** 2 - d_1) / (4 * c_1)))
        k_2 = 2 / self.a * np.arcsin(np.sqrt((m_2 * omega ** 2 - d_2) / (4 * c_2)))

        g_1 = self.a / (2 * omega) * np.sqrt((omega ** 2 - d_1 / m_1) * ((4 * c_1 + d_1) / m_1 - omega ** 2))
        g_2 = self.a / (2 * omega) * np.sqrt((omega ** 2 - d_2 / m_2) * ((4 * c_2 + d_2) / m_2 - omega ** 2))

        amp_frac = (2 * I * c12 * np.sin(k_1 * self.a)) / \
                   (c12 * (1 - exp(-I * k_1 * self.a)) +
                    c_2 * (exp(I * k_2 * self.a) - 1) * (1 + exp(-I * k_1 * self.a) * (c12 - c_1) / c_1))
        amp_frac = amp_frac.evalf()

        trans_coeff = m_2 * g_2 / (m_1 * g_1) * (Abs(amp_frac)) ** 2

        return trans_coeff

    def plot_field(self, field="disp_undim", title="", x_label="", y_label=""):
        plt.plot(self.coords, getattr(self, field))
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid()
        plt.show()

    frames_containers = ["disp_undim_frames", "vel_undim_frames", "energy_field_undim_frames",
                         "energy_both_undim_frames", "energy_left_undim_frames",
                         "energy_right_undim_frames", "transmission_coeff_numerical_frames",
                         "transmission_coeff_analytical_frames"]
    frames_container_names = list(map(lambda s: s.replace("_frames", ""), frames_containers))

    def save_history(self):
        for i, frames_container in enumerate(self.frames_containers):
            if not hasattr(self, frames_container):
                setattr(self, frames_container, [])
            getattr(self, frames_container).append(deepcopy(getattr(self, self.frames_container_names[i])))


if __name__ == "__main__":
    chain_chain = ChainChainStructure(m_1=0.5, m_2=1.0,
                                      c_1=0.1, c_2=0.1, c_12=0.1,
                                      d_1=0.0, d_2=0.2,
                                      cnt=301, a=1)
    chain_chain.specify_initial_and_boundary(beta=0.035, n_0=-70, u_0=1, omega=np.sqrt(0.4))
    chain_chain.plot_field(field="energy_field_undim", title="Энергия",
                           x_label=r"$n$", y_label=r"$2e_n \;/\; \left(m_1U_0^2\Omega^2\right)$")
    chain_chain.solve(dt=0.05, t_max=400, save_time=15)
    chain_chain.plot_field(field="energy_field_undim", title="Энергия",
                           x_label=r"$n$", y_label=r"$2e_n \;/\; \left(m_1U_0^2\Omega^2\right)$")
