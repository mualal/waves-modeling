import numpy as np
import numba as nb
from numba_progress import ProgressBar
from tqdm import tqdm
from copy import deepcopy
import matplotlib.pyplot as plt
from sympy import symbols, Abs, I, exp


class ChainChainStructure:
    def __init__(self, m1, m2, c1, c2, c12, d1, d2, cnt, a):
        if cnt % 2 == 0:
            raise ValueError("Количество частиц должно быть нечётным")
        self.a = a

        self.indices = np.arange(-(cnt // 2), cnt // 2 + 1, 1)
        self.coords = a * self.indices
        self.masses = m1 * (self.indices < 0) + m2 * (self.indices >= 0)
        self.stiffnesses = c1 * (self.indices < -1) + c12 * (self.indices == -1) + c2 * (self.indices > -1)
        self.foundation_stiffnesses = d1 * (self.indices < 0) + d2 * (self.indices >= 0)

        self.disp = np.zeros(cnt)
        self.vel = np.zeros(cnt)

        self.ps = {}

    def specify_initial_and_boundary(self, beta, u0, n0=None, omega=None, omega_undim=None):
        if omega_undim is not None:
            omega = np.sqrt(self.omega_low ** 2 + omega_undim ** 2 * (self.omega_high ** 2 - self.omega_low ** 2))
        if n0 is None:
            n0 = -3 / beta
        for i in (0, -1):
            omega_min = np.sqrt(self.foundation_stiffnesses[i] / self.masses[i])
            omega_max = np.sqrt((4 * self.stiffnesses[i] + self.foundation_stiffnesses[i]) / self.masses[i])

            if not omega_min < omega < omega_max:
                label = "левой" if i == 0 else "правой"
                raise ValueError(f"Не выполнены условия: {omega_min} < {omega} < {omega_max} для {label} цепочки")
            print("Выполнено условие: ", f"{omega_min} < {omega} < {omega_max}")

        self.ps.update(omega=omega, u0=u0, beta=beta, n0=n0)

        expr = (omega ** 2 - self.foundation_stiffnesses / self.masses) / (4 * self.stiffnesses)
        k1 = np.arcsin(np.sqrt(self.masses * expr)) * 2 / self.a
        expr = ((4 * self.stiffnesses + self.foundation_stiffnesses) / self.masses - omega ** 2)
        g1 = self.a / (2 * omega) * np.sqrt((omega ** 2 - self.foundation_stiffnesses / self.masses) * expr)

        self.ps.update(g1=g1)

        self.disp = u0 * np.exp(-beta ** 2 / 2 * (self.coords - n0) ** 2) * np.sin(self.coords * k1)
        self.vel = -u0 * np.exp(-beta ** 2 / 2 * (self.coords - n0) ** 2)
        self.vel *= (omega * np.cos(k1 * self.coords) -
                     beta ** 2 * g1 / self.a * (self.coords - n0) * np.sin(self.coords * k1))
        self.disp[np.where(self.indices >= -1)] = 0
        self.vel[np.where(self.indices >= -1)] = 0

    def solve(self, dt=None, t_max=None, save_time=None, auto_stop=True, accelerate=False):
        if dt is None:
            # dt = 0.05 / self.omega_high
            dt = 0.05
        if t_max is None:
            t_max = 3 * abs(self.ps["n0"]) * self.a / self.ps["g1"][0]
        if save_time is None:
            save_time = 15

        time_steps = np.arange(0, t_max, dt)
        if accelerate:
            with ProgressBar(total=len(time_steps)) as progress:
                self.disp, self.vel = numba_accelerate(dt, time_steps, self.masses, self.disp, self.vel,
                                                       self.stiffnesses, self.foundation_stiffnesses, progress)
        else:
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
                    self.save_history(t)

                # autostop
                if auto_stop:
                    interface_energy = getattr(self, "energy_interface_undim_frames", None)
                    if interface_energy and interface_energy[-1] < max(interface_energy) / 1e3:
                        break

    @property
    def energy_field(self):
        e = self.masses / 2 * self.vel ** 2 + self.stiffnesses / 4 * (np.roll(self.disp, -1) - self.disp) ** 2 + \
            np.roll(self.stiffnesses, 1) / 4 * (np.roll(self.disp, 1) - self.disp) ** 2 + \
            self.foundation_stiffnesses / 2 * self.disp ** 2
        return e

    @property
    def disp_undim(self):
        return self.disp / self.ps["u0"]

    @property
    def vel_undim(self):
        return self.vel / (self.a * self.ps["omega"])

    @property
    def energy_field_undim(self):
        m1 = self.masses[0]
        return 2 * self.energy_field / (m1 * self.ps["u0"] ** 2 * self.ps["omega"] ** 2)

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
    def energy_interface_undim(self):
        return np.sum(self.energy_field_undim * (self.indices == 0))

    @property
    def transmission_coeff_numerical(self):
        return self.energy_right_undim / self.energy_both_undim

    @property
    def transmission_coeff_analytical(self):
        m1, m2 = self.masses[0], self.masses[-1]
        c1, c2 = self.stiffnesses[0], self.stiffnesses[-1]
        d1, d2 = self.foundation_stiffnesses[0], self.foundation_stiffnesses[-1]
        omega = self.ps["omega"]

        # c12 = symbols("c12")
        c12 = self.stiffnesses[np.where(self.indices == -1)][0]

        k1 = 2 / self.a * np.arcsin(np.sqrt((m1 * omega ** 2 - d1) / (4 * c1)))
        k2 = 2 / self.a * np.arcsin(np.sqrt((m2 * omega ** 2 - d2) / (4 * c2)))

        g1 = self.a / (2 * omega) * np.sqrt((omega ** 2 - d1 / m1) * ((4 * c1 + d1) / m1 - omega ** 2))
        g2 = self.a / (2 * omega) * np.sqrt((omega ** 2 - d2 / m2) * ((4 * c2 + d2) / m2 - omega ** 2))

        amp_frac = (2 * I * c12 * np.sin(k1 * self.a)) / \
                   (c12 * (1 - exp(-I * k1 * self.a)) +
                    c2 * (exp(I * k2 * self.a) - 1) * (1 + exp(-I * k1 * self.a) * (c12 - c1) / c1))
        amp_frac = amp_frac.evalf()

        trans_coeff = m2 * g2 / (m1 * g1) * (Abs(amp_frac)) ** 2

        return trans_coeff

    @property
    def omega_low(self):
        return np.sqrt(max(self.foundation_stiffnesses[0] / self.masses[0],
                           self.foundation_stiffnesses[-1] / self.masses[-1]))

    @property
    def omega_high(self):
        return np.sqrt(min((4 * self.stiffnesses[0] + self.foundation_stiffnesses[0]) / self.masses[0],
                           (4 * self.stiffnesses[-1] + self.foundation_stiffnesses[-1]) / self.masses[-1]))

    def plot_field(self, field="energy_field_undim", title="Энергия",
                   x_label=r"$n$", y_label=r"$2e_n \;/\; \left(m_1U_0^2\Omega^2\right)$"):
        plt.plot(self.coords, getattr(self, field))
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid(linewidth=0.5)
        plt.grid(which="minor", linestyle=":", linewidth=0.3)
        plt.minorticks_on()
        plt.show()

    frames_containers = ["time_undim_frames", "disp_undim_frames", "vel_undim_frames", "energy_field_undim_frames",
                         "energy_both_undim_frames", "energy_left_undim_frames",
                         "energy_right_undim_frames", "energy_interface_undim_frames",
                         "transmission_coeff_numerical_frames",
                         "transmission_coeff_analytical_frames"]
    frames_container_names = list(map(lambda s: s.replace("_frames", ""), frames_containers))

    def save_history(self, t):
        setattr(self, "time_undim", t * self.ps["g1"][0] / (abs(self.ps["n0"]) * self.a))
        for i, frames_container in enumerate(self.frames_containers):
            if not hasattr(self, frames_container):
                setattr(self, frames_container, [])
            getattr(self, frames_container).append(deepcopy(getattr(self, self.frames_container_names[i])))


@nb.jit(nopython=True, nogil=True)
def numba_accelerate(dt, time_steps, masses, disp, vel,
                     stiffnesses, foundation_stiffnesses, progress_proxy):
    for t in time_steps:
        # leapfrog synchronized form
        acc1 = (stiffnesses / masses) * (np.roll(disp, -1) - disp) + \
               (np.roll(stiffnesses, 1) / masses) * (np.roll(disp, 1) - disp) - \
               foundation_stiffnesses / masses * disp
        disp += vel * dt + 1 / 2 * acc1 * dt ** 2
        acc2 = (stiffnesses / masses) * (np.roll(disp, -1) - disp) + \
               (np.roll(stiffnesses, 1) / masses) * (np.roll(disp, 1) - disp) - \
            foundation_stiffnesses / masses * disp
        vel += 1 / 2 * (acc1 + acc2) * dt
        progress_proxy.update(1)
    return disp, vel


if __name__ == "__main__":
    chain_chain = ChainChainStructure(m1=0.5, m2=1.0,
                                      c1=0.1, c2=0.1, c12=0.1,
                                      d1=0.0, d2=0.2,
                                      cnt=601, a=1)
    chain_chain.specify_initial_and_boundary(beta=0.02, u0=1, omega_undim=np.sqrt(0.5))
    chain_chain.plot_field()
    chain_chain.solve(auto_stop=False, accelerate=True)
    chain_chain.plot_field()
