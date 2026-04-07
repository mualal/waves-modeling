from chains import ChainChainStructure
from lattices import LatticeLatticeStructure
import numpy as np
import matplotlib.pyplot as plt
from waves_vis_utils import monitor_energy, animate_chains, animate_lattices
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import itertools


def lattice_from_previous_matlab_experiments():
    lattice_lattice = LatticeLatticeStructure(m_1=1.0, m_2=0.5,
                                              c_1=0.75, c_2=0.75, c_12=0.75,
                                              d_1=0, d_2=0,
                                              cnt_x=201, cnt_y=201, a=1.0)

    lattice_lattice.specify_initial_and_boundary(gamma=np.radians(40), beta_x=0.1, beta_y=0.1,
                                                 u_0=1, omega=1)
    # lattice_lattice.plot_field()
    # lattice_lattice.solve(t_max=120, accelerate=True)
    # lattice_lattice.plot_field()
    lattice_lattice.solve(t_max=180, auto_stop=True)
    monitor_energy(lattice_lattice)
    animate_lattices(lattice_lattice)


def gamma_dependence_fast():
    gamma_values = list(range(0, 71, 10))
    # gamma_values = [10, 20]
    lattices = [LatticeLatticeStructure(m_1=0.5, m_2=1.0,
                                        c_1=1, c_2=1, c_12=1,
                                        d_1=0.0, d_2=0.0,
                                        cnt_x=1501, cnt_y=501, a=1.0) for _ in range(len(gamma_values))]
    for i, lattice in enumerate(lattices):
        lattice.specify_initial_and_boundary(gamma=np.radians(gamma_values[i]), beta_x=0.08, beta_y=0.02,
                                             u_0=1, omega_undim=0.5)

    for lattice in lattices:
        lattice.plot_field()

    with ThreadPoolExecutor(max_workers=4) as executor:
        for lattice in lattices:
            executor.submit(lattice.solve, auto_stop=False, accelerate=True)

    # for lattice in lattices:
    #     lattice.solve(auto_stop=False, accelerate=True)

    for lattice in lattices:
        lattice.plot_field()

    fig, ax = plt.subplots()

    ax.plot(gamma_values, [lattice.transmission_coeff_numerical for lattice in lattices],
            label=f"Численно $\\left(\\beta={getattr(lattices[0], 'beta_x')}\\right)$", color="black", alpha=1)

    ax.plot(gamma_values, [lattice.transmission_coeff_analytical for lattice in lattices],
            label="Аналитически", color="red")

    ax.plot(gamma_values, [lattice.transmission_coeff_continuum for lattice in lattices],
            label="Континуальный", color="blue")

    plt.title(f"$\\left(m_1 / m_2 = {round(lattices[0].masses[0, 0] / lattices[0].masses[0, -1], 1)}"
              f";\\,c_1 / c_2 = {round(lattices[0].stiffnesses[0, 0] / lattices[0].stiffnesses[0, -1], 1)}"
              f";\\,\\Omega / \\Omega_{{max}} = {round(getattr(lattices[0], 'omega_undim'), 5)}"
              f";\\,k_1 \\approx {round(getattr(lattices[0], 'k_1'), 3)}"
              # f";\\,g_1 \\approx {round(getattr(lattices[0], 'g_1')[0, 0], 3)}"
              f"\\right)$")
    plt.xlabel("Угол падения, градусов")
    plt.ylabel("Коэффициент прохождения T")
    plt.grid(linewidth=0.5)
    plt.grid(which="minor", linestyle=":", linewidth=0.3)
    plt.minorticks_on()
    plt.legend()
    plt.show()


def gamma_dependence_heavy():
    gamma_values = list(range(0, 71, 10))
    # gamma_values = [40]
    lattices = [LatticeLatticeStructure(m_1=0.5, m_2=1.0,
                                        c_1=1, c_2=1, c_12=1,
                                        d_1=0.0, d_2=0.0,
                                        cnt_x=3001, cnt_y=1001, a=1.0) for _ in range(len(gamma_values))]
    for i, lattice in enumerate(lattices):
        lattice.specify_initial_and_boundary(gamma=np.radians(gamma_values[i]), beta_x=0.01, beta_y=0.01,
                                             u_0=1, omega_undim=0.05)

    for lattice in lattices:
        lattice.plot_field()

    with ThreadPoolExecutor(max_workers=4) as executor:
        for lattice in lattices:
            executor.submit(lattice.solve, auto_stop=False, accelerate=True)

    # for lattice in lattices:
    #     lattice.solve(auto_stop=False, accelerate=True)

    for lattice in lattices:
        lattice.plot_field()

    fig, ax = plt.subplots()

    ax.plot(gamma_values, [lattice.transmission_coeff_numerical for lattice in lattices],
            label=f"Численно $\\left(\\beta={getattr(lattices[0], 'beta_x')}\\right)$", color="black", alpha=1)

    ax.plot(gamma_values, [lattice.transmission_coeff_analytical for lattice in lattices],
            label="Аналитически", color="red")

    ax.plot(gamma_values, [lattice.transmission_coeff_continuum for lattice in lattices],
            label="Континуальный", color="blue")

    plt.title(f"$\\left(m_1 / m_2 = {round(lattices[0].masses[0, 0] / lattices[0].masses[0, -1], 1)}"
              f";\\,c_1 / c_2 = {round(lattices[0].stiffnesses[0, 0] / lattices[0].stiffnesses[0, -1], 1)}"
              f";\\,\\Omega / \\Omega_{{max}} = {round(getattr(lattices[0], 'omega_undim'), 5)}"
              f";\\,k_1 \\approx {round(getattr(lattices[0], 'k_1'), 3)}"
              # f";\\,g_1 \\approx {round(getattr(lattices[0], 'g_1')[0, 0], 3)}"
              f"\\right)$")
    plt.xlabel("Угол падения, градусов")
    plt.ylabel("Коэффициент прохождения T")
    plt.grid(linewidth=0.5)
    plt.grid(which="minor", linestyle=":", linewidth=0.3)
    plt.minorticks_on()
    plt.legend()
    plt.show()


def gamma_beta_dependence(omega_undim):
    gamma_values = list(range(0, 71, 10))
    #gamma_values = [80]

    labels = ['02', '04', '06', '08', '10']
    for label, beta_value in zip(labels, [0.02, 0.04, 0.06, 0.08, 0.10]):
        globals()[f"lattices_{label}"] = [LatticeLatticeStructure(m_1=1.0, m_2=0.5,
                                                                  c_1=1, c_2=1, c_12=1,
                                                                  d_1=0.0, d_2=0.0,
                                                                  cnt_x=3001, cnt_y=1001, a=1.0)
                                          for _ in range(len(gamma_values))]

        for i, lattice in enumerate(globals()[f"lattices_{label}"]):
            lattice.specify_initial_and_boundary(gamma=np.radians(gamma_values[i]), beta_x=beta_value,
                                                 beta_y=beta_value, u_0=1, omega_undim=omega_undim)

    all_lattices_nested = [globals()[f"lattices_{label}"] for label in labels]
    all_lattices = list(itertools.chain.from_iterable(all_lattices_nested))

    for lattice in all_lattices:
        lattice.plot_field()

    with ThreadPoolExecutor(max_workers=4) as executor:
        for lattice in all_lattices:
            executor.submit(lattice.solve, auto_stop=False, accelerate=True)

    # for lattice in lattices:
    #     lattice.solve(auto_stop=False, accelerate=True)

    for lattice in all_lattices:
        lattice.plot_field()
        #monitor_energy(lattice)

    # for lattice in lattices:
    #     animate_lattices(lattice)

    fig, ax = plt.subplots()

    ax.plot(gamma_values, [lattice.transmission_coeff_numerical for lattice in globals()[f"lattices_10"]],
            label="Численно $\\left(\\beta=0.1\\right)$", color="black", alpha=0.2)
    ax.plot(gamma_values, [lattice.transmission_coeff_numerical for lattice in globals()[f"lattices_08"]],
            label="Численно $\\left(\\beta=0.08\\right)$", color="black", alpha=0.4)
    ax.plot(gamma_values, [lattice.transmission_coeff_numerical for lattice in globals()[f"lattices_06"]],
            label="Численно $\\left(\\beta=0.06\\right)$", color="black", alpha=0.6)
    ax.plot(gamma_values, [lattice.transmission_coeff_numerical for lattice in globals()[f"lattices_04"]],
            label="Численно $\\left(\\beta=0.04\\right)$", color="black", alpha=0.8)
    ax.plot(gamma_values, [lattice.transmission_coeff_numerical for lattice in globals()[f"lattices_02"]],
            label="Численно $\\left(\\beta=0.02\\right)$", color="black", alpha=1.0)

    ax.plot(gamma_values, [lattice.transmission_coeff_analytical for lattice in globals()[f"lattices_02"]],
            label="Аналитически", color="red")

    ax.plot(gamma_values, [lattice.transmission_coeff_continuum for lattice in globals()[f"lattices_02"]],
            label="Континуальный", color="blue")

    plt.title(f"$\\left(m_1 / m_2 = {round(all_lattices[0].masses[0, 0] / all_lattices[0].masses[0, -1], 1)}"
                  f";\\,c_1 / c_2 = {round(all_lattices[0].stiffnesses[0, 0] / all_lattices[0].stiffnesses[0, -1], 1)}"
                  f";\\,\\Omega / \\Omega_{{max}} = {round(getattr(all_lattices[0], 'omega_undim'), 3)}"
                  f";\\,k_1 \\approx {round(getattr(all_lattices[0], 'k_1'), 3)}"
                  # f";\\,g_1 \\approx {round(getattr(lattices[0], 'g_1')[0, 0], 3)}"
                  f"\\right)$")
    plt.xlabel("Угол падения, градусов")
    plt.ylabel("Коэффициент прохождения T")
    plt.grid(linewidth=0.5)
    plt.grid(which="minor", linestyle=":", linewidth=0.3)
    plt.minorticks_on()
    plt.legend()
    plt.show()


def gamma_omega_dependence():
    gamma_values = list(range(0, 71, 10))
    latt_0_1 = [LatticeLatticeStructure(m_1=0.5, m_2=1.0,
                                        c_1=1, c_2=1, c_12=1,
                                        d_1=0.0, d_2=0.0,
                                        cnt_x=1501, cnt_y=501, a=1.0) for _ in range(len(gamma_values))]
    latt_0_3, latt_0_5 = deepcopy(latt_0_1), deepcopy(latt_0_1)
    latt_0_7, latt_0_9 = deepcopy(latt_0_1), deepcopy(latt_0_1)
    for freq, latt in zip([0.1, 0.3, 0.5, 0.7, 0.9], [latt_0_1, latt_0_3, latt_0_5, latt_0_7, latt_0_9]):
        for i, lattice in enumerate(latt):
            lattice.specify_initial_and_boundary(gamma=np.radians(gamma_values[i]), beta_x=0.08, beta_y=0.02,
                                                 u_0=1, omega_undim=freq)

    all_latt = [*latt_0_1, *latt_0_3, *latt_0_5, *latt_0_7, *latt_0_9]

    # with ThreadPoolExecutor(max_workers=4) as executor:
    #     for lattice in all_latt:
    #         executor.submit(lattice.solve, auto_stop=False, accelerate=True)

    fig, ax = plt.subplots()

    ax.plot(gamma_values, [lattice.transmission_coeff_numerical for lattice in latt_0_1],
            label=f"Численно $\\left(\\tilde{{\\omega}}={0.1}\\right)$", color="black", alpha=0.2)
    ax.plot(gamma_values, [lattice.transmission_coeff_numerical for lattice in latt_0_3],
            label=f"Численно $\\left(\\tilde{{\\omega}}={0.3}\\right)$", color="black", alpha=0.4)
    ax.plot(gamma_values, [lattice.transmission_coeff_numerical for lattice in latt_0_5],
            label=f"Численно $\\left(\\tilde{{\\omega}}={0.5}\\right)$", color="black", alpha=0.6)
    ax.plot(gamma_values, [lattice.transmission_coeff_numerical for lattice in latt_0_7],
            label=f"Численно $\\left(\\tilde{{\\omega}}={0.7}\\right)$", color="black", alpha=0.8)
    ax.plot(gamma_values, [lattice.transmission_coeff_numerical for lattice in latt_0_9],
            label=f"Численно $\\left(\\tilde{{\\omega}}={0.9}\\right)$", color="black", alpha=1.0)

    ax.plot(gamma_values, [lattice.transmission_coeff_analytical for lattice in latt_0_1],
            label=f"Аналитически $\\left(\\tilde{{\\omega}}={0.1}\\right)$", color="red", alpha=0.2)
    ax.plot(gamma_values, [lattice.transmission_coeff_analytical for lattice in latt_0_3],
            label=f"Аналитически $\\left(\\tilde{{\\omega}}={0.3}\\right)$", color="red", alpha=0.4)
    ax.plot(gamma_values, [lattice.transmission_coeff_analytical for lattice in latt_0_5],
            label=f"Аналитически $\\left(\\tilde{{\\omega}}={0.5}\\right)$", color="red", alpha=0.6)
    ax.plot(gamma_values, [lattice.transmission_coeff_analytical for lattice in latt_0_7],
            label=f"Аналитически $\\left(\\tilde{{\\omega}}={0.7}\\right)$", color="red", alpha=0.8)
    ax.plot(gamma_values, [lattice.transmission_coeff_analytical for lattice in latt_0_9],
            label=f"Аналитически $\\left(\\tilde{{\\omega}}={0.9}\\right)$", color="red", alpha=1.0)

    ax.plot(gamma_values, [lattice.transmission_coeff_continuum for lattice in latt_0_1],
            label="Континуальный", color="blue")

    plt.title(f"$\\left(m_1 / m_2 = {round(latt_0_1[0].masses[0, 0] / latt_0_1[0].masses[0, -1], 1)}"
              f";\\,c_1 / c_2 = {round(latt_0_1[0].stiffnesses[0, 0] / latt_0_1[0].stiffnesses[0, -1], 1)}"
              f"\\right)$")
    plt.xlabel("Угол падения, градусов")
    plt.ylabel("Коэффициент прохождения T")
    plt.grid(linewidth=0.5)
    plt.grid(which="minor", linestyle=":", linewidth=0.3)
    plt.minorticks_on()
    plt.legend()
    plt.show()


def refraction_angles():
    gamma_values = list(range(0, 71, 10))
    # gamma_values = [70]
    lattices = [LatticeLatticeStructure(m_1=0.5, m_2=1.0,
                                        c_1=1, c_2=1, c_12=1,
                                        d_1=0.0, d_2=0.0,
                                        cnt_x=1501, cnt_y=1501, a=1.0) for _ in range(len(gamma_values))]
    for i, lattice in enumerate(lattices):
        lattice.specify_initial_and_boundary(gamma=np.radians(gamma_values[i]), beta_x=0.08, beta_y=0.02,
                                             u_0=1, shift_y=-500, omega_undim=0.5)

    for lattice in lattices:
        lattice.plot_field()

    with ThreadPoolExecutor(max_workers=4) as executor:
        for lattice in lattices:
            executor.submit(lattice.solve, auto_stop=False, accelerate=True)

    # for lattice in lattices:
    #     lattice.solve(auto_stop=False, accelerate=True)

    for lattice in lattices:
        lattice.plot_field()

    fig, ax = plt.subplots()

    ax.plot(gamma_values, [np.degrees(lattice.theta_numerically) for lattice in lattices],
            label=f"Численно", color="black", alpha=1)

    ax.plot(gamma_values, [np.degrees(lattice.theta[0])for lattice in lattices],
            label="Аналитически", color="red")

    plt.title(f"$\\left(m_1 / m_2 = {round(lattices[0].masses[0, 0] / lattices[0].masses[0, -1], 1)}"
              f";\\,c_1 / c_2 = {round(lattices[0].stiffnesses[0, 0] / lattices[0].stiffnesses[0, -1], 1)}"
              f";\\,\\Omega / \\Omega_{{max}} = {round(getattr(lattices[0], 'omega_undim'), 5)}"
              f";\\,k_1 \\approx {round(getattr(lattices[0], 'k_1'), 3)}"
              # f";\\,g_1 \\approx {round(getattr(lattices[0], 'g_1')[0, 0], 3)}"
              f"\\right)$")
    plt.xlabel("Угол падения, градусов")
    plt.ylabel("Угол преломления, градусов")
    plt.grid(linewidth=0.5)
    plt.grid(which="minor", linestyle=":", linewidth=0.3)
    plt.minorticks_on()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # lattice_from_previous_matlab_experiments()
    gamma_dependence_fast()
    # gamma_dependence_heavy()
    # gamma_beta_dependence(0.9)
    # gamma_omega_dependence()
    # refraction_angles()
