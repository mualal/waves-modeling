from lattices import LatticeLatticeStructure
import numpy as np
import matplotlib.pyplot as plt
from waves_vis_utils import monitor_energy, chains_animation, lattices_animation


if __name__ == "__main__":
    gamma_values = list([30])

    lattices = [LatticeLatticeStructure(m_1=0.5, m_2=1.0,
                                        c_1=0.1, c_2=0.1, c_12=0.1,
                                        d_1=0.0, d_2=0.2,
                                        cnt_x=301, cnt_y=301, a=1) for _ in range(len(gamma_values))]

    for i, lattice in enumerate(lattices):
        lattice.specify_initial_and_boundary(gamma=np.radians(gamma_values[i]), beta_x=0.035, beta_y=0.035,
                                             shift_x=-50, shift_y=-75, u_0=1, omega_undim=np.sqrt(0.5))

    for lattice in lattices:
        lattice.solve(dt=0.05, t_max=800, save_time=15)

    for lattice in lattices:
        lattice.plot_field()
        monitor_energy(lattice)

    # for lattice in lattices:
    #    lattices_animation(lattice)

    fig, ax = plt.subplots()
    ax.plot(gamma_values, [lattice.transmission_coeff_numerical for lattice in lattices], label="Численно")
    ax.plot(gamma_values, [lattice.transmission_coeff_analytical for lattice in lattices], label="Аналитически")
    plt.title("Коэффициент прохождения T")
    plt.xlabel("Угол падения, градусов")
    plt.ylabel("T")
    plt.grid()
    plt.legend()
    plt.show()
