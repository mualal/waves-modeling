from chains import ChainChainStructure
from lattices import LatticeLatticeStructure
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


def monitor_energy(structure: ChainChainStructure | LatticeLatticeStructure):
    plt.plot(getattr(structure, "time_undim_frames"), getattr(structure, "energy_both_undim_frames"),
             label="Полная энергия")
    plt.plot(getattr(structure, "time_undim_frames"), getattr(structure, "energy_left_undim_frames"),
             label="В левой решётке")
    plt.plot(getattr(structure, "time_undim_frames"), getattr(structure, "energy_right_undim_frames"),
             label="В правой решётке")
    plt.xlabel(r"$g_1t/a$")
    plt.ylabel(r"$\sum_{n}{2e_n \,/\, \left(m_1U_0^2\Omega^2\right)}$")
    plt.title("Энергия в системе")
    plt.grid()
    plt.legend()
    plt.show()


def lattices_animation(structure: LatticeLatticeStructure,
                       field="energy_field_undim",
                       title="Энергия",
                       x_label="n",
                       y_label="m",
                       cbar_label=r"$2e_{n,m} \;/\; \left(m_1U_0^2\Omega^2\right)$"):
    plt.rcParams["animation.html"] = "jshtml"
    plt.rcParams['figure.dpi'] = 150
    plt.ioff()

    cur_field_frames = getattr(structure, field + "_frames")
    fig, ax = plt.subplots()
    levels = np.linspace(cur_field_frames[0].min(), cur_field_frames[0].max(), 100)
    #levels = np.linspace(0, 0.1, 100)
    ax.plot([0] * structure.coords_y.shape[0], structure.coords_y[:, 0], linestyle="dashed", color="red", linewidth=1)
    cf = ax.contourf(structure.coords_x, structure.coords_y, cur_field_frames[0], levels=levels)
    cbar = fig.colorbar(cf, ticks=np.linspace(0, cur_field_frames[0].max(), 10), label=cbar_label, ax=ax)
    plt.title(f"{title} {cbar_label}")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    ax.set_aspect('equal', adjustable='box')

    # ax.plot(structure.coords_x[0][np.where(structure.coords_x[0] >= 0)],
    #        np.add(np.tan(structure.zeta[0]) * structure.coords_x[0][np.where(structure.coords_x[0] >= 0)], -20),
    #        linestyle="dashed", color="orange", linewidth=1)

    def update(frame):
        nonlocal cbar
        cbar.remove()
        ax.clear()
        ax.plot([0] * structure.coords_y.shape[0], structure.coords_y[:, 0], linestyle="dashed", color="red",
                linewidth=1)
        plt.title(f"{title} {cbar_label}")
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        levels = np.linspace(cur_field_frames[frame].min(), cur_field_frames[frame].max(), 100)
        cf = ax.contourf(structure.coords_x, structure.coords_y, cur_field_frames[frame], levels=levels)
        cbar = fig.colorbar(cf, ticks=np.linspace(0, cur_field_frames[frame].max(), 10), label=cbar_label, ax=ax)
        return cf,

    anim = animation.FuncAnimation(fig, update, frames=len(cur_field_frames))
    plt.show()


def chains_animation(structure: ChainChainStructure,
                     field="energy_field_undim",
                     title="Энергия",
                     x_label=r"$n$",
                     y_label=r"$2e_n \;/\; \left(m_1U_0^2\Omega^2\right)$"):
    plt.rcParams["animation.html"] = "jshtml"
    plt.rcParams['figure.dpi'] = 150
    plt.ioff()

    cur_field_frames = getattr(structure, field + "_frames")
    fig, ax = plt.subplots()
    line1 = ax.plot(structure.coords, cur_field_frames[0])[0]
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid()

    def update(frame):
        ax.set_ylim((0.9 * min(cur_field_frames[frame]), 1.1 * max(cur_field_frames[frame])))
        line1.set_xdata(structure.coords)
        line1.set_ydata(cur_field_frames[frame])
        return line1,

    anim = animation.FuncAnimation(fig, update, frames=len(cur_field_frames))
    plt.show()
