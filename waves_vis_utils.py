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


def lattices_animation(structure: LatticeLatticeStructure, field="disp_undim"):
    plt.rcParams["animation.html"] = "jshtml"
    plt.rcParams['figure.dpi'] = 150
    plt.ioff()

    cur_field_frames = getattr(structure, field + "_frames")
    fig, ax = plt.subplots()
    levels = np.linspace(cur_field_frames[0].min(), cur_field_frames[0].max(), 100)
    ax.plot([0] * structure.coords_y.shape[0], structure.coords_y[:, 0], linestyle="dashed", color="red", linewidth=1)
    cf = ax.contourf(structure.coords_x, structure.coords_y, cur_field_frames[0], levels=levels)
    cbar = fig.colorbar(cf)
    plt.xlabel('n')
    plt.ylabel('m')
    ax.set_aspect('equal', adjustable='box')

    def update(frame):
        cf = ax.contourf(structure.coords_x, structure.coords_y, cur_field_frames[frame], levels=levels)
        return cf,

    anim = animation.FuncAnimation(fig, update, frames=len(cur_field_frames))
    plt.show()


def chains_animation(structure: ChainChainStructure, field="disp_undim"):
    plt.rcParams["animation.html"] = "jshtml"
    plt.rcParams['figure.dpi'] = 150
    plt.ioff()

    cur_field_frames = getattr(structure, field + "_frames")
    fig, ax = plt.subplots()
    line1 = ax.plot(structure.coords, cur_field_frames[0])[0]
    plt.title("")
    plt.xlabel("")
    plt.ylabel("")
    plt.grid()

    def update(frame):
        line1.set_xdata(structure.coords)
        line1.set_ydata(cur_field_frames[frame])
        return line1,

    anim = animation.FuncAnimation(fig, update, frames=len(cur_field_frames))
    plt.show()
