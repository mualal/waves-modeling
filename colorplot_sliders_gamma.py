import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def get_trans(a, m1, m2, c1, c2, c12, d1, d2):

    cnt = 1000

    omega = np.tile(np.linspace(np.sqrt(d1 / m1), np.sqrt((8 * c1 + d1) / m1), cnt), (cnt, 1))
    k_y = np.tile(np.linspace(0, np.pi / a, cnt), (cnt, 1)).T

    k1_x = 2 / a * np.arcsin(np.sqrt((m1 * omega ** 2 - d1) / (4 * c1) - (np.sin(k_y * a / 2)) ** 2))

    gamma = np.arctan(k_y / k1_x) * 180 / np.pi

    k2_x = 2 / a * np.arcsin(np.sqrt((m2 * omega ** 2 - d2) / (4 * c2) - (np.sin(k_y * a / 2)) ** 2))

    theta = np.arctan(k_y / k2_x) * 180 / np.pi

    g1_x = (2 * a * c1 * np.sin(k1_x * a / 2) * np.cos(k1_x * a / 2)) / \
           (m1 * np.sqrt(4 * c1 / m1 * ((np.sin(k1_x * a / 2)) ** 2 + (np.sin(k_y * a / 2)) ** 2) + d1 / m1))
    g2_x = (2 * a * c2 * np.sin(k2_x * a / 2) * np.cos(k2_x * a / 2)) / \
           (m2 * np.sqrt(4 * c2 / m2 * ((np.sin(k2_x * a / 2)) ** 2 + (np.sin(k_y * a / 2)) ** 2) + d2 / m2))

    amp_frac = (2 * 1j * c12 * np.sin(k1_x * a)) / \
               (c12 * (1 - np.exp(-1j * k1_x * a)) +
                c2 * (np.exp(1j * k2_x * a) - 1) * (1 + np.exp(-1j * k1_x * a) * (c12 - c1) / c1))

    # amp_frac = c1 * (np.exp(-1j * k1_x * a) - np.exp(1j * k1_x * a)) / \
    #            (c2 * np.exp(-1j * k2_x * a) - c1 * np.exp(1j * k1_x * a) + c1 - c2)

    trans_coeff = ((m2 * g2_x) / (m1 * g1_x)) * (np.abs(amp_frac)) ** 2

    gamma_mask = np.where(np.logical_not(np.isnan(gamma)), 1, gamma)
    trans_coeff_mask = np.where(np.logical_not(np.isnan(trans_coeff)), 1, trans_coeff)

    return k_y, omega, gamma, theta, trans_coeff, gamma_mask, trans_coeff_mask


if __name__ == '__main__':

    a = 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7.7))

    ax_m1 = plt.axes([0.02, 0, 0.09, 0.1])
    ax_m2 = plt.axes([0.16, 0, 0.09, 0.1])
    ax_c1 = plt.axes([0.30, 0, 0.09, 0.1])
    ax_c2 = plt.axes([0.44, 0, 0.09, 0.1])
    ax_c12 = plt.axes([0.58, 0, 0.09, 0.1])
    ax_d1 = plt.axes([0.72, 0, 0.09, 0.1])
    ax_d2 = plt.axes([0.86, 0, 0.09, 0.1])
    slider_m1 = Slider(ax_m1, "m1", 0.01, 1, valinit=0.1, valstep=0.01)
    slider_m2 = Slider(ax_m2, "m2", 0.01, 1, valinit=0.2, valstep=0.01)
    slider_c1 = Slider(ax_c1, "c1", 0.01, 1, valinit=0.1, valstep=0.01)
    slider_c2 = Slider(ax_c2, "c2", 0.01, 1, valinit=0.1, valstep=0.01)
    slider_c12 = Slider(ax_c12, "c12", 0.01, 5, valinit=0.1, valstep=0.01)
    slider_d1 = Slider(ax_d1, "d1", 0, 1, valinit=0, valstep=0.01)
    slider_d2 = Slider(ax_d2, "d2", 0, 1, valinit=0, valstep=0.01)

    k_y, omega, gamma, theta, trans_coeff, gamma_mask, trans_coeff_mask = get_trans(a=a, m1=slider_m1.val, m2=slider_m2.val,
                                                                                    c1=slider_c1.val, c2=slider_c2.val,
                                                                                    c12=slider_c12.val,
                                                                                    d1=slider_d1.val, d2=slider_d2.val)

    contourf1 = ax1.contour(k_y * a, omega, gamma, levels=np.linspace(0, 90, 19), cmap='coolwarm')
    contourf2_mask = ax1.contourf(k_y * a, omega, trans_coeff_mask, levels=0, cmap="inferno", alpha=0.07)
    contourf2 = ax2.contourf(gamma, omega, trans_coeff, levels=np.linspace(0, 1, 45), cmap="inferno")  # inferno
    contourf1_mask = ax2.contourf(k_y * a * 28.9, omega, gamma_mask, levels=0, cmap="inferno", alpha=0.00)  # inferno
    plt.colorbar(contourf1, ax=ax1, ticks=np.linspace(0, 90, 10))
    plt.colorbar(contourf2, ax=ax2, ticks=np.linspace(0, 1, 11))

    manual_locations = []
    for idx, level_path in enumerate(contourf1.get_paths()):
        if idx in [1, 6, 12, 17]:
            vertices = level_path.vertices
            if len(vertices) > 0:
                center = (np.median(vertices[:, 0]), np.median(vertices[:, 1]))
                manual_locations.append(center)

    plt.clabel(contourf1, manual=manual_locations, inline=1, fontsize=10)

    ax1.set_xlabel(r"$k_y\,a$")
    ax1.set_ylabel(r"$\Omega$")
    ax1.set_title("Угол падения, градусов")
    ax1.grid(linewidth=0.5)
    ax1.grid(which="minor", linestyle=":", linewidth=0.3)
    ax1.minorticks_on()
    ax2.set_xlabel(r"Угол падения, градусов")
    ax2.set_ylabel(r"$\Omega$")
    ax2.set_title("Коэффициент прохождения T")
    ax2.grid(linewidth=0.5)
    ax2.grid(which="minor", linestyle=":", linewidth=0.3)
    ax2.minorticks_on()
    plt.subplots_adjust(left=0.07, right=0.98, bottom=0.15, top=0.9, wspace=0.15, hspace=0.15)

    def update(val):
        ax1.clear()
        ax2.clear()

        k_y, omega, gamma, theta, trans_coeff, gamma_mask, trans_coeff_mask = get_trans(a=a, m1=slider_m1.val,
                                                                                        m2=slider_m2.val,
                                                                                        c1=slider_c1.val, c2=slider_c2.val,
                                                                                        c12=slider_c12.val,
                                                                                        d1=slider_d1.val, d2=slider_d2.val)

        contourf1 = ax1.contour(k_y * a, omega, gamma, levels=np.linspace(0, 90, 19), cmap='coolwarm')
        contourf2_mask = ax1.contourf(k_y * a, omega, trans_coeff_mask, levels=0, cmap="inferno", alpha=0.07)
        contourf2 = ax2.contourf(gamma, omega, trans_coeff, levels=np.linspace(0, 1, 45), cmap="inferno")  # inferno
        contourf1_mask = ax2.contourf(k_y * a * 28.9, omega, gamma_mask, levels=0, cmap="inferno", alpha=0.00)  # inferno

        manual_locations = []
        for idx, level_path in enumerate(contourf1.get_paths()):
            if idx in [1, 6, 12, 17]:
                vertices = level_path.vertices
                if len(vertices) > 0:
                    center = (np.median(vertices[:, 0]), np.median(vertices[:, 1]))
                    manual_locations.append(center)

        plt.clabel(contourf1, manual=manual_locations, inline=1, fontsize=10)

        ax1.set_xlabel(r"$k_y\,a$")
        ax1.set_ylabel(r"$\Omega$")
        ax1.set_title("Угол падения, градусов")
        ax1.grid(linewidth=0.5)
        ax1.grid(which="minor", linestyle=":", linewidth=0.3)
        ax1.minorticks_on()
        ax2.set_xlabel(r"Угол падения, градусов")
        ax2.set_ylabel(r"$\Omega$")
        ax2.set_title("Коэффициент прохождения T")
        ax2.grid(linewidth=0.5)
        ax2.grid(which="minor", linestyle=":", linewidth=0.3)
        ax2.minorticks_on()

        # contourf2 = ax2.contourf(k_y * a, omega, trans_coeff, levels=np.linspace(0, 1, 45), cmap="inferno")  # inferno
        # contourf1_mask = ax2.contourf(k_y * a, omega, gamma_mask, levels=0, cmap="inferno", alpha=0.05)  # inferno

    for slider in [slider_m1, slider_m2, slider_c1, slider_c2, slider_c12, slider_d1, slider_d2]:
        slider.on_changed(update)

    plt.show()
