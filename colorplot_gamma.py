import numpy as np
import matplotlib.pyplot as plt

a = 1

m1 = 0.3
m2 = 0.2
c1 = 0.1
c2 = 0.1
d1 = 0.2
d2 = 0

cnt = 1000

# omega_low = np.sqrt(d1 / m1)
# omega_high = np.sqrt((8 * c1 + d1) / m1)

omega = np.tile(np.linspace(np.sqrt(d1 / m1), np.sqrt((8 * c1 + d1) / m1), cnt), (cnt, 1))
# omega_undim_pow2 = (omega ** 2 - omega_low ** 2) / (omega_high ** 2 - omega_low ** 2)

k_y = np.tile(np.linspace(0, np.pi / a, cnt), (cnt, 1)).T

# omega = 2
# k_y = 1.6503

k1_x = 2 / a * np.arcsin(np.sqrt((m1 * omega ** 2 - d1) / (4 * c1) - (np.sin(k_y * a / 2)) ** 2))

gamma = np.arctan(k_y / k1_x) * 180 / np.pi

k2_x = 2 / a * np.arcsin(np.sqrt((m2 * omega ** 2 - d2) / (4 * c2) - (np.sin(k_y * a / 2)) ** 2))

theta = np.arctan(k_y / k2_x) * 180 / np.pi

g1_x = (2 * a * c1 * np.sin(k1_x * a / 2) * np.cos(k1_x * a / 2)) / \
       (m1 * np.sqrt(4 * c1 / m1 * ((np.sin(k1_x * a / 2)) ** 2 + (np.sin(k_y * a / 2)) ** 2) + d1 / m1))
g2_x = (2 * a * c2 * np.sin(k2_x * a / 2) * np.cos(k2_x * a / 2)) / \
       (m2 * np.sqrt(4 * c2 / m2 * ((np.sin(k2_x * a / 2)) ** 2 + (np.sin(k_y * a / 2)) ** 2) + d2 / m2))

amp_frac = c1 * (np.exp(-1j * k1_x * a) - np.exp(1j * k1_x * a)) / \
            (c2 * np.exp(-1j * k2_x * a) - c1 * np.exp(1j * k1_x * a) + c1 - c2)

# amp_frac =

trans_coeff = ((m2 * g2_x) / (m1 * g1_x)) * (np.abs(amp_frac)) ** 2

# trans_coeff = (c2 * np.sin(k2_x * a)) / (c1 * np.sin(k1_x * a)) * (np.abs(amp_frac)) ** 2

# trans_coeff = np.sin(k1_x * a) * np.sin(k2_x * a) / (np.sin((k1_x + k2_x) * a / 2)) ** 2

# trans_coeff_2 = 16 * omega ** 2 * m1 * m2 * g1_x * g2_x / \
#               (4 * omega ** 2 * (m1 * g1_x + m2 * g2_x) ** 2 + a ** 2 * ((m1 - m2) * omega ** 2 + d2 - d1) ** 2)

# print(np.nansum(np.abs(trans_coeff - trans_coeff_2)))

gamma_mask = np.where(np.logical_not(np.isnan(gamma)), 1, gamma)
trans_coeff_mask = np.where(np.logical_not(np.isnan(trans_coeff)), 1, trans_coeff)

print(np.nanmax(gamma))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
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
plt.subplots_adjust(left=0.07, right=0.98, bottom=0.1, top=0.9, wspace=0.15, hspace=0.15)
plt.show()

# gamma[np.isnan(gamma)] = 0
# print(gamma.max())
