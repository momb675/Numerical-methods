import numpy as np
import matplotlib.pyplot as plt

k = np.arange(6)
x = np.cos(2 * np.pi * k / 6)
y = np.sin(2 * np.pi * k / 6)

hexagon = np.column_stack((x, y))
hexagon = np.vstack((hexagon, hexagon[0]))

theta = np.deg2rad(45)
rotate_45 = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta),  np.cos(theta)]
])

reflect_yx = np.array([
    [0, 1],
    [1, 0]
])

rotated = hexagon @ rotate_45.T
reflected = hexagon @ reflect_yx.T

fig, ax = plt.subplots(figsize=(7, 7))

ax.plot(hexagon[:, 0], hexagon[:, 1], 'o-', color='gray',
        label='Исходный', lw=2, markersize=6)
ax.plot(rotated[:, 0], rotated[:, 1], 'o-', color='steelblue',
        label='Поворот на 45°', lw=2, markersize=6)
ax.plot(reflected[:, 0], reflected[:, 1], 'o-', color='crimson',
        label='Отражение от y=x', lw=2, markersize=6)

ax.axhline(0, color='black', lw=0.5)
ax.axvline(0, color='black', lw=0.5)

ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_title('Шестиугольник: поворот и отражение')

plt.tight_layout()
plt.show()