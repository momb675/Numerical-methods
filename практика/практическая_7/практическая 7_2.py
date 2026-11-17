import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
points = rng.uniform(-2, 2, size=(200, 2))

scale = np.array([
    [1.5, 0.0],
    [0.0, 0.5]
])

theta = np.deg2rad(60)
rotate = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta),  np.cos(theta)]
])

translate = np.array([
    [1, 0, 1],
    [0, 1, -1],
    [0, 0, 1]
])

pts_scaled_rotated = points @ scale.T @ rotate.T

pts_h = np.hstack([pts_scaled_rotated, np.ones((len(points), 1))])
pts_translated = (translate @ pts_h.T).T
transformed = pts_translated[:, :2]

inv_translate = np.array([
    [1, 0, -1],
    [0, 1,  1],
    [0, 0,  1]
])

inv_rotate = np.array([
    [np.cos(-theta), -np.sin(-theta)],
    [np.sin(-theta),  np.cos(-theta)]
])

inv_scale = np.array([
    [1/1.5, 0.0],
    [0.0, 1/0.5]
])

pts_h_back = np.hstack([transformed, np.ones((len(points), 1))])
pts_back_untranslated = (inv_translate @ pts_h_back.T).T
pts_back_untranslated = pts_back_untranslated[:, :2]

restored = pts_back_untranslated @ inv_rotate.T @ inv_scale.T

print("Восстановление успешно:", np.allclose(points, restored, atol=1e-12))

fig, ax = plt.subplots(figsize=(8, 8))

ax.scatter(points[:, 0], points[:, 1], s=15, color='gray',
           alpha=0.5, label='Исходные')
ax.scatter(transformed[:, 0], transformed[:, 1], s=15, color='crimson',
           alpha=0.8, label='Преобразованные')
ax.scatter(restored[:, 0], restored[:, 1], s=40, facecolors='none',
           edgecolors='steelblue', linewidths=1.5, label='Восстановленные')

ax.axhline(0, color='black', lw=0.5)
ax.axvline(0, color='black', lw=0.5)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_title('Композиция и восстановление')

plt.tight_layout()
plt.show()