import timeit
import tracemalloc
import matplotlib.pyplot as plt
import numpy as np

N_values = [100_000, 1_000_000]

stmt_loop = "out = []\nfor i in range(N):\n    out.append(i * i)"
stmt_comp = "out = [i * i for i in range(N)]"
stmt_map  = "out = list(map(lambda i: i * i, range(N)))"

fragments = {
    "Цикл + append":       stmt_loop,
    "List comprehension":  stmt_comp,
    "map + lambda":        stmt_map,
}

time_results = {name: [] for name in fragments}
mem_results  = {name: [] for name in fragments}


print("ТРИ СПОСОБА СОЗДАТЬ СПИСОК КВАДРАТОВ")


for N in N_values:
    print(f"\n--- N = {N:,} ---")
    for name, stmt in fragments.items():
        t = min(timeit.repeat(stmt, globals={"N": N}, number=1, repeat=5))
        time_ms = t * 1000
        time_results[name].append(time_ms)

        tracemalloc.start()
        exec(stmt, {"N": N})
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        mem_mb = peak / (1024 * 1024)
        mem_results[name].append(mem_mb)

        print(f"  {name:20s}  время={time_ms:8.2f} мс   память={mem_mb:6.2f} МБ")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

markers = {"Цикл + append": "o", "List comprehension": "s", "map + lambda": "^"}
colors  = {"Цикл + append": "crimson", "List comprehension": "seagreen", "map + lambda": "steelblue"}

for name in fragments:
    ax1.plot(N_values, time_results[name],
             marker=markers[name], color=colors[name],
             linewidth=2, markersize=9, label=name)

ax1.set_xlabel("N (размер списка)", fontsize=12)
ax1.set_ylabel("Время (мс)", fontsize=12)
ax1.set_title("Время создания списка квадратов", fontsize=13, fontweight="bold")
ax1.set_xscale("log")
ax1.legend(fontsize=10)
ax1.grid(True, which="both", alpha=0.2)

x = np.arange(len(N_values))
width = 0.25

for i, (name, data) in enumerate(mem_results.items()):
    ax2.bar(x + (i - 1) * width, data, width,
            label=name, color=colors[name], alpha=0.85)

ax2.set_xticks(x)
ax2.set_xticklabels([f"N={N:,}" for N in N_values])
ax2.set_ylabel("Пиковая память (МБ)", fontsize=12)
ax2.set_title("Пиковая память создания списка", fontsize=13, fontweight="bold")
ax2.legend(fontsize=10)
ax2.grid(True, axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("task2_three_ways.png", dpi=150, bbox_inches="tight")
plt.show()

