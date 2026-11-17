import timeit
import matplotlib.pyplot as plt

n_values = [100, 1000, 10000]
target = -1

times_list = []
times_set = []

print("СРАВНЕНИЕ ПОИСКА: list vs set")
print(f"Ищем target={target} (отсутствует в коллекции)\n")

for n in n_values:
    data_list = list(range(n))
    data_set = set(range(n))

    t_list = min(timeit.repeat(
        "target in data_list",
        globals={"target": target, "data_list": data_list},
        number=1000,
        repeat=5,
    ))
    us_list = t_list / 1000 * 1_000_000

    t_set = min(timeit.repeat(
        "target in data_set",
        globals={"target": target, "data_set": data_set},
        number=1000,
        repeat=5,
    ))
    us_set = t_set / 1000 * 1_000_000

    times_list.append(us_list)
    times_set.append(us_set)

    ratio = us_list / us_set
    print(f"n={n:6d}:  list={us_list:8.3f} мкс   set={us_set:8.3f} мкс   "
          f"set быстрее в {ratio:6.1f} раз")

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(n_values, times_list, "o-", color="crimson",
        linewidth=2, markersize=8, label="x in list — O(n)")
ax.plot(n_values, times_set, "s-", color="steelblue",
        linewidth=2, markersize=8, label="x in set — O(1)")

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel("Размер коллекции (n, лог. шкала)", fontsize=12)
ax.set_ylabel("Время одной проверки (мкс, лог. шкала)", fontsize=12)
ax.set_title("Поиск в list vs set: set не зависит от размера",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, which="both", alpha=0.2)

plt.tight_layout()
plt.savefig("task1_list_vs_set.png", dpi=150, bbox_inches="tight")
plt.show()