import time
import numpy as np
import math

def measure(fn, *args, repeat=3):
    best = float("inf")
    result = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = fn(*args)
        t1 = time.perf_counter()
        best = min(best, t1 - t0)
    return result, best

# Задача 1
rng = np.random.default_rng(42)
a = rng.normal(size=10**6)

def loop_sum(arr):
    total = 0.0
    for x in arr:
        total += x
    return total

def numpy_sum(arr):
    return arr.sum()

r_loop,  t_loop  = measure(loop_sum,  a)
r_numpy, t_numpy = measure(numpy_sum, a)
print(f"цикл:   {t_loop*1000:10.2f} мс")
print(f"NumPy:  {t_numpy*1000:10.2f} мс")
print(f"ускорение: {t_loop / t_numpy:.1f}×")
print(f"результаты совпадают: {np.isclose(r_loop, r_numpy)}\n")

# Задача 2
rng = np.random.default_rng(42)
a = rng.normal(size=10**6)

def loop_mean_std(arr):
    n = len(arr)
    s = 0.0
    for x in arr: s += x
    mean = s / n
    sq = 0.0
    for x in arr: sq += (x - mean) ** 2
    return mean, math.sqrt(sq / n)

def numpy_mean_std(arr):
    return arr.mean(), arr.std()

r_loop,  t_loop  = measure(loop_mean_std,  a)
r_numpy, t_numpy = measure(numpy_mean_std, a)
print(f"цикл:   {t_loop*1000:10.2f} мс")
print(f"NumPy:  {t_numpy*1000:10.2f} мс")
print(f"ускорение: {t_loop / t_numpy:.1f}×")
print(f"результаты совпадают: {np.allclose(r_loop, r_numpy)}\n")

# Задача 3
rng = np.random.default_rng(42)
a = rng.normal(size=10**6)

def loop_sin_sq(arr):
    res = [0.0] * len(arr)
    for i, x in enumerate(arr):
        res[i] = math.sin(x) + x * x
    return res

def numpy_sin_sq(arr):
    return np.sin(arr) + arr**2

r_loop,  t_loop  = measure(loop_sin_sq,  a)
r_numpy, t_numpy = measure(numpy_sin_sq, a)
print(f"цикл:   {t_loop*1000:10.2f} мс")
print(f"NumPy:  {t_numpy*1000:10.2f} мс")
print(f"ускорение: {t_loop / t_numpy:.1f}×")
print(f"результаты совпадают: {np.allclose(r_loop, r_numpy)}\n")

# Задача 4
rng = np.random.default_rng(42)
a = rng.normal(size=10**6)

def loop_clip(arr):
    res = [0.0] * len(arr)
    for i, x in enumerate(arr):
        if x < 0: res[i] = 0
        elif x > 1: res[i] = 1
        else: res[i] = x
    return res

def numpy_clip(arr):
    return np.clip(arr, 0, 1)

r_loop,  t_loop  = measure(loop_clip,  a)
r_numpy, t_numpy = measure(numpy_clip, a)
print(f"цикл:   {t_loop*1000:10.2f} мс")
print(f"NumPy:  {t_numpy*1000:10.2f} мс")
print(f"ускорение: {t_loop / t_numpy:.1f}×")
print(f"результаты совпадают: {np.allclose(r_loop, r_numpy)}\n")

# Задача 5
rng = np.random.default_rng(42)
m = rng.normal(size=(100_000, 10))

def loop_sum_rows(mat):
    result = [0.0] * mat.shape[0]
    for i in range(mat.shape[0]):
        s = 0.0
        for j in range(mat.shape[1]):
            s += mat[i, j]
        result[i] = s
    return result

def numpy_sum_rows(mat):
    return mat.sum(axis=1)

r_loop,  t_loop  = measure(loop_sum_rows,  m)
r_numpy, t_numpy = measure(numpy_sum_rows, m)
print(f"цикл:   {t_loop*1000:10.2f} мс")
print(f"NumPy:  {t_numpy*1000:10.2f} мс")
print(f"ускорение: {t_loop / t_numpy:.1f}×")
print(f"результаты совпадают: {np.allclose(r_loop, r_numpy)}\n")

# Задача 6
rng = np.random.default_rng(42)
m = rng.normal(size=(100_000, 10))

def loop_norm_cols(mat):
    n, k = mat.shape
    means = [0.0] * k
    for j in range(k):
        s = 0.0
        for i in range(n): s += mat[i, j]
        means[j] = s / n
    res = np.zeros_like(mat)
    for i in range(n):
        for j in range(k):
            res[i, j] = mat[i, j] - means[j]
    return res

def numpy_norm_cols(mat):
    return mat - mat.mean(axis=0)

r_loop,  t_loop  = measure(loop_norm_cols,  m)
r_numpy, t_numpy = measure(numpy_norm_cols, m)
print(f"цикл:   {t_loop*1000:10.2f} мс")
print(f"NumPy:  {t_numpy*1000:10.2f} мс")
print(f"ускорение: {t_loop / t_numpy:.1f}×")
print(f"результаты совпадают: {np.allclose(r_loop, r_numpy)}\n")

# Задача 7
rng = np.random.default_rng(42)
a = rng.normal(size=10**6)

def loop_max(arr):
    mx, idx = arr[0], 0
    for i in range(1, len(arr)):
        if arr[i] > mx:
            mx, idx = arr[i], i
    return mx, idx

def numpy_max(arr):
    return arr.max(), arr.argmax()

r_loop,  t_loop  = measure(loop_max,  a)
r_numpy, t_numpy = measure(numpy_max, a)
print(f"цикл:   {t_loop*1000:10.2f} мс")
print(f"NumPy:  {t_numpy*1000:10.2f} мс")
print(f"ускорение: {t_loop / t_numpy:.1f}×")
print(f"результаты совпадают: {np.isclose(r_loop[0], r_numpy[0]) and r_loop[1] == r_numpy[1]}\n")

# Задача 8
rng = np.random.default_rng(42)
a = rng.normal(size=10**6)

def loop_filter(arr):
    res = []
    for x in arr:
        if x > 0: res.append(x)
    return res

def numpy_filter(arr):
    return arr[arr > 0]

r_loop,  t_loop  = measure(loop_filter,  a)
r_numpy, t_numpy = measure(numpy_filter, a)
print(f"цикл:   {t_loop*1000:10.2f} мс")
print(f"NumPy:  {t_numpy*1000:10.2f} мс")
print(f"ускорение: {t_loop / t_numpy:.1f}×")
print(f"результаты совпадают: {np.allclose(r_loop, r_numpy)}\n")

# Задача 9
rng = np.random.default_rng(42)
a = rng.normal(size=10**6)

def loop_abs_inplace(arr):
    for i in range(len(arr)):
        if arr[i] < 0: arr[i] = -arr[i]
    return arr  

def numpy_abs_inplace(arr):
    return np.abs(arr, out=arr)

r_loop,  t_loop  = measure(loop_abs_inplace,  a.copy())
r_numpy, t_numpy = measure(numpy_abs_inplace, a.copy())

print(f"цикл:   {t_loop*1000:10.2f} мс")
print(f"NumPy:  {t_numpy*1000:10.2f} мс")
print(f"ускорение: {t_loop / t_numpy:.1f}×")
print(f"результаты совпадают: {np.allclose(r_loop, r_numpy)}\n")

# Задача 10
rng = np.random.default_rng(42)
a = rng.normal(size=10**6)

def loop_comb_mask(arr):
    res = []
    for x in arr:
        if -1 < x < 1: res.append(x)
    return res

def numpy_comb_mask(arr):
    return arr[(arr > -1) & (arr < 1)]

r_loop,  t_loop  = measure(loop_comb_mask,  a)
r_numpy, t_numpy = measure(numpy_comb_mask, a)
print(f"цикл:   {t_loop*1000:10.2f} мс")
print(f"NumPy:  {t_numpy*1000:10.2f} мс")
print(f"ускорение: {t_loop / t_numpy:.1f}×")
print(f"результаты совпадают: {np.allclose(r_loop, r_numpy)}\n")

# Задача 11
rng = np.random.default_rng(42)
X = rng.normal(size=(1000, 3))
Y = rng.normal(size=(1000, 3))

def loop_dist(x, y):
    n, m = x.shape[0], y.shape[0]
    D = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            d = 0.0
            for k in range(3):
                d += (x[i, k] - y[j, k])**2
            D[i, j] = math.sqrt(d)
    return D

def numpy_dist(x, y):
    return np.sqrt(((x[:, None, :] - y[None, :, :])**2).sum(axis=2))

r_loop,  t_loop  = measure(loop_dist,  X, Y)
r_numpy, t_numpy = measure(numpy_dist, X, Y)
print(f"цикл:   {t_loop*1000:10.2f} мс")
print(f"NumPy:  {t_numpy*1000:10.2f} мс")
print(f"ускорение: {t_loop / t_numpy:.1f}×")
print(f"результаты совпадают: {np.allclose(r_loop, r_numpy)}\n")

# Задача 12
rng = np.random.default_rng(42)
a = rng.normal(size=10**5)
k = 10

def loop_moving_avg(arr):
    res = []
    for i in range(k - 1, len(arr)):
        s = 0.0
        for j in range(i - k + 1, i + 1): s += arr[j]
        res.append(s / k)
    return res

def numpy_moving_avg(arr):
    return np.convolve(arr, np.ones(k)/k, mode='valid')

r_loop,  t_loop  = measure(loop_moving_avg,  a)
r_numpy, t_numpy = measure(numpy_moving_avg, a)
print(f"цикл:   {t_loop*1000:10.2f} мс")
print(f"NumPy:  {t_numpy*1000:10.2f} мс")
print(f"ускорение: {t_loop / t_numpy:.1f}×")
print(f"результаты совпадают: {np.allclose(r_loop, r_numpy)}\n")

# Задача 13
rng = np.random.default_rng(42)
a = rng.normal(size=10**5)

def loop_cumsum(arr):
    s = 0.0
    res = []
    for x in arr:
        s += x
        res.append(s)
    return res

def numpy_cumsum(arr):
    return np.cumsum(arr)

r_loop,  t_loop  = measure(loop_cumsum,  a)
r_numpy, t_numpy = measure(numpy_cumsum, a)
print(f"цикл:   {t_loop*1000:10.2f} мс")
print(f"NumPy:  {t_numpy*1000:10.2f} мс")
print(f"ускорение: {t_loop / t_numpy:.1f}×")
print(f"результаты совпадают: {np.allclose(r_loop, r_numpy)}\n")

# Задача 14
rng = np.random.default_rng(42)
ints = rng.integers(0, 100, size=10**6)

def loop_bincount(arr):
    counts = {}
    for x in arr:
        counts[x] = counts.get(x, 0) + 1
    return counts

def numpy_bincount(arr):
    return np.bincount(arr, minlength=100)

r_loop,  t_loop  = measure(loop_bincount,  ints)
r_numpy, t_numpy = measure(numpy_bincount, ints)

print(f"цикл:   {t_loop*1000:10.2f} мс")
print(f"NumPy:  {t_numpy*1000:10.2f} мс")
print(f"ускорение: {t_loop / t_numpy:.1f}×")

loop_counts = [r_loop.get(i, 0) for i in range(100)]
print(f"результаты совпадают: {np.array_equal(loop_counts, r_numpy)}\n")

# Задача 15
rng = np.random.default_rng(42)
raw = rng.normal(size=(10**5, 3))
raw[rng.integers(0, 10**5, 100), rng.integers(0, 3, 100)] = 1e6

def loop_pipeline(data):
    means = [0.0]*3
    stds = [0.0]*3
    for j in range(3):
        s = 0.0
        for i in range(data.shape[0]): s += data[i, j]
        means[j] = s / data.shape[0]
        sq = 0.0
        for i in range(data.shape[0]): sq += (data[i, j] - means[j])**2
        stds[j] = math.sqrt(sq / data.shape[0])
    for i in range(data.shape[0]):
        for j in range(3):
            if abs(data[i, j] - means[j]) > 5 * stds[j]:
                data[i, j] = means[j]
    res = np.zeros_like(data)
    for i in range(data.shape[0]):
        for j in range(3):
            res[i, j] = (data[i, j] - means[j]) / stds[j]
    return res

def numpy_pipeline(data):
    means = data.mean(axis=0)
    stds = data.std(axis=0)
    mask = np.abs(data - means) > 5 * stds
    data[mask] = np.nan
    col_means = np.nanmean(data, axis=0)
    data = np.where(np.isnan(data), col_means, data)
    return (data - data.mean(axis=0)) / data.std(axis=0)

r_loop,  t_loop  = measure(loop_pipeline,  raw.copy())
r_numpy, t_numpy = measure(numpy_pipeline, raw.copy())
print(f"цикл:   {t_loop*1000:10.2f} мс")
print(f"NumPy:  {t_numpy*1000:10.2f} мс")
print(f"ускорение: {t_loop / t_numpy:.1f}×\n")