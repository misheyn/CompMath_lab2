import numpy as np
import timeit


def matrix_max_row(matrix, n):
    max_elem = matrix[n][n]
    max_row = n
    for i in range(n + 1, len(matrix)):
        if abs(matrix[n][i]) > abs(max_elem):
            max_elem = matrix[n][i]
            max_row = i
        if max_row != n:
            matrix[n], matrix[max_row] = matrix[max_row], matrix[n]


def gauss_method(matrix):
    n = len(matrix)
    x = np.zeros(n)
    for k in range(n - 1):
        matrix_max_row(matrix, k)
        for i in range(k + 1, n):
            div = matrix[i][k] / matrix[k][k]
            matrix[i][-1] -= div * matrix[k][-1]
            for j in range(k, n):
                matrix[i][j] -= div * matrix[k][j]
    if is_singular(matrix):
        print('The system has infinite number of answers')
        return
    for k in range(n - 1, -1, -1):
        x[k] = (matrix[k][-1] - sum([matrix[k][j] * x[j] for j in range(k + 1, n)])) / matrix[k][k]
        x[k] = float("%.4f" % x[k])
    return x


def is_singular(matrix):
    for i in range(len(matrix)):
        if not matrix[i][i]:
            return True
        return False


def sign(indexes):
    s = sum((0, 1)[x < y] for k, y in enumerate(indexes) for x in indexes[k + 1:])
    return (s + 1) % 2 - s % 2


def column(row):
    i = 0
    n = len(row)
    while not row[i] and i < n:
        i += 1
    return i if i < n else -1


def kramer_method(les):
    n = len(les)
    x = np.zeros(n)
    tmp = list(zip(*les))
    b = tmp[-1]
    del tmp[-1]

    delta = np.linalg.det(tmp)
    if not delta:
        raise RuntimeError("No solution")

    result = []
    for i in range(n):
        a = tmp[:]
        a[i] = b
        result.append(np.linalg.det(a) / delta)
        x[i] = result[i]
        x[i] = float("%.4f" % x[i])
    return x


def iteration_method(a, b, eps):
    n = len(a)
    max = 1
    x = np.zeros(n)
    condition = False
    while not condition:
        x_new = np.copy(x)
        for i in range(n):
            s1 = sum(a[i][j] * x_new[j] for j in range(i))
            s2 = sum(a[i][j] * x[j] for j in range(i + 1, n))
            x_new[i] = (b[i] - s1 - s2) / a[i][i]
            x_new[i] = float("%.4f" % x_new[i])
        for i in range(1, n - 1):
            if abs(x_new[max] - x[max]) < abs(x_new[i + 1] - x[i + 1]):
                max = i + 1
        condition = abs(x_new[max] - x[max]) < eps
        x = x_new

    return x


a = [[25.0, -1.0, 1.0, 1.0],
     [1.0, 20.0, -1.0, 1.0],
     [1.0, 1.0, 10.0, 2.0],
     [12.0, 56.0, 0.0, 30.0]]
b = [39.0, 0.0, -25.0, 18.0]
c = []
x = np.zeros(4)
eps = 10e-4

for i in range(len(a)):
    c.append(a[i].copy())
    c[i].append(b[i])

print("Kramer method: ")
start_time = timeit.default_timer()
print(kramer_method(c))
time = (timeit.default_timer() - start_time) * 1000
print("Runtime in milliseconds: %.3f" % time)

print("\nIteration method: ")
start_time = timeit.default_timer()
print(iteration_method(a, b, eps))
time = (timeit.default_timer() - start_time) * 1000
print("Runtime in milliseconds: %.3f" % time)

print("\nGaussian method with pivot selection by column:")
start_time = timeit.default_timer()
print(gauss_method(c))
time = (timeit.default_timer() - start_time) * 1000
print("Runtime in milliseconds: %.3f" % time)

print("\nCheck:")
res = np.zeros(4)
for i in range(len(res)):
    res[i] = np.linalg.solve(a, b)[i]
    res[i] = float("%.4f" % res[i])
print(res)
