import os
import numpy as np


SOURCE = "assets/data"
FILE_A = "matrix_A"
FILE_B = "matrix_B"
FILE_C = "result_matrix_C"


def read_matrix(name):
    f = open(name)
    n = int(f.readline().strip())
    matrix = [list(map(int, f.readline().split())) for i in range(n)]
    return np.array(matrix)


def main():
    matrix_a = read_matrix(os.path.join(SOURCE, FILE_A))
    matrix_b = read_matrix(os.path.join(SOURCE, FILE_B))
    matrix_c = read_matrix(os.path.join(SOURCE, FILE_C))

    mul = matrix_a @ matrix_b
    if np.array_equal(matrix_c, mul):
        print("Верификация пройдена")
    else:
        print("Верификация не удалась")


if __name__ == "__main__":
    main()









