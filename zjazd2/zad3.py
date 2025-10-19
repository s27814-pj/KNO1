import argparse
import math

import tensorflow as tf
from numpy.matrixlib.defmatrix import matrix


def main():
    a = tf.constant([[1, 2], [3, 4]])
    b = tf.constant([[4, 5], [5, 6]])
    deg = math.radians(90)
    parser = argparse.ArgumentParser(description="degrees")
    parser.add_argument(
        "integers",
        metavar="N",
        type=int,
        nargs="+",
        help="an integer for the accumulator",
    )
    parser.add_argument("--size", help="degrees")
    parser.add_argument("--result", metavar="N", type=int, nargs="+")
    args = parser.parse_args()
    # print(args.integers)
    # print(args.size)
    # print(args.result)
    n = int(args.size)
    array = []
    matrix = tf.Variable(tf.zeros([n, n]))
    matrix_result = tf.Variable(tf.zeros([n, 1]))
    print(args.integers[2])
    for i in range(n):
        matrix_result[i].assign(int(args.result[i]))

    for i in range(n):
        for j in range(n):
            matrix[i, j].assign(int(args.integers[i + j]))

    print(matrix_result)
    print(matrix)

    out = tf.linalg.solve(matrix, matrix_result)

    print(out)


if __name__ == "__main__":
    main()
