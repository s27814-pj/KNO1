import argparse

import tensorflow as tf
import math


# python full.py --equations 1 0 1 0 --size 2 --result 2 4
# python full.py --equations 1 6 1 9 --size 2 --result 2 4
# python full.py --equations 1 2 3 41 5 6 11 3 3 --size 3 --result 2 4 8
# python full.py --rotate 2 4 90


def main():
    parser = argparse.ArgumentParser(description="degrees")
    parser.add_argument(
        "--rotate",
        metavar="N",
        type=int,
        nargs="+",
        help="point a, point b, degrees separated by space",
        default=[],
    )
    parser.add_argument(
        "--equations",
        metavar="N",
        type=int,
        nargs="+",
        help="values on left side of =",
        default=[],
    )
    parser.add_argument("--size", help="degrees")
    parser.add_argument("--result", metavar="N", type=int, nargs="+")

    args = parser.parse_args()

    if len(args.rotate) > 2:
        print(
            rotate_point(
                int(args.rotate[0]),
                int(args.rotate[1]),
                math.radians(int(args.rotate[2])),
            )
        )
    elif len(args.equations) > 0:
        if len(args.equations) / int(args.size) != int(args.size) or len(
            args.result
        ) != int(args.size):
            print("invalid equations with size")
        else:
            matrix = tf.Variable(tf.zeros([int(args.size), int(args.size)]))
            matrix_result = tf.Variable(tf.zeros([int(args.size), 1]))
            for i in range(int(args.size)):
                matrix_result[i].assign(int(args.result[i]))

            for i in range(int(args.size)):
                for j in range(int(args.size)):
                    matrix[i, j].assign(int(args.equations[i * int(args.size) + j]))
            if tf.math.abs(tf.linalg.det(matrix)) < 1e-6:
                print("matrix not solvable")
            else:
                print(solve_equations(matrix, matrix_result))


@tf.function
def solve_equations(matrix, matrix_result):

    out = tf.linalg.solve(matrix, matrix_result)

    return out


@tf.function
def rotate_point(a, b, deg):
    selected_point = tf.constant([[a, b, 1]], float)  # x,y,1
    rotation_matrix = tf.constant(
        [
            [math.cos(deg), math.sin(deg), 0],
            [-math.sin(deg), math.cos(deg), 0],
            [0, 0, 1],
        ],
        float,
    )
    output = tf.matmul(selected_point, rotation_matrix)
    return output


if __name__ == "__main__":
    main()
