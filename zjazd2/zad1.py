import argparse
import math

import tensorflow as tf


@tf.function
def rotate_point(deg):
    selected_point = tf.constant([[9, 6, 1]], float)  # x,y,1
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


def main():
    a = tf.constant([[1, 2], [3, 4]])
    b = tf.constant([[4, 5], [5, 6]])
    deg = math.radians(90)
    parser = argparse.ArgumentParser(description="degrees")
    parser.add_argument("--deg", help="degrees")
    args = parser.parse_args()
    if args.deg:
        deg = math.radians(int(args.deg))
        print(rotate_point(deg))


if __name__ == "__main__":
    main()
