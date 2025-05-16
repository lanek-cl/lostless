# predict_random.py
import sys
import time

from functions import test_random


def main():
    # Load dataset
    sample_size = int(sys.argv[1])
    path = sys.argv[2]
    start = time.time()
    labels, result = test_random(sample_size, path)
    print(labels, result)
    stop = time.time()
    print(f"Testing time: {stop - start:.2f} seconds")


if __name__ == "__main__":
    main()
