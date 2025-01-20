import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark different file formats')
    parser.add_argument('--array-size', type=tuple, default=(10, 10, 1, 10, 10, 10),
                      help='Size of the test array')
    parser.add_argument('--chunk-size', type=tuple, default=(10, 5, 1, 1, 1, 1),
                      help='Chunk size for writing data')
    parser.add_argument('--iterations', type=int, default=5,
                      help='Number of iterations for each test')
    return parser.parse_args()
