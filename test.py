# TODO add test cases

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-test", action="store_true", help="Whether to freeze the feature encoder")

args = parser.parse_args()

if args.test:
    print("test")

