import os
import argparse

def parse_args():
    """
    args for training.
    :return: args
    """
    parser = argparse.ArgumentParser(description='Parse args for training')

    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    if args.mode = "single":
        train_cmd = "python lib/train/run_trainig.py"

    else :
        raise ValueError("mode should be 'single' or 'multiple'.")
    os.system(train_cmd)

if __name__ == "__main__":
    main()