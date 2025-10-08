"""This is main.py use this to add arguments and run your training and evaluation code"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Training and Evaluation")
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/dtu/datasets1/02516/ufc10",
        help="Root directory of the dataset",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val"],
        help="Dataset split to use",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["2D_CNN_aggr", "2D_CNN_late_fusion", "2D_CNN_early_fusion", "3D_CNN"],
        help="Model architecture to use",
    )
    args = parser.parse_args()
    return args
