import torch
import numpy as np
import math
import json
import os
import argparse

def default(obj):
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    raise TypeError('Unknown type:', type(obj))

def split_data(dataset, args:argparse.Namespace) -> dict:
    """
    Split the dataset into training and validation sets, if not already split.

    Args:
        dataset: The dataset to split
        args: The command line arguments
    
    Returns:
        A dictionary containing the training and validation sets
    """

    # Load or generate splits
    if args.split_file:
        with open(args.split_file, "r") as fp:
            splits = json.load(fp)
    else:
        datalen = len(dataset)
        # perform a random split of the dataset
        num_validation_val = int(math.ceil(datalen * args.val_ratio))
        num_validation_test = int(math.ceil(datalen * args.test_ratio))
        indices = np.random.permutation(datalen)
        splits = {
            "train": indices[num_validation_val + num_validation_test:],
            "val": indices[:num_validation_val],
            "test": indices[num_validation_val:num_validation_val + num_validation_test]
        }

    # Save split file
    with open(os.path.join(args.output_dir, "datasplits.json"), "w") as f:
        json.dump(splits, f, default=default)

    # Split the dataset
    datasplits = {}
    for key, indices in splits.items():
        datasplits[key] = torch.utils.data.Subset(dataset, indices)
    return datasplits