import argparse

import numpy as np
import torch

from mnist_classifier.mnist_dataset import MnistDataset
from mnist_classifier.train_test import TrainTest

def main():

    # Parse command line arguments.
    parser = argparse.ArgumentParser(description='Train an MNIST model, evaluate it, or deploy it.')
    parser.add_argument('--mode', help="Choose mode = {train, eval, launch_server, inference_server", choices=['train', 'eval', 'launch_server', 'inference_server'], required=True)
    parser.add_argument('--seed', help="Choose random seed.", default=None, type=int , required=False)
    args = parser.parse_args()

    # Set random seed.
    if args.seed != None:
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)

    # Enter mode.
    # If in in train or eval mode, get datasets and launch train or eval.
    # If server, launch server.
    if args.mode == 'train':
        train_dataset = MnistDataset("train")
        val_dataset = MnistDataset("val")
        train_test = TrainTest()
        train_test.launch_training(train_dataset, val_dataset)

    elif args.mode == 'eval':
        test_dataset = MnistDataset("test")
        train_test = TrainTest()
        train_test.launch_evaluation(test_dataset)

    elif args.mode =='launch_server':
        from mnist_classifier.deploy_launch_server import server as launch_server
        launch_server()

    elif args.mode =='inference_server':
        from mnist_classifier.deploy_inference_server import server as inference_server
        inference_server()
    
    else:
        raise ValueError("Wrong mode input.")


if __name__ == "__main__":
    main()