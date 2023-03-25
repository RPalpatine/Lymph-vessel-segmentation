from __future__ import print_function


import argparse
from typing import List, Dict, Tuple
import engine, data_utils


parser = argparse.ArgumentParser(description='Keras lymph vessel segmentation')

parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')

parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--backbones', type=List, default=["vgg16", "resnet18"], metavar="B", 
                    help="Backbones (default is a list ['vgg16', 'resnet18'])")

parser.add_argument('--architecture', type=str, default="Unet", metavar="A",
                    help="Architecture, default is Unet")

parser.add_argument('--batch_size', type=int, default=16, metavar="BS",
                    help="Batch size, defaults to 16")

parser.add_argument('--input_dir', type=str, default=None, required=True, 
                    help="Path for input images, required.")

parser.add_argument('--masks_dir', type=str, required=True, default=None,
                    help="Path for masks, required.")

parser.add_argument('--test_dir', type=str, required=True, default=None,
                    help='Path to test images, required.')

parser.add_argument('--test_masks', type=str, required=True, default=None,
                    help='Path to test masks, required.')

parser.add_argument('--img_size', type=Tuple, default=(128, 128), 
                    help="Image size (tuple), defaults to 128, 128.")


args = parser.parse_args()


if __name__ == "__main__":

    print("Creating data")

    x_train, y_train, x_test, y_test, len_train, len_test = data_utils.create_data(input_dir=args.input_dir, 
                                                                                masks_dir=args.masks_dir,
                                                                                test_dir=args.test_dir,
                                                                                test_masks_dir=args.test_masks,
                                                                                seed=args.seed,
                                                                                img_size=args.img_size)

    print("Data created successfully. Length of x_train is: {}, and of x_test is: {}".format(len_train, len_test))

    print("Starting training: ")

    to_save = {}

    engine.train_models(epochs=args.epochs, backbones=args.backbones, x_train=x_train,
                        y_train=y_train,
                        x_val=x_test,
                        y_val=y_test,
                        to_save=to_save,
                        architecture=args.architecture,
                        learning_rate=args.lr,
                        batch_size=args.batch_size)

