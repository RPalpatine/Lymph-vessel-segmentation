import os
from typing import Dict, Tuple
import random
from keras.utils import load_img, img_to_array
import numpy as np

def create_data(input_dir: str, masks_dir: str, test_dir: str, test_masks_dir: str, seed: int, img_size: Tuple):
    
    input_img_paths = sorted(
        [
            os.path.join(input_dir, fname)
            for fname in os.listdir(input_dir)
            if fname.endswith(".jpg")
        ]
    )
    target_img_paths = sorted(
        [
            os.path.join(masks_dir, fname)
            for fname in os.listdir(masks_dir)
            if fname.endswith(".png") and not fname.startswith(".")
        ]
    )

    test_input_img_paths = sorted(
        [
            os.path.join(test_dir, fname)
            for fname in os.listdir(test_dir)
            if fname.endswith(".jpg")
        ]
    )
    test_target_img_paths = sorted(
        [
            os.path.join(test_masks_dir, fname)
            for fname in os.listdir(test_masks_dir)
            if fname.endswith(".png") and not fname.startswith(".")
        ]
    )
    
    random.Random(seed).shuffle(input_img_paths)
    random.Random(seed).shuffle(target_img_paths)
    
    num_imgs = len(input_img_paths)
    test_num_imgs = len(test_input_img_paths)
    
    def path_to_input_image(path):
        return img_to_array(load_img(path, target_size = img_size))

    def path_to_target(path):
        img = img_to_array(load_img(path, target_size = img_size, color_mode = "grayscale"))
        img = img.astype("uint8")/255
        return img
    
    input_imgs = np.zeros((num_imgs,) + img_size + (3,), dtype="float32")
    targets = np.zeros((num_imgs,) + img_size + (1,), dtype ="uint8")

    test_input_imgs = np.zeros((test_num_imgs,) + img_size + (3,), dtype="float32")
    test_targets = np.zeros((test_num_imgs,) + img_size + (1,), dtype ="uint8")

    for i in range(num_imgs):
        input_imgs[i] = path_to_input_image(input_img_paths[i])
        targets[i] = path_to_target(target_img_paths[i])
        
    for i in range(test_num_imgs):
        test_input_imgs[i] = path_to_input_image(test_input_img_paths[i])
        test_targets[i] = path_to_target(test_target_img_paths[i])
    
    x_train = input_imgs.astype(np.float32)
    y_train = targets.astype(np.float32)
    x_test = test_input_imgs.astype(np.float32)
    y_test = test_targets.astype(np.float32)
    
    return x_train, y_train, x_test, y_test, len(x_train), len(x_test)


