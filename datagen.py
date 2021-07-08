from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
import numpy as np
import cv2 as cv

class clf_generator(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, zip_path_list):
        self.batch_size = batch_size
        self.img_size = img_size
        self.zip_path_list = zip_path_list

    def __len__(self):
        return len(self.zip_path_list) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.zip_path_list[i : i + self.batch_size]
        
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        y = np.zeros((self.batch_size,) + (1,), dtype="uint8")
        for j, path in enumerate(batch_input_img_paths):
            img = cv.imread(path[0])
            img = img.astype(np.float32)/255.0
            x[j] = img
        
            patch_group = path[1].split('.')[0][-1]; label=0
            if patch_group=='2' or patch_group=='3':
                label=1
            elif patch_group=='1' or patch_group=='4':
                label=0
            y[j] = np.expand_dims(label, 1)
            
        return x, y

class generator(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, zip_path_list):
        self.batch_size = batch_size
        self.img_size = img_size
        self.zip_path_list = zip_path_list
        self.augmentor = ImageDataGenerator(
            rotation_range=90,
            horizontal_flip=True,
            vertical_flip=True
        )
        
    def __len__(self):
        return len(self.zip_path_list) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_patch_pairs = self.zip_path_list[i : i + self.batch_size]
        
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="float32")
        for j, path in enumerate(batch_patch_pairs):
            img_path = path[0]; mask_path = path[1]
            #img = load_img(path, target_size=self.img_size)
            img = cv.imread(img_path)
            #img = img.astype(np.float32)/255.0
            x[j] = img
            if '_p3' in mask_path or '_p4' in mask_path:
                msk_img = np.zeros(self.img_size)
            else:
                msk_img = np.squeeze(cv.imread(mask_path,0))
            y[j] = np.expand_dims(msk_img, 2)
        x_gen = self.augmentor.flow(x, batch_size=self.batch_size, shuffle=False,seed=311)
        y_gen = self.augmentor.flow(y, batch_size=self.batch_size, shuffle=False,seed=311)
        return x, y