import os
import numpy as np

import torch
import torch.nn as nn
from scipy import ndimage
from skimage.transform import rescale, resize

## Data Loader
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        lst_data = os.listdir(self.data_dir)

        lst_label = [f for f in lst_data if f.startswith('label')]
        lst_input = [f for f in lst_data if f.startswith('input')]

        lst_label.sort()
        lst_input.sort()

        self.lst_label = lst_label
        self.lst_input = lst_input

    def __len__(self):
        return len(self.lst_label)

    def __getitem__(self, index):
        label = np.load(os.path.join(self.data_dir, self.lst_label[index]))
        input = np.load(os.path.join(self.data_dir, self.lst_input[index]))


        label = label/255.0
        # input = input - np.min(input)
        # input = input / 255.0
        input = input/255.0

        # If the images do not contain a channel axis, add this axis.
        if label.ndim == 3:
            label = label[:, :, :, np.newaxis]
        if input.ndim == 3:
            input = input[:, :, :, np.newaxis]

        data = {'input': input, 'label': label}

        if self.transform:
            data = self.transform(data)

        return data

## Implement Transformer
## Must-do transformer
class ToTensor(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        # [ZYXC] --> [CZYX]
        label = label.transpose((3, 0, 1, 2)).astype(np.float32)
        input = input.transpose((3, 0, 1, 2)).astype(np.float32)

        data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}

        return data

class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        label, input = data['label'], data['input']

        input = (input - self.mean) / self.std

        data = {'label': label, 'input': input}

        return data


class RandomFlip(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        if np.random.rand() > 0.5:
            label = np.flip(label,1)
            input = np.flip(input,1)

        if np.random.rand() > 0.5:
            label = np.flip(label,2)
            input = np.flip(input,2)

        data = {'label': label, 'input': input}

        return data

class RandomRotation_XY(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        if np.random.rand() > 0.75:
            angle = np.random.normal()*90
            label = ndimage.rotate(label, angle, axes= (1,2), mode = 'nearest', cval = np.min(label))
            input = ndimage.rotate(input, angle, axes= (1,2), mode = 'nearest', cval = np.min(input))

        data = {'label': label, 'input': input}
        return data

def corners(np_array):
    result = np.where(np_array > 0)
    x1 = np.min(result[0])
    x2 = np.max(result[0])
    y1 = np.min(result[1])
    y2 = np.max(result[1])
    z1 = np.min(result[2])
    z2 = np.max(result[2])
    # return x1, y1, x2, y2
    return x1, y1, z1, x2+1, y2+1, z2+1

def match_size(np_array, x1, x2, y1, y2, z1, z2, XYmax = 64, Zmax = 48):
        Lx = x2 - x1
        Ly = y2 - y1
        Lz = z2 - z1
        RI_cut = np.zeros(np_array.shape(0), max(Zmax, Lx), max(XYmax, Ly), max(XYmax, Lz))
        RI_cut[RI_cut.shape[0]//2-Lx//2:RI_cut.shape[0]//2+Lx-Lx//2,
        RI_cut.shape[1]//2-Ly//2:RI_cut.shape[1]//2+Ly-Ly//2,
        RI_cut.shape[2]//2-Lz//2:RI_cut.shape[1]//2+Lz-Lz//2] = np_array[x1:x2, y1:y2, z1:z2]


class RandomCrop(object):
    def __init__(self, shape = (48, 64, 64)):
        self.shape = shape

    def __call__(self, data):
        z, h, w, _ = data['label'].shape
        new_z, new_h, new_w = self.shape


        # Updated at Feb 3th 2023 - If the cropped region does not have sufficient sample region, remove it. 
        bool_finish = True
        while bool_finish:
          top = np.random.randint(0, max(h - new_h,1))
          left = np.random.randint(0, max(w - new_w,1))
          zup = np.random.randint(0, max(z - new_z,1))
          id_y = np.arange(top, top + new_h, 1)[:, np.newaxis]
          id_x = np.arange(left, left + new_w, 1)
          id_z = np.arange(zup, zup + new_z, 1)[:, np.newaxis, np.newaxis]

          bools_total = []
          sum_Z = np.sum(data['input'][id_z, id_y, id_x] > -0.3)
          bools_total.append(sum_Z > 125)
          sum_Z = np.sum(data['label'][id_z, id_y, id_x] > -0.3)
          bools_total.append(sum_Z > 125)
          # print(bools_total,all(bools_total))
          if all(bools_total):
            for key, value in data.items():
                data[key] = value[id_z, id_y, id_x]
            bool_finish = False

        return data

class RandomResize(object):
    def __init__(self, shape= (48, 64, 64)):
        self.shape = shape

    def __call__(self, data):
        if np.random.rand() > 0.75:
          resize_factor = np.random.normal()*0.2+1
          
          for key, value in data.items():
            temp = value*0
            img = rescale(value, resize_factor, anti_aliasing=True)
            if img.shape[0] < temp.shape[0]:
              Lx = img.shape[0]
              Ly = img.shape[1]
              Lz = img.shape[2]
              temp[temp.shape[0]//2-Lx//2:temp.shape[0]//2+Lx-Lx//2,
              temp.shape[1]//2-Ly//2:temp.shape[1]//2+Ly-Ly//2,
              temp.shape[2]//2-Lz//2:temp.shape[1]//2+Lz-Lz//2] = img
              data[key] = temp
        return data