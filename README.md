# Color Image Processing Homework 2

> Author: M11102122 陳煜翔

> Date: 2022/12/09

[TOC]

## Intorduction

It has 3 question in this homework.

First is fuse 2 image with different focus point to a front and background all clear image.

Second is to simulate visual abnormalities image.

Final is to analysis the leaves with different x-y axis scale and see the different.

## Coding Detail

### Envionment

I use Python environment by version 3.8, and use cv2, numpy, torch to achieve the question.

### utils.py

I type all qestion in utils.py file, and use class data structure to define the qestion.

```python
import numpy as np
import cv2
import torch
import pandas as pd
import os

class fusion_depth:
    def __init__(self, fg_path:str, bg_path:str):
        self.fg_image = cv2.imread(fg_path)
        self.bg_image = cv2.imread(bg_path)

    def hipass(self):
        Laplacian_filter = 1/9 * np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]) # High pass filter
        gray_fg = cv2.cvtColor(self.fg_image, cv2.COLOR_BGR2GRAY).astype(np.float32) # Transform to grayscale image
        gray_bg = cv2.cvtColor(self.bg_image, cv2.COLOR_BGR2GRAY).astype(np.float32)

        gray_fg = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(gray_fg), 0), 0)
        gray_bg = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(gray_bg), 0), 0)
        
        conv = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        conv.weight = torch.nn.Parameter(torch.unsqueeze(torch.unsqueeze(torch.from_numpy(Laplacian_filter), 0), 0).float())

        fg = conv(gray_fg) # Get highpass fg image
        fg = torch.abs(fg) # Absolute value fg image
        bg = conv(gray_bg) # Get highpass bg image
        bg = torch.abs(bg) # Absolute value fg image

        fg_highpass = fg
        fg_highpass = torch.squeeze(fg_highpass)
        fg_highpass = fg_highpass.detach().numpy() # Get numpy format highpass image for fg
        bg_highpass = bg
        bg_highpass = torch.squeeze(bg_highpass)
        bg_highpass = bg_highpass.detach().numpy() # Get numpy format highpass image for bg

        mask = fg - bg # Mask Create

        Median_filter = 1/(20 * 20) * np.ones((20, 20)) # Median filter
        conv = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=20, padding="same", bias=False)
        conv.weight = torch.nn.Parameter(torch.unsqueeze(torch.unsqueeze(torch.from_numpy(Median_filter), 0), 0).float())
        mask = conv(mask)
        mask = torch.squeeze(mask)
        mask = mask.cpu().detach().numpy()

        fg = torch.squeeze(fg)
        bg = torch.squeeze(bg)  
        fg = fg.cpu().detach().numpy()
        bg = bg.cpu().detach().numpy()

        mask_fg = np.where(mask > 0, 1, 0) # Binary mask
        mask_bg = np.where(mask > 0, 0, 1)
        mask_fg = np.expand_dims(mask_fg, axis=2)
        mask_bg = np.expand_dims(mask_bg, axis=2)
        
        mask_fg = np.concatenate((mask_fg, mask_fg, mask_fg), axis=2)
        mask_bg = np.concatenate((mask_bg, mask_bg, mask_bg), axis=2)

        fusion_image = self.fg_image * mask_fg + self.bg_image * mask_bg # Fuse image

        return fusion_image, mask, fg_highpass, bg_highpass


class simulate_abnormal_vision:
    def __init__(self, image_path: str):
        self.image = cv2.imread(image_path).astype(np.float32) / 255

    def red_green_blind(self):
        Lab_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2Lab) # Transfer to Lab System
        Lab_image = np.transpose(Lab_image, (2, 0, 1))
        Lab_image[1] = 0 # Let a = 0
        Lab_image = np.transpose(Lab_image, (1, 2, 0))
        Lab_image = cv2.cvtColor(Lab_image, cv2.COLOR_Lab2BGR) * 255 # Transfer to RGB System

        return Lab_image

    def blue_yellow_blind(self):
        Lab_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2Lab) # Transfer to Lab System
        Lab_image = np.transpose(Lab_image, (2, 0, 1))
        Lab_image[2] = 0 # Let b = 0
        Lab_image = np.transpose(Lab_image, (1, 2, 0))
        Lab_image = cv2.cvtColor(Lab_image, cv2.COLOR_Lab2BGR) * 255 # Transfer to RGB System

        return Lab_image

    def glaucoma(self, sigma):
        pi = 3.141592653589793
        exponential = 2.718
        gaussian_filter = np.zeros(self.image[:,:,2].shape)
        center_point = (gaussian_filter.shape[0] / 2, gaussian_filter.shape[1]/2) # Find center point
        for i in range(gaussian_filter.shape[0]):
            for j in range(gaussian_filter.shape[1]):
                gaussian_filter[i][j] = (1/(2 * pi * (sigma ** 2))) * (exponential ** -((((i - center_point[0]) **2 ) + ((j - center_point[1]) ** 2)) / (2 * (sigma ** 2))))
        gaussian_filter /= gaussian_filter.max()

        gaussian_filter = np.expand_dims(gaussian_filter, axis=2)
        gaussian_filter = np.concatenate([gaussian_filter, gaussian_filter, gaussian_filter], axis=2) # Expand gaussian filter
        glaucoma_image = self.image * gaussian_filter * 255

        return glaucoma_image
        

class different_leave:
    def __init__(self, image_path: list, signatures_csv: str):
        self.image_path = image_path
        self.signatures_csv = pd.read_csv(signatures_csv, header=None)

    
    def get_mask(self, image_path) -> np.ndarray:
        image = cv2.imread(image_path, 0) # Using 0 to read image in grayscale mode
        image_mask = np.where(image == 255, 0, 1) # Get mask
        return image_mask

    def leave_feature_statistics(self) -> dict:
        feature_statistics = {}
        Brightness = []
        Red = [] 
        High_freq_image = []
        signatures = []
        mask = []
        for idx in range(len(self.image_path)):
            image = cv2.imread(self.image_path[idx]).astype(np.float32)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
            image_mask = self.get_mask(self.image_path[idx])
            image_mask = np.expand_dims(image_mask, axis=0)
            image_mask_3ch = np.concatenate([image_mask, image_mask, image_mask], axis=0)
            image = np.transpose(image, (2,0,1))
            image = image * image_mask_3ch # Get front image

            mask.append(image_mask_3ch)
            Brightness.append(np.sum(gray_image * image_mask) / np.sum(image_mask)) # Brightness
            Red.append(np.sum(image[2]) / np.sum(image))

            Laplacian_filter = 1/9 * np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]) # High pass filter
            conv = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
            conv.weight = torch.nn.Parameter(torch.unsqueeze(torch.unsqueeze(torch.from_numpy(Laplacian_filter), 0), 0).float())
            gray_image = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(gray_image), dim=0), dim=0) # (1, 1, W, H) shape
            gray_image_conv = conv(gray_image)
            gray_image_conv = torch.squeeze(gray_image_conv) # (W, H) shape
            gray_image_conv = torch.abs(gray_image_conv)
            gray_image_conv = gray_image_conv.cpu().detach().numpy()
            gray_image_conv = np.squeeze(gray_image_conv)
            image_mask = np.squeeze(image_mask)
            High_freq_image.append(np.sum(gray_image_conv * image_mask) / np.sum(image_mask)) # High_freq_image

            signatures_data = self.signatures_csv.iloc[idx].tolist()
            signatures_data = np.mean(np.abs(np.gradient(signatures_data)))
            signatures.append(signatures_data)

        feature_statistics["Brightness"] = Brightness
        feature_statistics["Red"] = Red
        feature_statistics["High_freq_image"] = High_freq_image
        feature_statistics["signatures"] = signatures
        feature_statistics["mask"] = mask
        return feature_statistics

    def plot_statistics(self):
        feature_statistic = self.leave_feature_statistics() # Get statistic
        statistic_image = []

        Max = 880 # Set New maximum value

        key_pair = {"Red_Light": ["Red", "Brightness"],
                    "Signature_HighFreq": ["signatures", "High_freq_image"]}

        for key, pair in key_pair.items():
            background = 255 * np.ones((1000, 1000, 3)) # White Background
            for idx in range(len(self.image_path)):
                image = cv2.imread(self.image_path[idx]).astype(np.float32)
                
                max_x, min_x = max(feature_statistic[pair[1]]), min(feature_statistic[pair[1]])
                max_y, min_y = max(feature_statistic[pair[0]]), min(feature_statistic[pair[0]])
                

                x_start = np.ceil((feature_statistic[pair[1]][idx] - min_x) * (Max / (max_x - min_x))).astype(np.int16) # Find x,y start point (Left upper corner)
                y_start = np.ceil((feature_statistic[pair[0]][idx] - min_y) * (Max / (max_y - min_y))).astype(np.int16)

                mask = np.transpose(feature_statistic["mask"][idx], (1, 2, 0)) 
                inverse_mask = np.where(mask == 0, 1, 0)

                leave_position_data = background[background.shape[0] - 1 - x_start - image.shape[0]:background.shape[0] - 1 - x_start, y_start:y_start + image.shape[1] , :] # read paste image
                leave_position_data = leave_position_data * inverse_mask + image * mask


                background[background.shape[0] - 1 - x_start - image.shape[0]:background.shape[0] - 1 - x_start, y_start:y_start + image.shape[1] , :] = leave_position_data # paste image

            statistic_image.append(background)

        return list(key_pair.keys()), statistic_image
```

### main.py
```python
import utils
import config
import cv2
import os

if __name__ == "__main__":
    fg_path = config.fg_path
    bg_path = config.bg_path
    leave_path = config.leave_path
    signatures_path = config.signatures_path
    save_path = config.SAVE_PATH
    
    # Program_1
    Program_1 = utils.fusion_depth(fg_path, bg_path)
    fusion_image, mask, fg_highpass, bg_highpass = Program_1.hipass()
    cv2.imwrite(os.path.join(save_path, "Fusion_depth.jpg"), fusion_image)
    cv2.imwrite(os.path.join(save_path, "mask.jpg"), 255 * mask)
    cv2.imwrite(os.path.join(save_path, "fg_highpass.jpg"), 3 * fg_highpass)
    cv2.imwrite(os.path.join(save_path, "bg_highpass.jpg"), 3 * bg_highpass)

    # Program_2
    Program_2 = utils.simulate_abnormal_vision(fg_path)
    blue_yellow_blind = Program_2.blue_yellow_blind()
    red_green_blind = Program_2.red_green_blind()
    glaucoma = Program_2.glaucoma(sigma=300)

    cv2.imwrite(os.path.join(save_path, "blue_yellow_blind.jpg"), blue_yellow_blind)
    cv2.imwrite(os.path.join(save_path, "red_green_blind.jpg"), red_green_blind)
    cv2.imwrite(os.path.join(save_path, "glaucoma.jpg"), glaucoma)


    # Program_3
    leave_list = []

    for root, dirs, files in os.walk(leave_path):
        for file in files:
            leave_list.append(os.path.join(root, file))
    Program_3 = utils.different_leave(leave_list, signatures_path)
    key, static_image = Program_3.plot_statistics()
    
    for idx in range(len(key)):
        cv2.imwrite(os.path.join(save_path, key[idx] + ".jpg"), static_image[idx])
```

<div style="page-break-after: always"></div>

## Result

### Problem 1

<table>
  <tr>
    <td>fg highpass image</td>
    <td>fg highpass image</td>
    <td>mask</td>
  </tr>
  <tr>
    <td><img src = ".\save_figure\fg_highpass.jpg"></td>
    <td><img src = ".\save_figure\bg_highpass.jpg"></td>
    <td><img src = ".\save_figure\mask.jpg"></td>
  </tr>
</table>

<table>
  <tr>
    <td>Front focusing image</td>
    <td>Back focusing image</td>
    <td>Fuse depth image</td>
  </tr>
  <tr>
    <td><img src = ".\depth of field\5fg.jpg"></td>
    <td><img src = ".\depth of field\5bg.jpg"></td>
    <td><img src = ".\save_figure\Fusion_depth.jpg"></td>
  </tr>
</table>

### Problem 2
<table>
  <tr>
    <td>Original image</td>
    <td>Red Green Blind</td>
    <td>Blue Yellow Blind</td>
    <td>Glaucoma</td>
  </tr>
  <tr>
    <td><img src = ".\depth of field\5bg.jpg"></td>
    <td><img src = ".\save_figure\red_green_blind.jpg"></td>
    <td><img src = ".\save_figure\blue_yellow_blind.jpg"></td>
    <td><img src = ".\save_figure\glaucoma.jpg"></td>
  </tr>
</table>

### Problem 3
<table>
  <tr>
    <td>Red-Lightness axis</td>
    <td>Signature curve-Laplacian Texture axis</td>
  </tr>
  <tr>
    <td><img src = ".\save_figure\Red_Light.jpg" width="280" height="280"></td>
    <td><img src = ".\save_figure\Signature_HighFreq.jpg" width="280" height="280"></td>
  </tr>
</table>