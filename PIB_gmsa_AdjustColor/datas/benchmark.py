import os
import imageio
import torch.utils.data as data
import skimage.color as sc
from eutils import ndarray2tensor
from .adjustcolor import AdjustColor

class Benchmark(data.Dataset):
    def __init__(self, HR_folder, LR_folder, scale=2, colors=1):
        super(Benchmark, self).__init__()
        self.HR_folder = HR_folder
        self.LR_folder = LR_folder

        self.img_postfix = '.png'
        self.scale = scale
        self.colors = colors

        self.nums_dataset = 0

        self.hr_filenames = []
        self.lr_filenames = []

        # adjust color
        hr_write_dir = os.path.join(self.HR_folder, 'EC')
        lr_write_dir = os.path.join(self.LR_folder, 'X{}'.format(self.scale), 'EC')
        lr_img = os.path.join(self.LR_folder, 'X{}'.format(self.scale))
        AdjustColor(HR_folder, hr_write_dir)
        AdjustColor(lr_img, lr_write_dir)
        
        ## generate dataset
        tags = os.listdir(hr_write_dir)
        for tag in tags:
            hr_filename = os.path.join(hr_write_dir, tag)
            lr_filename = os.path.join(lr_write_dir, tag.replace('.png', 'x{}.png'.format(self.scale)))
            self.hr_filenames.append(hr_filename)
            self.lr_filenames.append(lr_filename)
        self.nums_trainset = len(self.hr_filenames)
        ## if store in ram
        self.hr_images = []
        self.lr_images = []

        LEN = len(self.hr_filenames)
        for i in range(LEN):
            lr_image, hr_image = imageio.imread(self.lr_filenames[i], pilmode="RGB"), imageio.imread(self.hr_filenames[i], pilmode="RGB")
            if self.colors == 1:
                lr_image, hr_image = sc.rgb2ycbcr(lr_image)[:, :, 0:1], sc.rgb2ycbcr(hr_image)[:, :, 0:1]
            self.hr_images.append(hr_image)
            self.lr_images.append(lr_image) 

    def __len__(self):
        return len(self.hr_filenames)
    
    def __getitem__(self, idx):
        # get whole image, store in ram by default
        lr, hr = self.lr_images[idx], self.hr_images[idx]
        lr_h, lr_w, _ = lr.shape
        hr = hr[0:lr_h*self.scale, 0:lr_w*self.scale, :]
        lr, hr = ndarray2tensor(lr), ndarray2tensor(hr)
        return lr, hr

