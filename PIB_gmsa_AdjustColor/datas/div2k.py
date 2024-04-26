import os
import glob
import random

import numpy as np
import imageio
import torch.utils.data as data
import skimage.color as sc
import time
from eutils import ndarray2tensor
from .adjustcolor import AdjustColor

def crop_patch(lr,hr,patch_size,scale,augment=True):
    lr_h,lr_w,_ = lr.shape
    hp = patch_size
    lp = patch_size // scale
    lx,ly = random.randrange(0,lr_w-lp+1),random.randrange(0,lr_h-lp+1)
    hx,hy = lx * scale,ly * scale
    lr_patch=lr[ly:(ly+lp),lx:(lx+lp),:]
    hr_patch=hr[hy:(hy+hp),hx:(hx+hp),:]
    
    if augment:
        hflip = random.random()>0.5
        wflip = random.random()>0.5
        rot90 = random.random()>0.5
        if hflip:lr_patch,hr_patch = lr_patch[::-1,:,:],hr_patch[::-1,:,:]
        if wflip:lr_patch,hr_patch = lr_patch[:,::-1,:],hr_patch[:,::-1,:]
        if rot90:lr_patch,hr_patch = lr_patch.transpose(1,0,2),hr_patch.transpose(1,0,2)
    lr_patch,hr_patch = ndarray2tensor(lr_patch),ndarray2tensor(hr_patch)
    return lr_patch,hr_patch

class DIV2K(data.Dataset):
    def __init__(self,HR_folder,LR_folder,CACHE_folder,train=True,augment=True,scale=2,colors=1,patch_size=96,repeat=168) -> None:
        super(DIV2K,self).__init__()
        self.HR_folder=HR_folder
        self.LR_folder=LR_folder
        self.train=train
        self.augment=augment
        self.cache_dir=CACHE_folder
        self.scale=scale
        self.colors=colors
        self.patch_size=patch_size
        self.repeat=repeat
        self.image_postfix='.png'
        self.nums_trainset=0
        
        self.hr_filenames=[]
        self.lr_filenames=[]
        
        self.hr_npy_names=[]
        self.lr_npy_names=[]
        
        self.hr_images=[]
        self.lr_images=[]
        
        if self.train:
            self.start_idx=1
            self.end_idx=801
        else:
            self.start_idx=801
            self.end_idx=901
            
        hr_write_dir = os.path.join(self.HR_folder, 'EC')
        lr_write_dir = os.path.join(self.LR_folder, 'X{}'.format(self.scale), 'EC')
        lr_imgs = os.path.join(self.LR_folder, 'X{}'.format(self.scale))
        AdjustColor(HR_folder, hr_write_dir)
        AdjustColor(lr_imgs, lr_write_dir)
        
        
        for i in range(self.start_idx,self.end_idx):
            idx = str(i).zfill(4)
            hr_filename=os.path.join(hr_write_dir, 'clahe'+idx+self.image_postfix)
            lr_filename=os.path.join(lr_write_dir, 'clahe'+idx+'x{}'.format(self.scale)+self.image_postfix)
            self.hr_filenames.append(hr_filename)
            self.lr_filenames.append(lr_filename)
        self.nums_trainset=len(self.hr_filenames)
        
        LEN = self.end_idx - self.start_idx
        hr_dir = os.path.join(self.cache_dir,'div2k_hr', 'EC','ycbcr' if self.colors==1 else 'rgb')
        lr_dir = os.path.join(self.cache_dir,'div2k_lr_X{}'.format(self.scale), 'EC','ycbcr' if self.colors==1 else 'rgb')
        if not os.path.exists(lr_dir):
            os.makedirs(lr_dir)
        else:
            for i in range(LEN):
                lr_npy = os.path.basename(self.lr_filenames[i]).replace('.png', '.npy')
                lr_npy = os.path.join(lr_dir,lr_npy)
                self.lr_npy_names.append(lr_npy)

        if not os.path.exists(hr_dir):
            os.makedirs(hr_dir)
        else:
            for i in range(LEN):
                hr_npy = os.path.basename(self.hr_filenames[i]).replace('.png', '.npy')
                hr_npy = os.path.join(hr_dir,hr_npy)
                self.hr_npy_names.append(hr_npy)
        
        
        if len(glob.glob(os.path.join(hr_dir,'*.npy'))) < LEN:
            for i in range(LEN):
                if (i+1)%100 == 0:
                    print('convert {} hr image to npy data(RGB)'.format(i+1))
                hr_image = imageio.imread(self.hr_filenames[i],pilmode="RGB")
                if self.colors == 1:
                    hr_image = sc.rgb2ycbcr(hr_image)[:,:,0:1]
                hr_npy = os.path.basename(self.hr_filenames[i]).replace('.png', '.npy')
                hr_npy = os.path.join(hr_dir,hr_npy)
                self.hr_npy_names.append(hr_npy)
                np.save(hr_npy,hr_image)
        else:
            print("hr npy data have already been prepared! hr: {}".format(len(self.hr_filenames)))
        
        if len(glob.glob(os.path.join(lr_dir,'*.npy'))) < LEN:
            for i in range(LEN):
                if (i+1)%100 == 0:
                    print('convert {} lr image to npy data(RGB)'.format(i+1))
                lr_image = imageio.imread(self.lr_filenames[i],pilmode="RGB")
                if self.colors == 1:
                    lr_image = sc.rgb2ycbcr(lr_image)[:,:,0:1]
                lr_npy = os.path.basename(self.lr_filenames[i]).replace('.png', '.npy')
                lr_npy = os.path.join(lr_dir,lr_npy)
                self.lr_npy_names.append(lr_npy)
                np.save(lr_npy,lr_image)
        else:
            print("lr npy data have already been prepared! lr: {}".format(len(self.lr_filenames)))
            
            
    def __len__(self):
        if self.train:
            return self.nums_trainset * self.repeat
        else:
            return self.nums_trainset
    
    def __getitem__(self,idx):
        idx = idx % self.nums_trainset
        
        hr,lr = np.load(self.hr_npy_names[idx]), np.load(self.lr_npy_names[idx])
        if self.train:
            train_lr_patch,train_hr_patch = crop_patch(lr,hr,self.patch_size,self.scale,True)
            return train_lr_patch,train_hr_patch
        else:
            return lr,hr
        
        
        
if __name__=='__main__':
    HR_folder="D:\SR_DataSet\SR_datasets\SR_datasets\DIV2K/DIV2K_train_HR"
    LR_folder="D:\SR_DataSet\SR_datasets\SR_datasets\DIV2K/DIV2K_train_LR_bicubic"
    CACHE_folder="D:\SR_DataSet\SR_datasets\SR_datasets\div2k_cache"
    argment = True
    div2k_test = DIV2K(HR_folder,LR_folder,CACHE_folder,augment=True,scale=2,colors=3,patch_size=96,repeat=168)
    
    
    print('number of samples: {}'.format(len(div2k_test)))
    start = time.time()
    for i in range(5):
        tlr,thr = div2k_test[i]
        vlr,vhr = div2k_test[i]
        print(tlr.shape,thr.shape,vlr.shape,vhr.shape)
    end = time.time()
    print(end-start)
            
                        
                
        
                
                
                
            
            
            
    
        
                    


