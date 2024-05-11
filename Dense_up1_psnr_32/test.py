import argparse, yaml
import numpy as np
from PIL import Image
import cv2 as cv
import os
import torch
import torch.nn as nn
from Module import model
from torchvision.utils import save_image
import torchvision.transforms as transforms

os.environ['PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT'] = '2.0'

parser = argparse.ArgumentParser(description='CLSSAN')
## yaml configuration files

parser.add_argument('--config', type=str, default=None, help = 'pre-config file for training')

if __name__ == '__main__':
    args = parser.parse_args()
    
    out_path = os.path.join(r"./test/")
    
    if args.config:
       opt = vars(args)
       yaml_args = yaml.load(open(args.config), Loader=yaml.FullLoader)
       opt.update(yaml_args)
    ## set visibel gpu   
    gpu_ids_str = str(args.gpu_ids).replace('[','').replace(']','')
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpu_ids_str)
    
    ## select active gpu devices
    device = None
    if args.gpu_ids is not None and torch.cuda.is_available():
        print('use cuda & cudnn for acceleration!')
        print('the gpu id is: {}'.format(args.gpu_ids))
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
    else:
        print('use cpu for training!')
        device = torch.device('cpu')
    torch.set_num_threads(args.threads)

    
    model = model.DisNet(args)    

    model.eval()
    model.to(device)
    
    idx = 17

    # img = Image.open("/opt/data/private/ELAN_total/SR_datasets/DIV2K/DIV2K_test_LR_bicubic/X4/0901x4.png")
    img = Image.open("E:/SR_datasets/DIV2K/DIV2K_test_LR_bicubic/X4/09{}x4.png".format(idx))

    print(type(img))
    img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float()/255.0
    
    save_image(img, os.path.join(out_path, "input{}.png".format(idx)))
    c, h, w = img.shape
    img = torch.reshape(img, [1, c, h, w])
    img = img.to(device)
    print(img.shape)

    if os.path.exists('./train_prior/model_checkpoint17.pth'):
        checkpoint = torch.load('./train_prior/model_checkpoint17.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        
    output = model(img)

    print(output.shape)
    
    # 显示模型的总结信息
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型的参数量: {num_params}")
    save_image(output, os.path.join(out_path, "17output{}.png".format(idx)))
