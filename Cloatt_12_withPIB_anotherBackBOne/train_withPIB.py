import argparse, yaml
import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from datas.utils import create_datasets
from pib_utils import train_iiw
from Module import CLSSAN, model
from train_prior import plot_info_acc, plot_info
from eutils import get_logger

os.environ['PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT'] = '2.0'

parser = argparse.ArgumentParser(description='CLSSAN')
## yaml configuration files

parser.add_argument('--config', type=str, default=None, help = 'pre-config file for training')
if __name__ == '__main__':
    args = parser.parse_args()
    
    ## parameters for plain
    args.scale=4
    args.rgb_range= 255
    args.colors= 3
    args.Module2_depth= 12
    args.channels= 60
    args.r_expand= 2
    args.window_sizes= [4, 8, 16]

    ## parameters for model training
    args.patch_size = 256
    args.batch_size = 16
    args.data_repeat = 80
    args.data_augment = 1

    args.epochs = 107
    args.lr = 0.0002
    args.decays = [50, 100, 120, 150, 200]
    args.gamma = 0.5

    ## hardware specification
    args.gpu_ids= [0]
    args.threads= 8

    ## data specification
    args.data_path= '/opt/data/private/ELAN_total/SR_datasets'
    # args.data_path= 'E:/SR_datasets'
    # args.eval_sets= ['Set5', 'Set14', 'B100', 'Urban100', 'Manga109']
    args.eval_sets= ['Set5', 'Set14', 'B100']
    # args.eval_sets= ['Set5']
    
    
    __fig_dir__ = './figure'
    if not os.path.exists(__fig_dir__):
        os.makedirs(__fig_dir__)
    
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

    logger = get_logger('./logs/newBackbone.log')
    
    ## create dataset for training and validating
    train_dataloader, valid_dataloaders = create_datasets(args)
    
    model = model.Generator(args)

    logger.info(model)
    ## definition of loss and optimizer
    loss_func = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=args.decays, gamma=args.gamma)
    model.to(device)
    
    # train with iiw regularization!
    param_list = None
    info_dict, loss_acc_list = train_iiw(model, train_dataloader, valid_dataloaders,
                                    optimizer,
                                    loss_func,
                                    scheduler,
                                    logger=logger,
                                    param_list=param_list,
                                    scale=args.scale,colors=args.colors,
                                    device=device,
                                    num_epoch=args.epochs,
                                    batch_size=args.batch_size,
                                    learn_rate=args.lr,
                                    verbose=True)
    
    
    plot_info_acc(info_dict, loss_acc_list, "linear", __fig_dir__)
    plot_info(info_dict)
