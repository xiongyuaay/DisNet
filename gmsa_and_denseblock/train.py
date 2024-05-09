import argparse, yaml
import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from datas.utils import create_datasets
from pib_utils import train_iiw
from Module import dis_network
from train_prior import plot_info_acc
from eutils import get_logger

os.environ['PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT'] = '2.0'

parser = argparse.ArgumentParser(description='DisNet')
## yaml configuration files

parser.add_argument('--config', type=str, default=None, help = 'pre-config file for training')
if __name__ == '__main__':
    args = parser.parse_args()
    
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

    logger = get_logger('./logs/withPIB_residualresidualblock2gmsa.log')
    
    ## create dataset for training and validating
    train_dataloader, valid_dataloaders = create_datasets(args)
    
    model = dis_network.DisNet(args)

    # logger.info(model)
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
