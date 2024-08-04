# -*- coding: utf-8 -*-
import torch
import os
import eutils
import math
import time
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import radialProfile


N = 179
epsilon = 1e-8
lambda_freq = 1e-5 
criterion_freq = torch.nn.BCELoss()
def RGB2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def train(model,
    train_dataloader,
    eval_dataloader,
    optimizer,
    loss_func,
    scheduler,
    logger,
    num_epoch,
    batch_size,
    scale,
    colors=3,
    verbose=True,
    device='cuda',
    train_prior=False
    ):
    """Given selected subset, train the model until converge.
    """
    # early stop
    best_va_acc = 0
    start_epoch = 0
    if os.path.exists('./train_prior/model_checkpoint.pth') or os.path.exists('./train/model_checkpoint.pth'):
        if train_prior:
            checkpoint = torch.load('./train_prior/model_checkpoint.pth')
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
        else:
            checkpoint = torch.load('./train/model_checkpoint.pth')
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            
    writer = SummaryWriter("logs_pre")
    
    for epoch in range(start_epoch,num_epoch):
        epoch_loss = 0
        model.train()
        opt_lr = scheduler.get_last_lr()
        if train_prior:
            print('##==========={}-training,  lr: {} =============##'.format('prior', opt_lr))
        else:
            # print('##==========={}, {}-training,  lr: {} =============##'.format('PCB', 'pre', opt_lr))
            logger.info('##==========={}, {}-training,  lr: {} =============##'.format('PCB', 'pre', opt_lr))
        timer_start = time.time()
        for iter, batch in enumerate(train_dataloader):
            lr, hr = batch
            lr, hr = lr.to(device), hr.to(device)
            sr = model(lr)
            loss = loss_func(sr,hr)

            
            sum_loss = torch.sum(loss)
            epoch_loss = epoch_loss + sum_loss.detach()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (iter + 1) % 10 == 0:
                cur_steps = (iter+1)*batch_size
                total_steps = len(train_dataloader.dataset)
                fill_width = math.ceil(math.log10(total_steps))
                cur_steps = str(cur_steps).zfill(fill_width)
                epoch_width = math.ceil(math.log10(num_epoch))
                cur_epoch = str(epoch).zfill(epoch_width)
                avg_loss = epoch_loss / (iter + 1)
                timer_end = time.time()
                duration = timer_end - timer_start
                timer_start = timer_end
                writer.add_scalar("loss_epoch{}".format(epoch), avg_loss, iter)
                # print('Epoch:{}, {}/{}, loss: {:.4f}, time: {:.3f}, lr: {}'.format(cur_epoch, cur_steps, total_steps, avg_loss, duration, opt_lr))
                logger.info('Epoch:{}, {}/{}, loss: {:.4f}, time: {:.3f}, lr: {}'.format(cur_epoch, cur_steps, total_steps, avg_loss, duration, opt_lr))

        if eval_dataloader is not None:
            # evaluate on va set
            model.eval()
            acc_va=0
            count=0
            for valid_dataloader in eval_dataloader:
                avg_psnr, avg_ssim = 0.0, 0.0
                name = valid_dataloader['name']
                loader = valid_dataloader['dataloader']
                for lr, hr in loader:
                    lr, hr = lr.to(device), hr.to(device)
                    sr = model(lr)
                    # quantize output to [0, 255]
                    hr = hr.clamp(0, 255)
                    sr = sr.clamp(0, 255)
                    # conver to ycbcr
                    if colors == 3:
                        hr_ycbcr = eutils.rgb_to_ycbcr(hr)
                        sr_ycbcr = eutils.rgb_to_ycbcr(sr)
                        hr = hr_ycbcr[:, 0:1, :, :]
                        sr = sr_ycbcr[:, 0:1, :, :]
                    # crop image for evaluation
                    hr = hr[:, :, scale:-scale, scale:-scale]
                    sr = sr[:, :, scale:-scale, scale:-scale]
                    # calculate psnr and ssim
                    psnr = eutils.calc_psnr(sr, hr)
                    ssim = eutils.calc_ssim(sr, hr)
                    
                    count += 1
                    avg_psnr += psnr
                    avg_ssim += ssim
                avg_psnr = round(avg_psnr/len(loader) + 5e-3, 2)
                avg_ssim = round(avg_ssim/len(loader) + 5e-5, 4)
                writer.add_scalar("psnr_{}".format(name), avg_psnr, epoch)
                writer.add_scalar("ssim_{}".format(name), avg_ssim, epoch)
                # print("loader: {}, psnr/ssim: {}/{}".format(name,avg_psnr,avg_ssim))
                logger.info("loader: {}, psnr/ssim: {}/{}".format(name,avg_psnr,avg_ssim))
                acc_va+=avg_psnr
            
            acc_va=acc_va/len(eval_dataloader)
            # print("epoch: {}, acc: {}".format(epoch, acc_va))
            logger.info("epoch: {}, acc: {}".format(epoch, acc_va))
            
           
        ## update scheduler
        scheduler.step()
        if train_prior:
            prior_checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            torch.save(prior_checkpoint, './train_prior/model_checkpoint{}.pth'.format(epoch))
            torch.save(prior_checkpoint, './train_prior/model_checkpoint.pth')
        else:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            torch.save(checkpoint, './train/model_checkpoint{}.pth'.format(epoch))
            torch.save(checkpoint, './train/model_checkpoint.pth')
    
    w0_dict = dict()
    for param in model.named_parameters():
        w0_dict[param[0]] = param[1].clone().detach() # detach but still on gpu
    model.w0_dict = w0_dict
    print("done get prior weights")
    writer.close()
    return best_va_acc

    
    
def save_model(ckpt_path, model):
    torch.save(model.state_dict(), ckpt_path)
    return

def load_model(ckpt_path, model):
    try:
        model.load_state_dict(torch.load(ckpt_path))
    except:
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))

    return
def plot_info_acc(info_dict, loss_acc_list, act, fig_dir='./figure'):
    df_info = pd.DataFrame(info_dict)
    fig, axs = plt.subplots(2, 1, figsize=(6,8))
    for i,col in enumerate(df_info.columns):
        axs[0].plot(df_info[col], label=col, lw=2)
    axs[0].set_xlabel('epoch', size=24)
    axs[0].set_ylabel('IIW',size=24)
    axs[0].tick_params(labelsize=20)
    axs[0].set_title('IIW of {} MLP'.format(act), size=20)
    axs[0].legend(fontsize=24)

    # plot loss acc
    ax1 = axs[1]
    ax2 = ax1.twinx()
    lns = ax2.plot(loss_acc_list['va_acc'], label='test acc', lw=2)
    ax1.set_xlabel('epoch', size=24)
    ax1.set_ylabel('loss', size=24)
    ax2.set_ylabel('acc', size=24)
    ax1.tick_params(labelsize=20)
    ax2.tick_params(labelsize=20)
    ax1.set_ylim(0.3,2.5)
    ax2.set_ylim(0.5,0.8)
    ax1.set_yticks([0.5, 1.0, 1.5, 2.0, 2.5])
    ax2.set_yticks([0.5,0.6,0.7,0.8])
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, fontsize=24)
    plt.tight_layout()


    plt.savefig(os.path.join(fig_dir,"{}_acc_loss.png".format(act)),bbox_inches = 'tight')
    plt.show()


def plot_info(info_dict, fig_dir='./figure', use_legend=True):
    '''specifically used for plot jupyter notebook.
    '''
    df_info = pd.DataFrame(info_dict)
    fig, axs = plt.subplots(figsize=(6,4))
    for i,col in enumerate(df_info.columns):
        axs.plot(df_info[col], label=col, lw=2)
    axs.set_xlabel('iteration', size=28)
    axs.set_ylabel('IIW',size=28)
    axs.tick_params(labelsize=24)
    axs.yaxis.get_major_formatter().set_powerlimits((0,1))
    axs.set_title('IIW of {}-layer MLP'.format(int(len(df_info.columns))), size=28)
    if use_legend:
        axs.legend(fontsize=26)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir,"mlp_{}_info.pdf".format(int(len(df_info.columns)))),bbox_inches = 'tight')
    plt.show()
