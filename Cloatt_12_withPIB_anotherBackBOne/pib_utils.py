# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch import nn
from collections import defaultdict
import torch
import numpy as np
from torch.optim import Optimizer
from torch.autograd import grad

from tqdm import tqdm
import os

from train_prior import load_model, save_model
from train_prior import train

import eutils
import math
import time

from torch.utils.tensorboard import SummaryWriter

""" Custom optimizer implementations to track various runtime statistics 
refer to https://github.com/noahgolmant/SGLD/blob/eab60b67ff57b182452bc47dd65d2f58b10aabad/sgld/optimizers.py#L7
"""
class SGLD(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).
    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
        noise_scale (float, optional): variance of isotropic noise for langevin
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf
    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.
        Considering the specific case of Momentum, the update can be written as
        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v
        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.
        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form
        .. math::
             v = \rho * v + lr * g \\
             p = p - v
        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False,
                 noise_scale=0.1):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGLD, self).__init__(params, defaults)
        self.noise_scale = noise_scale

    def __setstate__(self, state):
        super(SGLD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        returns norm of the step we took for variance analysis later
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)
                p.data.add_(np.sqrt(self.noise_scale), torch.randn_like(p.data))
        return loss



def adjust_learning_rate(epoch, optimizer, lr, schedule, decay):
    if epoch in schedule:
        new_lr = lr * decay
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
    else:
        new_lr = lr
    return new_lr



def compute_iiw_bp(model, train_dataloader, param_list, batch_size=1000, no_bp=False):
    '''compute information in weights
    if no_bp set as True, the calculated iiw cannot be used for backward.
    param_list indicates which parameters are used for computing information, e.g.,
    ['extract_feature.0.weight', 'extract_feature.0.bias', 'extract_feature.2.weight', 'extract_feature.2.bias', ...]
    '''
    all_model_param_key = [p[0] for p in model.named_parameters()]
    if param_list is None:
        param_list = [p[0] for p in model.named_parameters() if 'weight' in p[0]]
    else:
        # check param list
        for param in param_list:
            if param not in all_model_param_key:
                raise RuntimeError('{} is not found in model parameters!'.format(param))

    delta_w_dict = dict().fromkeys(param_list)
    for pa in model.named_parameters():
        if pa[0] in param_list:
            w0 = model.w0_dict[pa[0]]
            delta_w = pa[1] - w0
            delta_w_dict[pa[0]] = delta_w

    # 防止模型参数无法计算梯度（修改）
    param_ts = []
    
    # 遍历模型参数，并将需要计算梯度的参数添加到 param_ts 列表
    for param_name, param in model.named_parameters():
        if param_name in param_list:
            param.requires_grad = True  # 确保需要计算梯度
            param_ts.append(param)

    # 确保模型处于训练模式
    model.train()
    info_dict = dict()
    gw_dict = dict().fromkeys(param_list)
    Loss_func = nn.L1Loss()
    for _ in range(10):
        # 创建 DataLoader 的迭代器
        data_iter = iter(train_dataloader)

        # DataLoader 中随机取一组数
        random_batch = next(data_iter)
        x_batch, y_batch = random_batch
        x_batch = x_batch.to('cuda')
        y_batch = y_batch.to('cuda')
        pred = model.forward(x_batch)
        
        loss = Loss_func(pred, y_batch)

        gradients = grad(loss, param_ts)        
        for i, gw in enumerate(gradients):
            gw_ = gw.flatten()
            if gw_dict[param_list[i]] is None:
                gw_dict[param_list[i]] = gw_
            else:
                gw_dict[param_list[i]] += gw_
    
    for k in gw_dict.keys():
        num_all_batch = len(train_dataloader)
        gw_dict[k] *= 1/num_all_batch
        delta_w = delta_w_dict[k]
        # delta_w.T gw @ gw.T delta_w = (delta_w.T gw)^2
        info_ = (delta_w.flatten() * gw_dict[k]).sum() ** 2
        if no_bp:
            info_dict[k] = info_.item()
        else:
            info_dict[k] = info_

    return info_dict

def train_iiw(model,train_dataloader,eval_dataloader,
    optimizer,
    loss_func,
    scheduler,
    logger,
    param_list=None,
    scale=4,
    colors=3,
    device='cuda',
    num_epoch=100,
    batch_size=8,
    learn_rate=1e-4,
    weight_decay=0,
    beta=1e-1,
    early_stop_tolerance=10,
    schedule = [50, 80, 100],
    gamma = 0.1,
    noise_scale=1e-10,
    pretrain_step=500,
    verbose=False,
    ):
    '''train model with iiw regularization
    param_list contains the list of parameters which are used to
    compute iiw and regularization. if set None, all parameters
    will be used.
    '''
    # pre-train
    train(model,train_dataloader,eval_dataloader,
        optimizer,
        loss_func,
        scheduler,
        logger=logger,
        num_epoch=pretrain_step, 
        batch_size=batch_size,
        scale=scale,
        colors=colors,
        verbose=True,
        device=device,
        train_prior=True)

    # early stop
    num_all_train = 0
    if early_stop_tolerance < 0:
        early_stop_tolerance = num_epoch

    info_dict = defaultdict(list)
    loss_acc_dict = defaultdict(list)

    # init training with the SGLD optimizer
    optimizer = SGLD(params=filter(lambda p: p.requires_grad, model.parameters()), 
                    lr=learn_rate,
                    momentum=0.9,
                    weight_decay=weight_decay,
                    noise_scale=noise_scale)



    # initialize log p(w) at the first epoch
    energy_decay = 0
    num_all_tr_batch = len(train_dataloader)
    
    start_epoch = 0
    if os.path.exists('./train_iiw/model_checkpoint.pth'):
        checkpoint = torch.load('./train_iiw/model_checkpoint.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        info_dict = checkpoint['info']
        loss_acc_dict = checkpoint['loss_acc']
        start_epoch = checkpoint['epoch']
        
    writer = SummaryWriter("logs_pib")
    
    # print('##==========={}-training=============##'.format('pib'))
    logger.info('##==========={}-training=============##'.format('pib'))
    for epoch in range(start_epoch, num_epoch):
        total_loss = 0
        model.train()

        # adjust learning rate
        learn_rate = adjust_learning_rate(epoch, optimizer, learn_rate, schedule, gamma)
        timer_start = time.time()
        for iter, batch in enumerate(train_dataloader):
            x_batch,y_batch = batch
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            pred = model(x_batch)

            loss = loss_func(pred,y_batch)
            avg_loss = torch.mean(loss)

            optimizer.zero_grad()

            if epoch > 0:
                energy_decay.backward(retain_graph=True)
                avg_loss.backward()
            else:
                avg_loss.backward()

            optimizer.step()
            num_all_train += len(x_batch)
            total_loss = total_loss + avg_loss.item()
            if (iter + 1) % 10 == 0:
                cur_steps = (iter+1)*batch_size
                total_steps = len(train_dataloader.dataset)
                fill_width = math.ceil(math.log10(total_steps))
                cur_steps = str(cur_steps).zfill(fill_width)
                epoch_width = math.ceil(math.log10(num_epoch))
                cur_epoch = str(epoch).zfill(epoch_width)
                avg_loss = total_loss / (iter + 1)
                timer_end = time.time()
                duration = timer_end - timer_start
                timer_start = timer_end
                # print('Epoch:{}, {}/{}, loss: {:.4f}, time: {:.3f}, lr: {}'.format(cur_epoch, cur_steps, total_steps, avg_loss, duration, learn_rate))
                logger.info('Epoch:{}, {}/{}, loss: {:.4f}, time: {:.3f}, lr: {}'.format(cur_epoch, cur_steps, total_steps, avg_loss, duration, learn_rate))
                writer.add_scalar("loss_pib_epoch{}".format(epoch), avg_loss, iter)
        
        # compute the information regularization term
        info = compute_iiw_bp(model,train_dataloader, param_list, no_bp=False)
        writer.add_scalars("info", info, epoch)
        energy_decay = 0
        for k in info.keys():
            # plus decay term for each weight
            energy_decay += info[k]
            info_dict[k].append(info[k].item())
        
        if verbose:
            # print("epoch: {}, info: {}".format(epoch, info))
            logger.info("epoch: {}, info: {}".format(epoch, info))
            # print("epoch: {}, tr loss: {}, lr: {}, e_decay: {}".format(epoch, total_loss/num_all_tr_batch, learn_rate, energy_decay))
            logger.info("epoch: {}, tr loss: {}, lr: {}, e_decay: {}".format(epoch, total_loss/num_all_tr_batch, learn_rate, energy_decay))
        
        energy_decay = beta * energy_decay

        model.eval()
        acc_tr=0
        for iter, batch in enumerate(train_dataloader):
            x_batch,y_batch = batch
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            with torch.no_grad():
                pred = model(x_batch)
            y_batch = y_batch.clamp(0, 255)
            pred = pred.clamp(0, 255)
            # conver to ycbcr
            if colors == 3:
                hr_ycbcr = eutils.rgb_to_ycbcr(y_batch)
                sr_ycbcr = eutils.rgb_to_ycbcr(pred)
                hr = hr_ycbcr[:, 0:1, :, :]
                sr = sr_ycbcr[:, 0:1, :, :]
            # crop image for evaluation
            hr = hr[:, :, scale:-scale, scale:-scale]
            sr = sr[:, :, scale:-scale, scale:-scale]
            acc_tr += eutils.calc_psnr(sr, hr)
        acc_tr = acc_tr/(num_all_tr_batch)
        print("loader: 'train_dataloader', psnr: {}".format(acc_tr))
        loss_acc_dict["tr_loss"].append((total_loss/num_all_tr_batch))
        loss_acc_dict["tr_acc"].append(acc_tr)

        if eval_dataloader is not None:
            # evaluate on va set
            model.eval()
            acc_va=0
            
            for valid_dataloader in eval_dataloader:
                avg_psnr, avg_ssim = 0.0, 0.0
                name = valid_dataloader['name']
                loader = valid_dataloader['dataloader']
                for lr, hr in tqdm(loader, ncols=80):
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
            if verbose:
                print("epoch: {}, va acc: {}".format(epoch, acc_va))
            loss_acc_dict["va_acc"].append(acc_va)
        writer.add_scalars("loss_acc", loss_acc_dict, epoch)
        checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'info': info_dict,
                'loss_acc': loss_acc_dict
            }
        torch.save(checkpoint, './train_iiw/model_checkpoint{}.pth'.format(epoch))
        torch.save(checkpoint, './train_iiw/model_checkpoint.pth')
    writer.close()
    return info_dict, loss_acc_dict
