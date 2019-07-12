from __future__ import print_function
import argparse
import os
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from networks import define_G, define_D, GANLoss, get_scheduler, update_learning_rate
from data import get_training_set, get_test_set
from tensorboardX import SummaryWriter
import numpy as np
import cv2
from utils import _gradient_penalty

# Training settings
parser = argparse.ArgumentParser(description='WEGAN')
parser.add_argument('--dataset', type=str, default='MSRA10K', help='which dataset')
parser.add_argument('--dataset_dir', type=str, default='./datasets/', help='location of datasets')
parser.add_argument('--visual_effect', type=str, default='color-selectivo', help='black-background, color-selectivo, defocus')
parser.add_argument('--batch_size', type=int, default=16, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=16, help='testing batch size')
parser.add_argument('--image_size', type=int, default=224, help='training batch size')
parser.add_argument('--input_nc', type=int, default=3, help='input image channels')
parser.add_argument('--output_nc', type=int, default=1, help='output mask channels')
parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
parser.add_argument('--ndf', type=int, default=64, help='discriminator filters in first conv layer')
parser.add_argument('--loss', type=str, default='wgan-gp', help='lsgan or wgan-gp')
parser.add_argument('--dis', type=str, default='patch', help='use patch/pixel gan architecture')
parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count')
parser.add_argument('--niter', type=int, default=50, help='# of iter at starting learning rate')
parser.add_argument('--niter_decay', type=int, default=150, help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
opt = parser.parse_args()

print(opt)
opt.cuda = True

if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

cudnn.benchmark = True

torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
train_set = get_training_set(opt.dataset, opt.dataset_dir, opt.visual_effect)
test_set = get_test_set(opt.dataset, opt.dataset_dir, opt.visual_effect)

training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.test_batch_size, shuffle=False)

device = torch.device("cuda:0" if opt.cuda else "cpu")

print('===> Building models')
net_g = define_G(opt.input_nc, opt.output_nc, opt.ngf, 'instance', False, 'normal', 0.02, gpu_id=device)
if opt.loss == 'wgan-gp':
    net_d = define_D(opt.input_nc, opt.ndf, opt.dis, gpu_id=device)
elif opt.loss == 'lsgan':
    net_d = define_D(opt.input_nc, opt.ndf, opt.dis, gpu_id=device)
else:
    print('wrong input')
    assert False

criterionGAN = GANLoss().to(device)
criterionL1 = nn.L1Loss().to(device)
criterionMSE = nn.MSELoss().to(device)

# setup optimizer
optimizer_g = optim.Adam(net_g.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizer_d = optim.Adam(net_d.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
net_g_scheduler = get_scheduler(optimizer_g, opt)
net_d_scheduler = get_scheduler(optimizer_d, opt)

print('===> saving directory')
path_name = '{}_{}_{}_{}'.format(opt.dataset, opt.dis, opt.loss, opt.visual_effect)
step = 0
writer = SummaryWriter('runs/{}/'.format(path_name))
image_path = 'img/{}'.format(path_name)
if not os.path.isdir(image_path):
    os.mkdir(image_path)


for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):

    
    # train
    for iteration, batch in enumerate(training_data_loader, 1):
        # forward
        real, back, effect = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        mask = net_g(real)

        fake = (mask * real) + ((1-mask) * back)
        fake = torch.clamp(fake, min=-1, max=1)

        ######################
        # (1) Update D network
        ######################
        optimizer_d.zero_grad()
        
        # train with fake
        pred_fake = net_d.forward(fake.detach())
        
        # train with real
        pred_real = net_d.forward(effect)
        
        # Combined D loss
        if opt.loss == 'wgan-gp':
            gradient_penalty = _gradient_penalty(net_d, effect, fake, opt.cuda)
            loss_d = pred_fake.mean() - pred_real.mean() + gradient_penalty
        else:
            loss_d_fake = criterionGAN(pred_fake, False)
            loss_d_real = criterionGAN(pred_real, True)
            loss_d = (loss_d_fake + loss_d_real) * 0.5

        loss_d.backward()
    
        optimizer_d.step()

        ######################
        # (2) Update G network
        ######################

        optimizer_g.zero_grad()

        # G(A) should fake the discriminator
        pred_fake = net_d.forward(fake)
        if opt.loss == 'wgan-gp':
            loss_g = - pred_fake.mean()
        else:
            loss_g = criterionGAN(pred_fake, True)

        
        loss_g.backward()

        optimizer_g.step()

        print("===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f}".format(
            epoch, iteration, len(training_data_loader), loss_d.item(), loss_g.item()))
        
        info = {'loss_G': loss_g, 'loss_D': loss_d}
    
        writer.add_scalars("losses", info, (step))
        step += 1

    
    # test
    if epoch%10 == 1:
        for iteration, batch in enumerate(testing_data_loader, 1):
            with torch.no_grad():
                image, back, effect = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                mask = net_g(image)
                mask = torch.cat((mask, mask, mask), 1)

                fake = (mask * image) + ((1-mask) * back)
                fake = torch.clamp(fake, min=-1, max=1)

                combine = torch.cat((image, mask, fake, effect), dim=3, out=None)
                combine = combine.data.cpu().numpy()
                combine = (combine + 1) * 128
                combine = np.transpose(combine, (0,2,3,1))[:4]
                combine = np.reshape(combine, (-1, opt.image_size*4, 3))
                print("===> Testing:[{}]({}/{})".format(epoch, iteration, len(testing_data_loader)))

                cv2.imwrite('{}/{}_{}.jpg'.format(image_path, epoch, iteration),combine)


    update_learning_rate(net_g_scheduler, optimizer_g)
    update_learning_rate(net_d_scheduler, optimizer_d)

    
    


    #checkpoint
    if epoch % 50 == 0:
        ckpt_path = os.path.join("checkpoint", path_name)
        if not os.path.exists("checkpoint"):
            os.mkdir("checkpoint")
        if not os.path.exists(ckpt_path):
            os.mkdir(ckpt_path)
        
        net_g_model_out_path = "{}/netG_model_epoch_{}.pth".format(ckpt_path, epoch)
        net_d_model_out_path = "{}/netD_model_epoch_{}.pth".format(ckpt_path, epoch)
        torch.save(net_g, net_g_model_out_path)
        torch.save(net_d, net_d_model_out_path)
        print("Checkpoint saved to {}".format(ckpt_path))
