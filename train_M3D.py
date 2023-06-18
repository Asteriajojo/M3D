import argparse
import os
import numpy as np
from PIL import Image
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from generators import *
from gaussian_smoothing import *
from glob import glob
from utils import TwoCropTransform, rotation
from torch import distributed as dist
from utils import target_dict
import copy
import logging


logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(
    description='M3D')
parser.add_argument('--src', default='../data/image_random_50k',
                    help='Source Domain: natural images, paintings, medical scans, etc')
parser.add_argument('--match_target', type=int, default=245,
                    help='Target Domain samples')
parser.add_argument('--match_dir', default='../data/imagenet_train',
                    help='Path to data folder with target domain samples')
parser.add_argument('--batch_size', type=int, default=16,
                    help='Number of trainig samples/batch')
parser.add_argument('--epochs', type=int, default=20,
                    help='Number of training epochs')
parser.add_argument('--lr', type=float, default=0.0002,
                    help='Initial learning rate for adam')
parser.add_argument('--eps', type=int, default=10,
                    help='Perturbation Budget during training, eps')
parser.add_argument('--model_type', type=str, default='resnet101',
                    help='Model under attack (discrimnator)')
parser.add_argument('--gs', action='store_true',
                    help='Apply gaussian smoothing')
parser.add_argument('--save_dir', type=str,
                    default='./checkpoint', help='Directory to save generators')
parser.add_argument('--log_dir', type=str,
                    default='./logs', help='Directory to save generators')
parser.add_argument('--local_rank', default=-1, type=int)
parser.add_argument('--apex_train', default=0, type=int)
args = parser.parse_args()
print(args)


#################
# initialize
#################
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)
##################


##################### Basic information ####################################
if not os.path.isdir(args.save_dir):
    os.mkdir(args.save_dir)

eps = args.eps / 255
logfile = os.path.join(args.log_dir, 'attack_{0}_to_{1}.log'.format(args.model_type, args.match_target))
logging.basicConfig( format='[%(asctime)s] - %(message)s', datefmt='%Y/%m/%d %H:%M:%S', level=logging.INFO, filename=logfile)
logger.info('*'*100)

##################### GPU infomation ####################################
print("num of gpu:", torch.cuda.device_count())
torch.distributed.init_process_group(backend="nccl")
print("***********")
print('world_size', torch.distributed.get_world_size())
torch.cuda.set_device(args.local_rank)

##################### Create Generator Model  ####################################    
# Input dimensions
if args.model_type == 'inception_v3':
    scale_size, img_size = 300, 299
    netG = GeneratorResnet(inception=True)
else:
    scale_size, img_size = 256, 224
    netG = GeneratorResnet()

netG = netG.cuda(args.local_rank)
optimG = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

if args.model_type in model_names:
    model = models.__dict__[args.model_type](pretrained=True)
else:
    assert (args.model_type in model_names), 'Please provide correct target model names: {}'.format(model_names)
model = model.cuda(args.local_rank)

# Since the training data input to D1 and D2 are the same, 
# the two models need to be initialized slightly different to ensure the model discrepancy loss works. 
# We use finetuned model for convenience here. 
# You can also finetune the discriminator during the training period.

if args.model_type == 'resnet50':
    model_m3d=torch.load('./pretrain_save_models/model_resnet50.pth',map_location='cpu')
if args.model_type == 'densenet121':
    model_m3d=torch.load('./pretrain_save_models/model_densenet121.pth',map_location='cpu')
if args.model_type == 'vgg19_bn':
    model_m3d=torch.load('./pretrain_save_models/model_vgg19_bn.pth',map_location='cpu')
model_m3d = model_m3d.cuda(args.local_rank)

optimD = optim.SGD(list(model_m3d.parameters()) + list(model.parameters()), momentum=0.9, lr=0.0001, weight_decay=5e-4)

if args.apex_train:
    try:
        from apex import amp
        print("Using fp16 for faster training!")
    except:
        exit(0)
    netG, optimG = amp.initialize(netG, optimG, opt_level="O1")
    (model, model_m3d), optimD = amp.initialize([model, model_m3d], optimD, opt_level="O1")

netG = torch.nn.SyncBatchNorm.convert_sync_batchnorm(netG)
netG = torch.nn.parallel.DistributedDataParallel(
    netG,
    device_ids=[args.local_rank],
    output_device=args.local_rank,
    find_unused_parameters=False,
    broadcast_buffers=False
)

##################### utils for training  ####################################
train_transform = transforms.Compose([
    transforms.Resize(scale_size),
    transforms.CenterCrop(img_size),
    transforms.ToTensor()
])

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def normalize(_t):
    t = _t + 0.
    t[:, 0, :, :] = (t[:, 0, :, :] - mean[0]) / std[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean[1]) / std[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean[2]) / std[2]
    return t

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


if args.gs:
    kernel_size = 3
    pad = 2
    sigma = 1
    kernel = get_gaussian_kernel(
        kernel_size=kernel_size, pad=pad, sigma=sigma).cuda()

##################### information for dataset  ####################################

### training data
source_samples_train = [] 
train_set = torchvision.datasets.ImageFolder(args.src, TwoCropTransform(train_transform, img_size))
for img_name, label in train_set.samples:
    if label != args.match_target:
        source_samples_train.append((img_name, label))
train_set.samples = source_samples_train
train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                           pin_memory=True,sampler=train_sampler,drop_last=True)

# attack_size = len(train_set)


# clean data
train_all = torchvision.datasets.ImageFolder(
    args.match_dir, train_transform)
dataset_all_sampler = torch.utils.data.distributed.DistributedSampler(
    train_all)
train_loader_all = torch.utils.data.DataLoader(train_all, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                               pin_memory=True, drop_last=True, sampler=dataset_all_sampler)
dataiter_all = iter(train_loader_all)

############################### Run  ####################################
def load_data(data_iter):
    try:
        data = next(data_iter)
        samples, labels = data[0], data[1]

    except StopIteration:
        data_iter = iter(train_loader_all)
        data = next(data_iter)
        samples, labels = data[0], data[1]

    samples = samples.cuda(args.local_rank)
    labels = labels.cuda(args.local_rank)
    return samples, labels, data_iter
    
for epoch in range(args.epochs):
    
    train_sampler.set_epoch(epoch)
    dataset_all_sampler.set_epoch(epoch)

    loss_map = {
        'running_D_loss': 0,
        'running_loss': 0,
        'running_loss_f': 0,
        'running_loss_f2':0,
        'total': 0
    }

    for i, (imgs, _) in enumerate(train_loader):
        img = imgs[0].cuda(args.local_rank, non_blocking=True)
        img_rot = rotation(img)[0]
        img_aug = imgs[1].cuda(args.local_rank, non_blocking=True)
        target_label=torch.tensor(args.match_target).expand(args.batch_size)
        target_label=target_label.cuda(args.local_rank, non_blocking=True)
        img_all, img_all_label,dataiter_all = load_data(dataiter_all)


        model.eval()
        model_m3d.eval()
        for step in ['optim_G','optim_G','optim_G','optim_G', 'optim_D']:
            if step == 'optim_D':
                netG.eval()
            else:
                netG.train()

            loss_D, loss = torch.tensor(0), torch.tensor(0)
           
            if step == 'optim_D':

                ## ################### ATTACK #########################
                adv = netG(img)
                adv_rot = netG(img_rot)
                adv_aug = netG(img_aug)

                # Smoothing
                if args.gs:
                    adv = kernel(adv)
                    adv_rot = kernel(adv_rot)
                    adv_aug = kernel(adv_aug)

                # Projection
                adv = torch.min(torch.max(adv, img - eps), img + eps)
                adv = torch.clamp(adv, 0.0, 1.0)
                adv_rot = torch.min(torch.max(adv_rot, img_rot - eps), img_rot + eps)
                adv_rot = torch.clamp(adv_rot, 0.0, 1.0)
                adv_aug = torch.min(torch.max(adv_aug, img_aug - eps), img_aug + eps)
                adv_aug = torch.clamp(adv_aug, 0.0, 1.0)

                ## ################### RUN #########################

                adv_out = model(normalize(adv.detach()))
                adv_rot_out = model(normalize(adv_rot.detach()))
                adv_aug_out = model(normalize(adv_aug.detach()))
                img_all_out = model(normalize(img_all))

                adv_out_mcd = model_m3d(normalize(adv.detach()))
                adv_rot_out_mcd = model_m3d(normalize(adv_rot.detach()))
                adv_aug_out_mcd = model_m3d(normalize(adv_aug.detach()))
                img_all_out_mcd = model_m3d(normalize(img_all))                

                loss_L1=0.0
                for out, out_mcd in [[adv_out, adv_out_mcd], [adv_rot_out, adv_rot_out_mcd], [adv_aug_out, adv_aug_out_mcd]]:
                    
                    loss_L1 +=torch.mean(torch.abs(F.softmax(out, dim=1) - F.softmax(out_mcd, dim=1)))
                
                loss_class = torch.nn.CrossEntropyLoss()(img_all_out, img_all_label)
                loss_class_mcd = torch.nn.CrossEntropyLoss()(img_all_out_mcd, img_all_label)               
                loss_D = loss_class + loss_class_mcd - loss_L1

                loss_D = reduce_mean(loss_D, dist.get_world_size())  
                if args.apex_train and not torch.isnan(loss_D):
                    optimD.zero_grad()
                    with amp.scale_loss(loss_D, optimD) as scaled_loss:
                        scaled_loss.backward() 
                    optimD.step()
                else:
                    optimD.zero_grad()
                    loss_D.backward()
                    optimD.step() 
            else:
                ## ################### ATTACK #########################
                adv = netG(img)
                adv_rot = netG(img_rot)
                adv_aug = netG(img_aug)

                # Smoothing
                if args.gs:
                    adv = kernel(adv)
                    adv_rot = kernel(adv_rot)
                    adv_aug = kernel(adv_aug)

                # Projection
                adv = torch.min(torch.max(adv, img - eps), img + eps)
                adv = torch.clamp(adv, 0.0, 1.0)
                adv_rot = torch.min(torch.max(adv_rot, img_rot - eps), img_rot + eps)
                adv_rot = torch.clamp(adv_rot, 0.0, 1.0)
                adv_aug = torch.min(torch.max(adv_aug, img_aug - eps), img_aug + eps)
                adv_aug = torch.clamp(adv_aug, 0.0, 1.0)

                ## ################### RUN #########################
                adv_out = model(normalize(adv))
                adv_rot_out = model(normalize(adv_rot))
                adv_aug_out = model(normalize(adv_aug))

                adv_out_mcd = model_m3d(normalize(adv))
                adv_rot_out_mcd = model_m3d(normalize(adv_rot))
                adv_aug_out_mcd = model_m3d(normalize(adv_aug))

                loss_attack, loss_L1 = 0.0, 0.0
                for out in [adv_out, adv_rot_out, adv_aug_out, adv_out_mcd, adv_rot_out_mcd, adv_aug_out_mcd]:

                    loss_attack += torch.nn.CrossEntropyLoss()(out , target_label) 
                                                      
                for out, out_mcd in [[adv_out, adv_out_mcd], [adv_rot_out, adv_rot_out_mcd], [adv_aug_out, adv_aug_out_mcd]]:
                    loss_L1 +=torch.mean(torch.abs(F.softmax(out, dim=1) - F.softmax(out_mcd, dim=1)))
                loss = loss_attack + loss_L1
                loss = reduce_mean(loss, dist.get_world_size())       
                loss_f2 = loss
                
                if args.apex_train and not torch.isnan(loss_f2):
                    optimG.zero_grad()
                    with amp.scale_loss(loss_f2, optimG) as scaled_loss:
                        scaled_loss.backward()
                    optimG.step()
                else:
                    optimG.zero_grad()
                    loss_f2.backward()
                    optimG.step()
            
            loss_map['running_loss'] += loss.item() * adv_out.size(0)
            loss_map['running_D_loss'] += loss_D.item() * adv_out.size(0)
            # loss_map['running_loss_f2'] += loss_f2.item() * adv_out.size(0)
            loss_map['total'] += adv_out.size(0)

        if i % 10 == 9 and dist.get_rank() == 0:
            for data in loss_map:
                loss_map[data] /= loss_map['total']
        
            print('Epoch: {0} \t Batch: {1} \t D_loss: {2:.5f}  \t G_loss: {3:.5f}\t  '.format(
                epoch, i, loss_map['running_D_loss'], loss_map['running_loss']), end=' ')

            logger.info('Epoch: {0} \t Batch: {1} \t D_loss: {2:.5f}  \t G_loss: {3:.5f}\t  '.format(
                epoch, i, loss_map['running_D_loss'], loss_map['running_loss']))

            for data in loss_map:
                loss_map[data] = 0.

    if args.local_rank == 0:
        torch.save(netG.module.state_dict(), args.save_dir +
                   '/netG_{}_{}_{}.pth'.format(args.model_type, epoch, args.match_target))
        


            