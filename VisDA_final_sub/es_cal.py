#!/usr/bin/env python
import argparse
import builtins
import os
import random
import shutil
import time
import warnings
import collections
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# import torchvision.models as models

import models
from data.visda17 import VisDA17

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "  0, 1, 2, 3"
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=10., type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lrd', '--learning-rate-decay', default=0.2, type=float,
                    metavar='LRD', help='learning rate decay', dest='lrd')
parser.add_argument('--schedule', default=[10, 20], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by a ratio)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0., type=float,
                    metavar='W', help='weight decay (default: 0.)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 100)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:10001', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--pretrained', default='', type=str,
                    help='path to moco pretrained checkpoint')

parser.add_argument('--tb', action='store_true',
                    help='tensorboard')
parser.add_argument('--start-eval', default=3, type=int,
                    help='epoch number when starting evaluation')
parser.add_argument('--trial', default=None, type=str,
                    help='auxiliary string to distinguish trials')
parser.add_argument('--finetune', action='store_true',
                    help='do not freeze CNN')
parser.add_argument('--new-resume', action='store_true',
                    help='reset optimizer and start_epoch')
parser.add_argument('--dataset', default='imagenet', type=str, choices=['visda','imagenet', 'imagenet_val', 'cifar10', 'cifar100', 'speech_commands', 'covtype', 'higgs', 'higgsall', 'higgs100K', 'higgs100Kall', 'higgs1M', 'higgs1Mall'],
                    help='dataset to use')
parser.add_argument('--class-ratio', default=1., type=float,
                    help='reduce training dataset size by removing classes')
parser.add_argument('--data-ratio', default=1., type=float,
                    help='reduce training dataset size by removing data per class')

best_acc1 = 0


def main(args):

    # scale lr
    args.lr = args.lr * args.batch_size / 256
    print('lr is scaled to {}'.format(args.lr))
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    main_worker(args)


def main_worker(args):
    global best_acc1
    best_acc1 = 0
    tb_logger = None

    # suppress printing if not master

    # create model
    print("=> creating model '{}'".format(args.arch))

    num_classes = 128

    model = models.__dict__[args.arch](num_classes=num_classes, pretrained=True)


    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")

            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith('module.encoder_q.') and not k.startswith('module.encoder_q.fc_csg.'):
                    # remove prefix
                    state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                elif k.startswith('encoder_q.') and not k.startswith('encoder_q.fc_csg.'):
                    # remove prefix
                    state_dict[k[len("encoder_q."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]
            msg = model.load_state_dict(state_dict)
            print(msg.missing_keys)
            args.start_epoch = 0
            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))
    else:
        print("=> no pre-trained model")

    model = model.cuda()
    model = torch.nn.DataParallel(model)

    cudnn.benchmark = True

    # Data loading code
    if args.dataset == 'imagenet':
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')
    elif args.dataset == 'imagenet_val':
        traindir = valdir = os.path.join(args.data, 'val')

    else:
        traindir = valdir = args.data
    if 'imagenet' in args.dataset:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        normalize = None

    if args.dataset in ['covtype', 'higgs', 'higgsall', 'higgs100K', 'higgs100Kall', 'higgs1M', 'higgs1Mall']:
        train_transform = None
    elif 'imagenet' in args.dataset:
        train_transform = \
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
    elif args.dataset == 'visda':
        train_transform_list = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])
    if args.dataset in ['imagenet', 'imagenet_val']:
        print('ImageNet ImageFolder at: {}'.format(traindir))
        # train_dataset = datasets.ImageFolder(traindir, train_transform)
        train_dataset_main = datasets.ImageFolder(traindir, train_transform)
        train_dataset_main.targets = torch.tensor(train_dataset_main.targets)
        idx = torch.zeros_like(train_dataset_main.targets)

        for i in range(451, 491):
            idx += train_dataset_main.targets == i

        indexes = np.where(idx==1)[0]
        train_dataset = torch.utils.data.dataset.Subset(train_dataset_main, indexes)
    elif args.dataset in ['visda', 'visda_val']:
        print(f'Visda at:{traindir}')
        train_dataset = VisDA17(txt_file=os.path.join(args.data, "train/image_list.txt"), root_dir=os.path.join(args.data, "train"),
                                                transform=train_transform_list)
    else:
        raise NotImplementedError('unsupported dataset: {}'.format(args.dataset))

    train_sampler = None

    if args.dataset not in ['covtype', 'higgs', 'higgsall', 'higgs100K', 'higgs100Kall', 'higgs1M', 'higgs1Mall']:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, sampler=train_sampler, drop_last=True)

    if 'imagenet' in args.dataset:
        val_transform = \
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
    elif args.dataset == 'visda':
        val_transforms_list = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                0.229, 0.224, 0.225])
        ])

    else:
        raise NotImplementedError('unsupported dataset: {}'.format(args.dataset))

    if args.dataset not in ['covtype', 'higgs', 'higgsall', 'higgs100K', 'higgs100Kall', 'higgs1M', 'higgs1Mall']:
        val_dataset = VisDA17(txt_file=os.path.join(args.data, "validation/image_list.txt"), root_dir=os.path.join(
            args.data, "validation"), transform=val_transforms_list)

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=200, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    acc1 = validate(val_loader, model, args.start_epoch, None, args)



def validate(val_loader, model, epoch, tb_logger, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader),[batch_time, losses, top1, top5],prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    features_list = []

    with torch.no_grad():
        end = time.time()
        es_0 = 0
        es_1 = 0
        es_2 = 0
        
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
    
            bsz = target.size(0)
            features, _ = model(images)
            features = nn.functional.normalize(features, dim=1)
            inner_prod = 2. - 2.*torch.mm(features, features.t())
            inner_prod = inner_prod + 1e-5 + torch.eye(inner_prod.shape[0]).cuda()

            cnt = inner_prod.shape[0]*(inner_prod.shape[0] - 1)/2.

            ct_0 = -torch.log(inner_prod)
            ct_1 = torch.pow(inner_prod, -0.5)
            ct_2 = torch.pow(inner_prod, -1)

            cross_0 = torch.triu(ct_0, diagonal=1).sum()/cnt
            cross_1 = torch.triu(ct_1, diagonal=1).sum()/cnt
            cross_2 = torch.triu(ct_2, diagonal=1).sum()/cnt

            es_0 += cross_0
            es_1 += cross_1
            es_2 += cross_2
            if (i % args.print_freq == 0) or (i == len(val_loader) - 1):
                progress.display(i)
        print('ES_0', es_0/i)
        print('ES_1', es_1/i)
        print('ES_2', es_2/i)


def save_checkpoint(state, is_best, is_milestone, filename='checkpoint.pth', dataset='dataset', epoch=-1):
    torch.save(state, filename)
    # if is_best:
        # shutil.copyfile(filename, filename.replace('checkpoint_lincls', 'model_best'))
    if is_milestone:
        shutil.copyfile(filename, os.path.splitext(filename)[0] + '_{:d}.pth'.format(epoch))


def sanity_check(state_dict, pretrained_weights):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    print("=> loading '{}' for sanity check".format(pretrained_weights))
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    state_dict_pre = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        # only ignore fc layer
        if 'fc.weight' in k or 'fc.bias' in k:
            continue

        # name in pretrained model
        k_pre = 'module.encoder_q.' + k[len('module.'):] \
            if k.startswith('module.') else 'module.encoder_q.' + k

        if k_pre not in state_dict_pre:
            k_pre = k_pre[len('module.'):]

        assert ((state_dict[k].cpu() == state_dict_pre[k_pre]).all()), \
            '{} is changed in linear classifier training.'.format(k)

    print("=> sanity check passed.")


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    for milestone in args.schedule:
        lr *= args.lrd if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
