import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import sys
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F
from models import resnet_cifar
from autoaug import CIFAR10Policy, Cutout
import moco.loader
import moco.builder
from dataset.imbalance_cifar import ImbalanceCIFAR100, ImbalanceCIFAR10
import torchvision.datasets as datasets
from losses import GMLLoss
from utils import shot_acc

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


model_names += ['resnext101_32x4d']
model_names += ['resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110']

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar100', choices=['inat', 'imagenet', 'cifar100', 'cifar10'])
parser.add_argument('--data', metavar='DIR', default='./data')
parser.add_argument('--root_path', type=str, default='./outputs')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet32',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet32)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--imb-factor', type=float,
                    metavar='IF', help='imbalanced factor', dest='imb_factor')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[160, 180], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')

parser.add_argument('--arch_t', default='resnet32',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet32)')
parser.add_argument('--path_t', default=None, type=str, help='path to teacher model')
parser.add_argument('--feat_t', default=None, type=int,
                    help='last feature dim of teacher')

# moco specific configs:
parser.add_argument('--moco-dim', default=32, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=4096, type=int,
                    help='queue size; number of negative keys (default: 1024)')
parser.add_argument('--base_k', default=2, type=int,
                    help='minimum number of contrast for each class')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.1, type=float,
                    help='softmax temperature (default: 0.1)')

# options for moco v2
parser.add_argument('--mlp', default=True, type=str2bool,
                    help='use mlp head')
parser.add_argument('--aug-plus', default=True, type=str2bool,
                    help='use moco v2 data augmentation')
parser.add_argument('--normalize', default=True, type=str2bool,
                    help='use cosine classifier')


# options for paco
parser.add_argument('--mark', default=None, type=str,
                    help='log dir')
parser.add_argument('--reload', default=None, type=str,
                    help='load supervised model')
parser.add_argument('--warmup_epochs', default=5, type=int,
                    help='warmup epochs')
parser.add_argument('--alpha', default=1.0, type=float,
                    help='weight for kl_div loss (default: 1.0)')
parser.add_argument('--beta', default=1.0, type=float,
                    help='weight for contrast loss (default: 1.0)')
parser.add_argument('--gamma', default=1.0, type=float,
                    help='weight for cross-entropy loss (default: 1.0)')
parser.add_argument('--aug', default='cifar100', type=str,
                    help='aug strategy')
parser.add_argument('--num_classes', default=100, type=int,
                    help='num classes in dataset')
parser.add_argument('--feat_dim', default=64, type=int,
                    help='last feature dim of backbone')

parser.add_argument('--epoch-multiplier', default=1, type=int,
                    help='multiply epoch by times')
                    

def main():
    args = parser.parse_args()
    args.epochs *= args.epoch_multiplier
    args.warmup_epochs *= args.epoch_multiplier
    
    args.root_model = f'{args.root_path}/{args.dataset}/{args.mark}'
    os.makedirs(args.root_model, exist_ok=True)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    main_worker(args.gpu, 1, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    def print_pass(*args):
        tqdm.write(' '.join([str(v) for v in args]), file=sys.stdout)
    builtins.print = print_pass
    args.is_master = True
    tb_logger = SummaryWriter(os.path.join(args.root_model, 'tb_logs'), flush_secs=2)

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
        torch.cuda.set_device(args.gpu)

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = moco.builder.MoCo(
        getattr(resnet_cifar, args.arch),
        args.moco_dim, args.moco_k, args.moco_m, args.mlp,
        args.feat_dim, args.feat_t, args.normalize, args.num_classes)
    print(model)
    
    if args.path_t is not None:
        try:
            encoder_k = getattr(resnet_cifar, args.arch_t)(num_classes=args.num_classes, use_norm=False, return_features=True)
            encoder_k.load_state_dict(torch.load(args.path_t)['state_dict'])
        except RuntimeError:
            encoder_k = getattr(resnet_cifar, args.arch_t)(num_classes=args.num_classes, use_norm=True, return_features=True)
            encoder_k.load_state_dict(torch.load(args.path_t)['state_dict'])
        model.set_distiller(encoder_k)
    

    model = model.cuda(args.gpu)
    # define loss function (criterion) and optimizer
    criterion_ce = nn.CrossEntropyLoss().cuda(args.gpu)
    criterion = GMLLoss(args.beta, args.gamma, args.moco_t, args.num_classes, args.alpha).cuda(args.gpu) 
    
    def get_wd_params(model: nn.Module):
        all_params = tuple(model.parameters())
        no_wd_params = list()
        for n, p in model.named_parameters():
            if '__no_wd__' in n:
                no_wd_params.append(p)
        # Only weights of specific layers should undergo weight decay.
        wd_params = []
        for p in all_params:
            matched = False
            for nwp in no_wd_params:
                if p is nwp:
                    matched = True
                    break
            if not matched:
                wd_params.append(p)
        
        assert len(wd_params) + len(no_wd_params) == len(all_params), "Sanity check failed."
        return wd_params, no_wd_params
    
    wd_params, no_wd_params = get_wd_params(nn.ModuleList([model, criterion]))
    optimizer = torch.optim.SGD(wd_params, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    if len(no_wd_params) > 0:
        optimizer_no_wd = torch.optim.SGD(no_wd_params, args.lr,
                                    momentum=args.momentum,
                                    weight_decay=0.0)
        [optimizer.add_param_group(pg) for pg in optimizer_no_wd.param_groups]

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    augmentation_regular = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        CIFAR10Policy(),    # add AutoAug
        transforms.ToTensor(),
        Cutout(n_holes=1, length=16),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    CIFAR = datasets.CIFAR100 \
        if args.dataset == 'cifar100' else datasets.CIFAR10
    val_dataset = CIFAR(
        root='./cifar', 
        train=False, 
        download=True, 
        transform=val_transform)
        
    transform_train = [
        transforms.Compose(augmentation_regular),
        transforms.Compose(augmentation_regular),
        transforms.Compose(augmentation_regular),
    ]
     
    ImbalanceCIFAR = ImbalanceCIFAR100 \
        if args.dataset == 'cifar100' else ImbalanceCIFAR10
    train_dataset = ImbalanceCIFAR(root='./cifar', imb_type='exp', imb_factor=args.imb_factor, rand_number=0, 
    train=True, download=True, transform=transform_train)
    print(transform_train)

    print(f'===> Training data length {len(train_dataset)}')

    criterion.cal_weight_for_classes(train_dataset.get_cls_num_list())
    if args.base_k > 0:
        model.set_cls_weight(criterion.cls_weight, args.base_k)
        
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True,
        persistent_workers=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, persistent_workers=True)

    if args.evaluate:
        print(" start evaluation **** ")
        validate(val_loader, train_loader, model, criterion_ce, tb_logger, 0, args)
        return

    best_acc1 = 0

    for epoch in tqdm(range(args.start_epoch, args.epochs), position=0, leave=True, disable=not args.is_master):
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, tb_logger, epoch, args)
        acc1 = validate(val_loader, train_loader, model, criterion_ce, tb_logger, epoch, args)
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        output_cur = 'Current Prec@1: %.3f\n' % (acc1)
        output_best = 'Best Prec@1: %.3f\n' % (best_acc1)
        print(output_cur, output_best)
        save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, is_best=False, filename=f'{args.root_model}/moco_ckpt.pth.tar')


        if (epoch + 1) % args.print_freq == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, is_best=False, filename=f'{args.root_model}/moco_ckpt_{(epoch+1):04d}.pth.tar')


def train(train_loader, model, criterion, optimizer, tb_logger, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    total_logits = torch.empty((0, args.num_classes)).cuda()
    total_labels = torch.empty(0, dtype=torch.long).cuda()

    end = time.time()
    for i, (images, target, in_idx) in enumerate(tqdm(train_loader, position=1, leave=False, disable=not args.is_master)):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = [img.cuda(args.gpu, non_blocking=True) for img in images]
            target = target.cuda(args.gpu, non_blocking=True)
            in_idx = in_idx.cuda(args.gpu, non_blocking=True)

        # compute output
        query, key, k_labels, k_idx, logits, t_logits = model(*images, target, in_idx)
        loss = criterion(query, target, in_idx, key, k_labels, k_idx, logits, t_logits)

        total_logits = torch.cat((total_logits, logits))
        total_labels = torch.cat((total_labels, target))

        acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        losses.update(loss.item(), logits.size(0))
        top1.update(acc1[0], logits.size(0))
        top5.update(acc5[0], logits.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i, args)
            global_step = epoch * len(train_loader) + i
            if tb_logger is not None:
                tb_logger.add_scalar('Train/losses', losses.avg, global_step)
                tb_logger.add_scalar('Train/top1', top1.avg, global_step)
                tb_logger.add_scalar('Train/top5', top5.avg, global_step)
                
def validate(val_loader, train_loader, model, criterion, tb_logger, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    total_logits = torch.empty((0, args.num_classes)).cuda()
    total_labels = torch.empty(0, dtype=torch.long).cuda()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(tqdm(val_loader, position=1, leave=False, disable=not args.is_master)):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            total_logits = torch.cat((total_logits, output))
            total_labels = torch.cat((total_labels, target))

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i, args)

        # TODO: this should also be done with the ProgressMeter
        open(args.root_model+"/"+args.mark+".log","a+").write(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}\n'
              .format(top1=top1, top5=top5))
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}\n'.format(top1=top1, top5=top5))
        
        probs, preds = F.softmax(total_logits.detach(), dim=1).max(dim=1)
        many_acc_top1, median_acc_top1, low_acc_top1, cls_accs = shot_acc(preds, total_labels, train_loader, acc_per_cls=True)

        if tb_logger is not None:
            tb_logger.add_scalar('Validation/Acc@1', top1.avg, epoch)
            tb_logger.add_scalar('Validation/Acc@5', top5.avg, epoch)
            tb_logger.add_scalar('Validation/Many_acc', many_acc_top1, epoch)
            tb_logger.add_scalar('Validation/Medium_acc', median_acc_top1, epoch)
            tb_logger.add_scalar('Validation/Low_acc', low_acc_top1, epoch)

    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))


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

    def display(self, batch, args):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        open(args.root_model+"/"+args.mark+".log","a+").write('\t'.join(entries)+"\n")
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch <= args.warmup_epochs:
        lr = args.lr / args.warmup_epochs * (epoch + 1)
    elif epoch > args.schedule[1] * args.epoch_multiplier:
        lr = args.lr * 0.01
    elif epoch > args.schedule[0] * args.epoch_multiplier:
        lr = args.lr * 0.1
    else:
        lr = args.lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred)).contiguous()

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
