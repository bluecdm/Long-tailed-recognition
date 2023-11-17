import argparse
import builtins
import math
import os
os.environ['OPENBLAS_NUM_THREADS'] = '2'
import random
import shutil
import sys
import time
import warnings
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from models import resnet_imagenet
from randaugment import rand_augment_transform, GaussianBlur
import moco.loader
import moco.builder
from dataset.imagenet import ImageNetLT
from dataset.imagenet_moco import ImageNetLT_moco
from dataset.inat import INaturalist
from dataset.inat_moco import INaturalist_moco
from losses import GMLLoss
from utils import shot_acc, save_codes


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
model_names += ['resnext101_32x4d']

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
parser.add_argument('--dataset', default='imagenet', choices=['inat', 'imagenet'])
parser.add_argument('--data', metavar='DIR', default='./data')
parser.add_argument('--root_path', type=str, default='./outputs')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning_rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
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
parser.add_argument('--multiprocessing_distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--arch_t', default='resnet32',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet32)')
parser.add_argument('--path_t', default=None, type=str, help='path to teacher model')
parser.add_argument('--feat_t', default=None, type=int,
                    help='last feature dim of teacher')

# moco specific configs:

parser.add_argument('--moco_dim', default=1024, type=int,
                    help='feature dimension (default: 1024)')
parser.add_argument('--moco_k', default=8192, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--base_k', default=0, type=int,
                    help='minimum number of contrast for each class')
parser.add_argument('--moco_m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco_t', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')

# options for moco v2
parser.add_argument('--mlp', default=True, type=str2bool,
                    help='use mlp head')
parser.add_argument('--cos', default=True, type=str2bool,
                    help='use cosine lr schedule')
parser.add_argument('--normalize', default=True, type=str2bool,
                    help='use cosine classifier')

# options for paco
parser.add_argument('--mark', default=None, type=str,
                    help='log dir')
parser.add_argument('--warmup_epochs', default=0, type=int,
                    help='warmup epochs')
parser.add_argument('--aug', default=None, type=str,
                    help='aug strategy')
parser.add_argument('--randaug_m', default=10, type=int, help='randaug-m')
parser.add_argument('--randaug_n', default=2, type=int, help='randaug-n')
parser.add_argument('--num_classes', default=1000, type=int, help='num classes in dataset')
parser.add_argument('--feat_dim', default=2048, type=int,
                    help='last feature dim of backbone')

parser.add_argument('--beta', default=1.0, type=float,
                    help='weight for contrast loss (default: 1.0)')
parser.add_argument('--gamma', default=1.0, type=float,
                    help='weight for cross-entropy loss (default: 1.0)')


def main():
    args = parser.parse_args()
    
    args.root_model = f'{args.root_path}/{args.dataset}/{args.mark}'
    if args.resume == '':
        save_codes(args.root_model)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = int(os.environ["WORLD_SIZE"])
        ngpus_per_node = torch.cuda.device_count()
    
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(0, 1, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
        
    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
        args.do_print = False
        tb_logger = None
    else:
        def print_pass(*args):
            tqdm.write(' '.join([str(v) for v in args]), file=sys.stdout)
        builtins.print = print_pass
        args.do_print = True
        tb_logger = SummaryWriter(os.path.join(args.root_model, 'tb_logs'), flush_secs=2)

    if args.multiprocessing_distributed:
        args.rank = int(os.environ["RANK"]) * ngpus_per_node + gpu
        dist.init_process_group(backend='nccl', world_size=args.world_size, rank=args.rank)
    # create model
    print("=> creating model '{}'".format(args.arch))
    model = moco.builder.MoCo(
        getattr(resnet_imagenet, args.arch),
        args.moco_dim, args.moco_k, args.moco_m, args.mlp,
        args.feat_dim, args.feat_t, args.normalize, args.num_classes)
    print(model)

    if args.path_t is not None:
        t_state_dict = torch.load(args.path_t, map_location='cpu')['state_dict']
        t_state_dict = {(k[7:] if k.startswith('module.') else k)
                        : v for k, v in t_state_dict.items()}
        t_state_dict = {k[10:]: v for k, v in t_state_dict.items()
                        if k.startswith('encoder_q.')}
        try:
            encoder_k = getattr(resnet_imagenet, args.arch_t)(num_classes=args.num_classes, use_norm=False, return_features=True)
            encoder_k.load_state_dict(t_state_dict)
        except RuntimeError:
            encoder_k = getattr(resnet_imagenet, args.arch_t)(num_classes=args.num_classes, use_norm=True, return_features=True)
            encoder_k.load_state_dict(t_state_dict)
        model.set_distiller(encoder_k)
        
    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)
    criterion_ce = nn.CrossEntropyLoss().cuda(args.gpu)
    criterion = GMLLoss(args.beta, args.gamma, args.moco_t, args.num_classes).cuda(args.gpu)
    if args.multiprocessing_distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        # When using a single GPU per process and per
        # DistributedDataParallel, we need to divide the batch size
        # ourselves based on the total number of GPUs we have
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)

        criterion = torch.nn.SyncBatchNorm.convert_sync_batchnorm(criterion)
        criterion = torch.nn.parallel.DistributedDataParallel(criterion, device_ids=[args.gpu], find_unused_parameters=True)

    def get_wd_params(model: nn.Module):
        no_wd_params = list()
        wd_params = list()
        for n, p in model.named_parameters():
            if '__no_wd__' in n:
                no_wd_params.append(p)
            else:
                wd_params.append(p)
        
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
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            criterion.load_state_dict(checkpoint['criterion'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        args.start_epoch = 0

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    txt_train = f'./imagenet_inat/data/iNaturalist18/iNaturalist18_train.txt' if args.dataset == 'inat' \
        else f'./imagenet_inat/data/ImageNet_LT/ImageNet_LT_train.txt'

    txt_test = f'./imagenet_inat/data/iNaturalist18/iNaturalist18_val.txt' if args.dataset == 'inat' \
        else f'./imagenet_inat/data/ImageNet_LT/ImageNet_LT_test.txt'

    normalize = transforms.Normalize(mean=[0.466, 0.471, 0.380], std=[0.195, 0.194, 0.192]) if args.dataset == 'inat' \
        else transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    augmentation_sim = [
            transforms.RandomResizedCrop(224),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)  # not strengthened
            ], p=1.0),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
    ]

    rgb_mean = (0.485, 0.456, 0.406)
    ra_params = dict(translate_const=int(224 * 0.45), img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]),)
    augmentation_randnclsstack = [
            transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)
            ], p=1.0),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
            rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(args.randaug_n, args.randaug_m), ra_params),
            transforms.ToTensor(),
            normalize,
    ]

    augmentation_randncls = [
            transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)
            ], p=1.0),
            rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(args.randaug_n, args.randaug_m), ra_params),
            transforms.ToTensor(),
            normalize,
    ]

    val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize])

    val_dataset = INaturalist(
        root=args.data,
        txt=txt_test,
        transform=val_transform
    ) if args.dataset == 'inat' else ImageNetLT(
        root=args.data,
        txt=txt_test,
        transform=val_transform)

    if args.aug == 'randcls_randclsstack':
         transform_train = [
             transforms.Compose(augmentation_randncls),
             transforms.Compose(augmentation_randnclsstack),
             transforms.Compose(augmentation_randnclsstack),
         ]
    elif args.aug == 'randclsstack_randclsstack':
         transform_train = [
             transforms.Compose(augmentation_randnclsstack),
             transforms.Compose(augmentation_randnclsstack),
             transforms.Compose(augmentation_randnclsstack),
         ]
    elif args.aug == 'randcls_sim':
         transform_train = [
             transforms.Compose(augmentation_randncls),
             transforms.Compose(augmentation_sim),
             transforms.Compose(augmentation_sim),
         ]
    else:
        raise NotImplementedError

    train_dataset = INaturalist_moco(
        root=args.data,
        txt=txt_train,
        transform=transform_train
    ) if args.dataset == 'inat' else ImageNetLT_moco(
        root=args.data,
        txt=txt_train,
        transform=transform_train)
    print(f'===> Training data length {len(train_dataset)}')

    """ https://github.com/pytorch/pytorch/issues/25162 """
    class DistributedSamplerNoDuplicate(torch.utils.data.DistributedSampler):
        """ A distributed sampler that doesn't add duplicates. Arguments are the same as DistributedSampler """
        def __init__(self, *args, **kwargs):
            rank = kwargs.pop('rank')
            super().__init__(*args, **kwargs)
            if not self.drop_last and len(self.dataset) % self.num_replicas != 0:
                # some ranks may have less samples, that's fine
                if rank >= len(self.dataset) % self.num_replicas:
                    self.num_samples -= 1
                self.total_size = len(self.dataset)
    
    if args.multiprocessing_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
#         val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
        val_sampler = DistributedSamplerNoDuplicate(val_dataset, shuffle=False, rank=args.rank)
    else:
        train_sampler = None
        val_sampler = None
        
    try:
        model_without_ddp = model.module
    except AttributeError:
        model_without_ddp = model
    try:
        criterion_without_ddp = criterion.module
    except AttributeError:
        criterion_without_ddp = criterion
    
    criterion_without_ddp.cal_weight_for_classes(train_dataset.cls_num_list)
    if args.base_k > 0:
        model_without_ddp.set_cls_weight(criterion_without_ddp.cls_weight, args.base_k)
        
    
    if args.do_print:
        with open(os.path.join(args.root_model, 'cls_weight.txt'), 'w') as f:
            cls_weight = [str(v.item()) for v in criterion_without_ddp.cls_weight.view(-1)]
            f.write(' '.join(cls_weight) + '\n')
        
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True,
        persistent_workers=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler,
        persistent_workers=True)

    if args.evaluate:
        print(" **** start evaluation **** ")
        validate(val_loader, train_loader, model, criterion_ce, 0, None, args)
        return

    best_acc = 0
    for epoch in tqdm(range(args.start_epoch, args.epochs), position=0, leave=True, disable=not args.do_print):
        if args.multiprocessing_distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, tb_logger, args)
        acc = validate(val_loader, train_loader, model, criterion_ce, epoch, tb_logger, args)
        if acc > best_acc:
            best_acc = acc
            is_best = True
        else:
            is_best = False

        output_cur = 'Current Prec@1: %.3f\n' % (acc)
        output_best = 'Best Prec@1: %.3f\n' % (best_acc)
        print(output_cur, output_best)
        
        if args.do_print:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'acc': acc,
                'state_dict': model.state_dict(),
                'criterion': criterion.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, is_best=is_best, filename=f'{args.root_model}/moco_ckpt.pth.tar')


def train(train_loader, model, criterion, optimizer, epoch, tb_logger, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target, in_idx) in enumerate(tqdm(train_loader, position=1, leave=False, disable=not args.do_print)):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = [img.cuda(args.gpu, non_blocking=True) for img in images]
            target = target.cuda(args.gpu, non_blocking=True)
            in_idx = in_idx.cuda(args.gpu, non_blocking=True)

        # compute output
        query, key, k_labels, k_idx, logits, t_logits \
            = model(*images, target, in_idx)
        loss = criterion(query, target, in_idx, key, k_labels, k_idx, logits, t_logits)
        
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

        if (i + 1) % args.print_freq == 0:
            progress.display(i, args)
            if tb_logger is not None:
                global_step = epoch * len(train_loader) + i
                tb_logger.add_scalar('Train/Loss', losses.avg, global_step)
                tb_logger.add_scalar('Train/Acc@1', top1.avg, global_step)
                tb_logger.add_scalar('Train/Acc@5', top5.avg, global_step)
                


def validate(val_loader, train_loader, model, criterion, epoch, tb_logger, args):
    """ https://stackoverflow.com/questions/71433507/pytorch-python-distributed-multiprocessing-gather-concatenate-tensor-arrays-of """
    def all_gather(q, ws, device):
        """
        Gathers tensor arrays of different lengths across multiple gpus

        Parameters
        ----------
            q : tensor array
            ws : world size
            device : current gpu device

        Returns
        -------
            all_q : list of gathered tensor arrays from all the gpus

        """
        local_size = torch.tensor(q.shape[0], device=device)
        all_sizes = [torch.zeros_like(local_size) for _ in range(ws)]
        dist.all_gather(all_sizes, local_size)
        max_size = max(all_sizes)

        size_diff = max_size.item() - local_size.item()
        if size_diff:
            padding_shape = list(q.shape)
            padding_shape[0] = size_diff
            padding = torch.zeros(*padding_shape, device=device, dtype=q.dtype)
            q = torch.cat((q, padding))

        all_qs_padded = [torch.zeros_like(q) for _ in range(ws)]
        dist.all_gather(all_qs_padded, q)
        all_qs = []
        for q, size in zip(all_qs_padded, all_sizes):
            all_qs.append(q[:size])
        return all_qs
    
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    total_logits = torch.empty((0, args.num_classes)).cuda()
    total_labels = torch.empty(0, dtype=torch.long).cuda()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(tqdm(val_loader, position=1, leave=False, disable=not args.do_print)):
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

            if (i + 1) % args.print_freq == 0:
                progress.display(i, args)

        # TODO: this should also be done with the ProgressMeter
        acc_str = ' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}\n' \
              .format(top1=top1, top5=top5)
        open(args.root_model+"/"+args.mark+".log","a+").write(acc_str)
        print(acc_str)
        
        if torch.distributed.is_initialized():
            total_logits = torch.cat(all_gather(total_logits, args.world_size, total_logits.device))
            total_labels = torch.cat(all_gather(total_labels, args.world_size, total_logits.device))

        probs, preds = F.softmax(total_logits.detach(), dim=1).max(dim=1)
        many_acc_top1, median_acc_top1, low_acc_top1, cls_accs = shot_acc(preds, total_labels, train_loader, acc_per_cls=True)
        acc_str = 'Many_acc: %.5f, Medium_acc: %.5f Low_acc: %.5f\n' \
            % (many_acc_top1, median_acc_top1, low_acc_top1)
        open(args.root_model+"/"+args.mark+".log","a+").write(acc_str)
        print(acc_str)

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
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if epoch < args.warmup_epochs:
       lr = lr / args.warmup_epochs * (epoch + 1 )
    elif args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epochs + 1 ) / (args.epochs - args.warmup_epochs + 1 )))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
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
