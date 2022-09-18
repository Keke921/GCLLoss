import argparse
import os
import random
import time
import warnings
import sys
import numpy as np
from pathlib2 import Path
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models
from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix
from utils import *
from dataset.imbalance_cifar import IMBALANCECIFAR10, IMBALANCECIFAR100
from losses import *
from sampler import *

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Cifar Training')
parser.add_argument('--dataset', default='cifar10', help='dataset setting')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet32', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet32)')
parser.add_argument('--loss_type', default="GCL", type=str, help='loss type')   #LDAM GCL
parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
parser.add_argument('--imb_factor', default=0.01, type=float, help='imbalance factor')
parser.add_argument('--train_rule', default='None', type=str, help='data sampling strategy for train loader') 
parser.add_argument('--mixup', default=True, type=bool, help='if use mix-up') 
parser.add_argument('--rand_number', default=0, type=int, help='fix random number for data sampling')
parser.add_argument('--exp_str', default='a', type=str, #bs128
                    help='number to indicate which experiment it is')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=2e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default=None, type=str, metavar='PATH',  #'ckpt-159.pth.tar'
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=123, type=int,       #None
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

parser.add_argument('--root_model', type=str, default='checkpoint')

best_acc1 = 0
best_epoch = 0

def main():
    args = parser.parse_args()   
    args.store_name = prepare_folders(args)
    if args.seed is not None:
        os.environ['PYTHONHASHSEED'] = str(args.seed)        
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    global best_epoch
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    print("=> creating model '{}'".format(args.arch))
    num_classes = 100 if args.dataset == 'cifar100' else 10
    classifier = True
    
    if args.loss_type == 'Noise':
        use_norm = False
        use_noise = True
    elif args.loss_type == 'CE':
        use_norm = False
        use_noise = False        
    else:
        use_norm = True
        use_noise = False
    model = models.__dict__[args.arch](num_classes=num_classes, classifier = classifier, 
                                       use_norm= use_norm, use_noise = use_noise)
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()
    
    # Data loading
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(), #transforms.RandomApply(transforms_list, p=0.5)
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if args.dataset == 'cifar10':
        train_dataset = IMBALANCECIFAR10(root='./data', imb_type=args.imb_type, imb_factor=args.imb_factor, 
                                         rand_number=args.rand_number, train=True, download=True, transform=transform_train)
        val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_val)
    elif args.dataset == 'cifar100':
        train_dataset = IMBALANCECIFAR100(root='./data', imb_type=args.imb_type, imb_factor=args.imb_factor, 
                                          rand_number=args.rand_number, train=True, download=True, transform=transform_train)
        val_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_val)
    else:
        warnings.warn('Dataset is not listed')
        return
    cls_num_list = train_dataset.get_cls_num_list()
        
    print('cls num list:')
    print(cls_num_list)
    args.cls_num_list = cls_num_list
            
    #optimizer setting
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cuda:0')
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # init log for training
    root_log = 'log'
    log_training = open(os.path.join(args.store_name, root_log,  'log_train.csv'), 'w')
    log_testing = open(os.path.join(args.store_name, root_log, 'log_test.csv'), 'w')
    with open(os.path.join(args.store_name, root_log, 'args.txt'), 'w') as f:
        f.write(str(args))
    tf_writer = SummaryWriter(log_dir=os.path.join(args.store_name,root_log))
    #把code也存一下
    code_dir = os.path.join(args.store_name, root_log, "codes")
    print("=> code will be saved in {}".format(code_dir))   
    this_dir = Path.cwd()
    ignore = shutil.ignore_patterns(
        "*.pyc", "*.so", "*.out", "*pycache*","*spyproject*","*pth","*pth*", "*log*", \
        "*checkpoint*", "*data*", "*result*", "*temp*","saved"
    )
    shutil.copytree(this_dir, code_dir, ignore=ignore)

    if args.train_rule == 'None':
        train_sampler = None  
        per_cls_weights = None 
    elif args.train_rule == 'BalancedRS':
        train_sampler = BalancedDatasetSampler(train_dataset)
        per_cls_weights = None
    elif args.train_rule == 'EffectNumRS':
        train_sampler = EffectNumSampler(train_dataset)
        per_cls_weights = None
    elif args.train_rule == 'CBENRS':
        train_sampler = CBEffectNumSampler(train_dataset)
        per_cls_weights = None  
    elif args.train_rule == 'ClassAware':
        train_sampler = ClassAwareSampler(train_dataset)
        per_cls_weights = None
    elif args.train_rule == 'EffectNumRW':
        train_sampler = None
        sampler = EffectNumSampler(train_dataset)
        per_cls_weights = sampler.per_cls_weights/sampler.per_cls_weights.sum()  
        per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)
    elif args.train_rule == 'BalancedRW':
        train_sampler = None
        sampler = BalancedDatasetSampler(train_dataset)
        per_cls_weights = sampler.per_cls_weights/sampler.per_cls_weights.sum()   
        per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)        
    else:
        warnings.warn('Sample rule is not listed')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=256, shuffle=False, 
        num_workers=args.workers, pin_memory=True)

    if args.loss_type == 'CE':
        criterion = nn.CrossEntropyLoss(weight=per_cls_weights).cuda(args.gpu)
    elif args.loss_type == 'Focal':
        criterion = FocalLoss(weight=per_cls_weights, gamma=1).cuda(args.gpu)
    elif args.loss_type == 'GCL':
        criterion = GCLLoss(cls_num_list=cls_num_list, m=0., s=30, noise_mul =0.5, weight=per_cls_weights).cuda(args.gpu)           
    elif args.loss_type == 'LDAM':
        criterion = LDAMLoss(cls_num_list=cls_num_list, max_m=0.5, s=30, weight=per_cls_weights).cuda(args.gpu) 

    
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)
           
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, log_training, tf_writer)   #cont_img[epoch],
        
        # evaluate on validation set
        acc1, val_loss = validate(val_loader, model, criterion, epoch, args, log_testing, tf_writer)
        
        #scheduler.step(val_loss)
        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        if is_best:
            best_epoch = epoch
        
        tf_writer.add_scalar('acc/test_top1_best', best_acc1, epoch)
        output_best = 'Best Prec@1: %.3f, Best epoch: %d\n' % (best_acc1, best_epoch)
        print(output_best)
        log_testing.write(output_best + '\n')
        log_testing.flush()
        save_checkpoint(args, {
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)


def train(train_loader, model, criterion, optimizer, epoch, args, log, tf_writer):  #
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e') 
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    # switch to train mode
    model.train()    
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time       
        data_time.update(time.time() - end)
        
        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)        
        if args.mixup is True:
            images, targets_a, targets_b, lam = mixup_data(input, target)
            output = model(images)
            loss = mixup_criterion(criterion, output, targets_a, targets_b, lam) 
            if args.loss_type == 'Noise':
                output = output[0]   
            acc1_a, acc5_a = accuracy(output, targets_a, topk=(1, 5))
            acc1_b, acc5_b = accuracy(output, targets_b, topk=(1, 5))
            acc1, acc5 = lam*acc1_a+(1-lam)*acc1_b, lam*acc5_a+(1-lam)*acc5_b
        else:
            output = model(input)   
            loss = criterion(output, target)
            if args.loss_type == 'Noise':
                output = output[0]
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

        
        # measure accuracy and record loss    
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                      #'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      #'Data: {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Total Loss: {loss.val:.4f} ({loss.avg:.4f})\t'                
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), loss=losses, 
                top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr']))  
            print(output)
            log.write(output + '\n')
            log.flush()

    tf_writer.add_scalar('loss/train', losses.avg, epoch)
    tf_writer.add_scalar('acc/train_top1', top1.avg, epoch)
    tf_writer.add_scalar('acc/train_top5', top5.avg, epoch)
    tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)

def validate(val_loader, model, criterion, epoch, args, log=None, tf_writer=None, flag='val'):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
   
    # switch to evaluate mode
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            
            input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output       
            output= model(input)
            loss = criterion(output, target) 
            
            if args.loss_type == 'Noise':
                output = output[0]
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            _, pred = torch.max(output, 1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            if i % args.print_freq == 0:
                output = ('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
                print(output)
        cf = confusion_matrix(all_targets, all_preds).astype(float)
        cls_cnt = cf.sum(axis=1)
        cls_hit = np.diag(cf)
        cls_acc = cls_hit / cls_cnt
        output = ('{flag} Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
                .format(flag=flag, top1=top1, top5=top5, loss=losses))
        out_cls_acc = '%s Class Accuracy: %s'%(flag,(np.array2string(cls_acc, separator=',', formatter={'float_kind':lambda x: "%.3f" % x})))
        print(output)
        print(out_cls_acc)
        if log is not None:
            log.write(output + '\n')
            log.write(out_cls_acc + '\n')
            log.flush()

        tf_writer.add_scalar('loss/test_'+ flag, losses.avg, epoch)
        tf_writer.add_scalar('acc/test_' + flag + '_top1', top1.avg, epoch)
        tf_writer.add_scalar('acc/test_' + flag + '_top5', top5.avg, epoch)
        tf_writer.add_scalars('acc/test_' + flag + '_cls_acc', {str(i):x for i, x in enumerate(cls_acc)}, epoch)

    return top1.avg, loss

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    epoch = epoch + 1
    if epoch <= 5:
        lr = args.lr * epoch / 5
    elif epoch > 180:
        lr = args.lr * 0.01
    elif epoch >160:
        lr = args.lr * 0.1
    else:
        lr = args.lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    main()
