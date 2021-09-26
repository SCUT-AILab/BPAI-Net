import os
import time
import shutil
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import numpy as np
import random

import torch.optim
from torch.nn.utils import clip_grad_norm_

from ops.models import TSN
from ops.transforms import *
from opts import parser
from ops.utils import AverageMeter, accuracy
from ops.temporal_shift import make_temporal_pool
from tensorboardX import SummaryWriter
from tools import metric

SEED = 777
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

best_recall = 0


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    global args, best_recall
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    if args.dataset == 'drive':
        from ops.drive_dataset_with_keypoint import Drive as dataset
    elif args.dataset == 'pcl':  # for pcldriver
        from ops.pcldriver_dataset_with_keypoint import BusDeriverDataset3D as dataset
        from ops.pcldriver_dataset_with_keypoint import is_high_quality
        filters = [
            # only train with quality==0, the frames with other quality will disturb the training
            is_high_quality,
        ]
    else:
        raise Exception('dataset not support')
    print('Dataset: {}'.format(args.dataset))
    check_rootfolders()

    model = TSN(args.num_class, args.num_segments, args.modality,
                base_model=args.arch,
                consensus_type=args.consensus_type,
                dropout=args.dropout,
                img_feature_dim=args.img_feature_dim,
                partial_bn=not args.no_partialbn,
                pretrain=args.pretrain,
                is_shift=args.shift, shift_div=args.shift_div, shift_place=args.shift_place,
                fc_lr5=not (args.tune_from and args.dataset in args.tune_from),
                temporal_pool=args.temporal_pool,
                non_local=args.non_local,
                first=args.first, second=args.second, gcn_stride=args.gcn_stride,
                base_lr=args.lr, concat_layer=args.concat_layer, xyc=args.xyc, bn=args.bn,
                arch_cnn=args.arch_cnn, patch_size=args.patch_size, gcn_dropout=args.gcn_dropout)
    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    policies = model.get_optim_policies()
    train_augmentation = model.get_augmentation(
        flip=False if 'something' in args.dataset or 'jester' in args.dataset else True)

    model = torch.nn.DataParallel(model).cuda()

    optimizer = torch.optim.SGD(policies,
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.resume:
        if args.temporal_pool:  # early temporal pool so that we can load the state_dict
            make_temporal_pool(model.module.base_model, args.num_segments)
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_recall = checkpoint['best_recall']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(("=> loaded checkpoint '{}' (epoch {})"
                   .format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    if args.gcn_pretrained or args.tune_from:
        model_dict = model.state_dict()
        replace_dict_gcn = []
        replace_dict_cnn = []
        new_sd = {}
        if args.gcn_pretrained:
            print(("=> fine-tuning from '{}'".format(args.gcn_pretrained)))
            gcn_ckpt = torch.load(args.gcn_pretrained)
            for k, v in gcn_ckpt.items():
                if k == 'A' or 'data_bn' in k or 'edge_importance' in k:
                    continue
                if k not in model_dict and 'module.base_model.st_gcn.' + k in model_dict:
                    replace_dict_gcn.append((k, 'module.base_model.st_gcn.' + k))
        if args.tune_from:
            print(("=> fine-tuning from '{}'".format(args.tune_from)))
            sd = torch.load(args.tune_from)
            sd = sd['state_dict']
            for k, v in sd.items():
                if k not in model_dict and k.replace('base_model', 'module.base_model.cnn') in model_dict:
                    replace_dict_cnn.append((k, k.replace('base_model', 'module.base_model.cnn')))
                elif k not in model_dict and k.replace('base_model', 'base_model.cnn') in model_dict:
                    replace_dict_cnn.append((k, k.replace('base_model', 'base_model.cnn')))
            for k, v in sd.items():
                if k not in model_dict and 'module.' + k in model_dict:
                    replace_dict_cnn.append((k, 'module.' + k))

        for k, k_new in replace_dict_gcn:
            new_sd[k_new] = gcn_ckpt.pop(k)
        for k, k_new in replace_dict_cnn:
            new_sd[k_new] = sd.pop(k)
        if new_sd == {} and args.arch != 'fusion':
            new_sd = sd
        keys1 = set(list(new_sd.keys()))
        keys2 = set(list(model_dict.keys()))
        set_diff = (keys1 - keys2) | (keys2 - keys1)
        print('#### Notice: keys that failed to load: {}'.format(set_diff))
        if args.dataset not in args.tune_from:  # new dataset
            print('=> New dataset, do not load fc weights')
            new_sd = {k: v for k, v in new_sd.items() if 'fc' not in k}
            new_sd = {k: v for k, v in new_sd.items() if 'classifier' not in k}
        if args.modality == 'Flow' and 'Flow' not in args.tune_from:
            new_sd = {k: v for k, v in new_sd.items() if 'conv1.weight' not in k}
        model_dict.update(new_sd)
        model.load_state_dict(model_dict)

    if args.temporal_pool and not args.resume:
        make_temporal_pool(model.module.base_model, args.num_segments)

    # cudnn.benchmark = True

    # Data loading code
    if args.modality != 'RGBDiff':
        normalize = GroupNormalize(input_mean, input_std)
    else:
        normalize = IdentityTransform()

    if args.modality == 'RGB':
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff']:
        data_length = 5

    if args.gcn_stride == 2:
        gcn_segments = 64
    else:
        gcn_segments = args.num_segments
    train_transform = torchvision.transforms.Compose([
        train_augmentation,
        Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
        ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
        normalize,
    ])
    val_transform = torchvision.transforms.Compose([
        GroupScale(int(scale_size)),
        GroupCenterCrop(crop_size),
        Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
        ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
        normalize,
    ])

    if args.dataset == 'drive':
        trainset = dataset(args.root, args.train_split, view=args.view, mode='train',
                           num_segments=args.num_segments, gcn_segments=gcn_segments,
                           patch_size=args.patch_size,
                           transforms=train_transform)
        valset = dataset(args.root, args.val_split, view=args.view, mode='eval',
                         num_segments=args.num_segments, gcn_segments=gcn_segments,
                         patch_size=args.patch_size,
                         transforms=val_transform)
    elif args.dataset == 'pcl':
        trainset = dataset(
            root=args.root,
            anno_path=args.pcl_anno,
            train=True,
            filters=filters,
            transforms=train_transform,
            n_frames=args.num_segments, gcn_segments=gcn_segments,
            interval=0,
        )
        valset = dataset(
            root=args.root,
            anno_path=args.pcl_anno,
            train=False,
            filters=filters,
            transforms=val_transform,
            n_frames=args.num_segments, gcn_segments=gcn_segments,
            interval=0
        )
    else:
        raise Exception

    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
        drop_last=True)  # prevent something not % n_GPU

    val_loader = torch.utils.data.DataLoader(
        valset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss().cuda()
    ske_criterion = torch.nn.CrossEntropyLoss().cuda()

    for group in policies:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

    log_training = open(os.path.join(args.root_log, 'log.csv'), 'w')
    with open(os.path.join(args.root_log, 'args.txt'), 'w') as f:
        f.write(str(args))
    tf_writer = SummaryWriter(log_dir=args.root_log)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr_type, args.lr_steps)

        # train for one epoch
        train(train_loader, model, criterion, ske_criterion, optimizer, epoch, log_training, tf_writer)

        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1 or epoch == 0:
            recall = validate(val_loader, model, criterion, ske_criterion, epoch, log_training, tf_writer)

            # remember best prec@1 and save checkpoint
            is_best = recall > best_recall
            best_recall = max(recall, best_recall)
            tf_writer.add_scalar('acc/val_recall_best', best_recall, epoch)

            output_best = 'Now recall: %.3f, Best recall: %.3f\n' % (recall, best_recall)
            print(output_best)
            log_training.write(output_best + '\n')
            log_training.flush()

            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_recall': best_recall,
            }, is_best)


def train(train_loader, model, criterion, ske_criterion, optimizer, epoch, log, tf_writer):
    batch_time = AverageMeter()
    rgb_losses = AverageMeter()
    ske_losses = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    data_time = AverageMeter()

    if args.no_partialbn:
        model.module.partialBN(False)
    else:
        model.module.partialBN(True)

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, ske_joint, boxes) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        input = input.cuda()
        target = target.cuda()
        boxes = boxes.cuda()
        output = model(input, ske_joint, boxes)
        if type(output) is tuple:
            rgb_loss = criterion(output[0], target)
            ske_loss = ske_criterion(output[1], target)
            rgb_losses.update(rgb_loss.item(), input.size(0))
            ske_losses.update(ske_loss.item(), input.size(0))
            loss = rgb_loss + ske_loss
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output[0].data, target, topk=(1, 5))
        else:
            loss = criterion(output, target)
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        loss.backward()

        if args.clip_gradient is not None:
            total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)

        optimizer.step()
        optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        iteration = epoch * len(train_loader) + i
        if i % args.print_freq == 0:
            tf_writer.add_scalar('train_loss', losses.avg, iteration)
            tf_writer.add_scalar('rgb_loss', rgb_losses.avg, iteration)
            tf_writer.add_scalar('ske_loss', ske_losses.avg, iteration)
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Ske_Loss {ske_loss.val:.4f} ({ske_loss.avg:.4f})\t'
                      'RGB_Loss {rgb_loss.val:.4f} ({rgb_loss.avg:.4f})\t'
                      'Total_Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, ske_loss=ske_losses, rgb_loss=rgb_losses, top1=top1, top5=top5,
                lr=optimizer.param_groups[-1]['lr'] * 0.1))  # TODO
            print(output)
            log.write(output + '\n')
            log.flush()

    tf_writer.add_scalar('acc/train_top1', top1.avg, epoch)
    tf_writer.add_scalar('acc/train_top5', top5.avg, epoch)
    tf_writer.add_scalar('lr/rgb_lr', optimizer.param_groups[-1]['lr'], epoch)
    tf_writer.add_scalar('lr/gcn_lr', optimizer.param_groups[6]['lr'], epoch)


def validate(val_loader, model, criterion, ske_criterion, epoch, log=None, tf_writer=None):
    batch_time = AverageMeter()
    rgb_losses = AverageMeter()
    ske_losses = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    data_time = AverageMeter()
    CM = metric.ConfusionMatrix(args.num_class)
    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target, ske_joint, boxes) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()
            boxes = boxes.cuda()
            data_time.update(time.time() - end)
            # compute output
            output = model(input, ske_joint, boxes=boxes)
            if type(output) is tuple:
                rgb_loss = criterion(output[0], target)
                ske_loss = ske_criterion(output[1], target)
                rgb_losses.update(rgb_loss.item(), input.size(0))
                ske_losses.update(ske_loss.item(), input.size(0))
                loss = rgb_loss + ske_loss
                # measure accuracy and record loss
                prec1, prec5 = accuracy(output[0].data, target, topk=(1, 5))
                output = output[0]
            else:
                loss = criterion(output, target)
                # measure accuracy and record loss
                prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            CM.update(target, output)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:

                output = ('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'Ske_Loss {ske_loss.val:.4f} ({ske_loss.avg:.4f})\t'
                          'RGB_Loss {rgb_loss.val:.4f} ({rgb_loss.avg:.4f})\t'
                          'Total_Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, data_time=data_time, loss=losses, ske_loss=ske_losses,
                    rgb_loss=rgb_losses,
                    top1=top1, top5=top5))
                print(output)
                if log is not None:
                    log.write(output + '\n')
                    log.flush()
        recall = 0.0
        for p in CM.recall():
            recall += p
        recall_avg = recall / (args.num_class)

    if tf_writer is not None:
        tf_writer.add_scalar('loss/test', losses.avg, epoch)
        tf_writer.add_scalar('loss/rgb_loss', rgb_losses.avg, epoch)
        tf_writer.add_scalar('loss/ske_loss', ske_losses.avg, epoch)
        tf_writer.add_scalar('acc/test_top1', top1.avg, epoch)
        tf_writer.add_scalar('acc/test_top5', top5.avg, epoch)

    return recall_avg


def save_checkpoint(state, is_best):
    dir = args.root_model
    if not os.path.isdir(dir):
        os.makedirs(dir)
    filename = os.path.join(dir, "checkpoint.pth")
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth', 'best.pth'))


def adjust_learning_rate(optimizer, epoch, lr_type, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if lr_type == 'step':
        decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
        lr = args.lr * decay
        decay = args.weight_decay

    elif lr_type == 'cos':
        import math
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * epoch / args.epochs))
        decay = args.weight_decay
    else:
        raise NotImplementedError

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']


def check_rootfolders():
    """Create log and model folder"""
    folders_util = [args.root_log, args.root_model]
    if os.path.exists(args.root_model) and args.root_model != 'exp/test':
        print('The directory already exists, if you want to cover it?[y/n]')
        is_cover = input()
        assert is_cover.upper() not in ['N', 'NOT'], 'need to select a new files path!'

    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.makedirs(folder)


if __name__ == '__main__':
    main()
