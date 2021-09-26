import os
import numpy as np
import shutil
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from tensorboardX import SummaryWriter
from tools import metric
from opts import parser
from ops.utils import get_logger, AverageMeter, accuracy
from archs.i3d_model import I3D

from ops import videotransforms
from archs.fusion_i3d import fusion
from ops.drive_dataset_with_keypoint_i3d import Drive as Dataset

SEED = 777
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    global best_recall, args
    args = parser.parse_args()
    best_recall = 0
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    if not os.path.isdir(args.root_model):
        os.makedirs(args.root_model)

    global logger, writer
    logger = get_logger(args)
    logger.info('SEED {}\nruntime args\n{}\n\n'.format(SEED, args))
    writer = SummaryWriter(args.root_model)

    if args.arch == 'i3d':
        model = I3D(num_classes=args.num_class)
        pretrained_dict = torch.load(args.tune_from)

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'conv3d_0c_1x1' not in k}
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    elif args.arch == 'i3d_all':
        model = fusion(first=args.first, second=args.second,
                       stride=args.gcn_stride, patch_size=args.patch_size,
                       concat_layer=args.concat_layer, xyc=args.xyc, bn=args.bn)
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
                    if k not in model_dict and 'st_gcn.' + k in model_dict:
                        # print('=> Load after add st_gcn.: ', k)
                        replace_dict_gcn.append((k, 'st_gcn.' + k))
                    else:
                        print(k)
            print(("=> fine-tuning from '{}'".format(args.tune_from)))
            sd = torch.load(args.tune_from)
            for k, v in sd.items():
                if k not in model_dict and 'cnn.' + k in model_dict:
                    # print('=> Load after add cnn: ', k)
                    replace_dict_cnn.append((k, 'cnn.' + k))
                else:
                    print(k)
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
            print('=> New dataset, do not load fc weights')
            new_sd = {k: v for k, v in new_sd.items() if 'fc' not in k}
            new_sd = {k: v for k, v in new_sd.items() if 'conv3d_0c_1x1.conv3d' not in k}
            model_dict.update(new_sd)
            model.load_state_dict(model_dict)
    model = nn.DataParallel(model).cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0000001)

    if args.arch == 'c3d':
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.5)
    else:
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [50, 100], gamma=0.1)

    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch'] - 1
            best_recall = checkpoint['best_recall']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(("=> loaded checkpoint (epoch {})"
                   .format(checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    train_transforms = transforms.Compose(
        [videotransforms.RandomCrop(args.input_size), videotransforms.RandomHorizontalFlip()]
    )
    test_transforms = transforms.Compose(
        [videotransforms.CenterCrop(args.input_size)]
    )

    dataset = Dataset(args.root, args.train_split, args.task, args.view, 'train', transforms=train_transforms)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True
    )
    val_dataset = Dataset(args.root, args.val_split, args.task, args.view, 'val', transforms=test_transforms)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True
    )
    criterion = torch.nn.CrossEntropyLoss().cuda()
    ske_criterion = torch.nn.CrossEntropyLoss().cuda()
    epoch_time = AverageMeter()
    for epoch in range(args.start_epoch, args.epochs + 1):
        epoch_start = time.time()
        logger.info('Epoch {}/{}'.format(epoch, args.epochs))
        logger.info('-' * 10)
        train(model, dataloader, optimizer, criterion, ske_criterion, epoch)
        if epoch % 5 == 0 or epoch == 1:
            acc = val(model, val_dataloader, criterion, ske_criterion, epoch)
            is_best = acc > best_recall
            best_recall = max(acc, best_recall)
            logger.info(
                'epoch time: {:.3f}, {} epoch Val Recall: {:.4f} Best Val Recall: {:.4f}'.format(epoch_time.avg, epoch,
                                                                                                 acc, best_recall))
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_recall': best_recall,
            }, is_best)
        lr_scheduler.step()
        epoch_time.update(time.time() - epoch_start)


def train(model, train_loader, optimizer, criterion, ske_criterion, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    rgb_losses = AverageMeter()
    ske_losses = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.train()
    optimizer.zero_grad()
    end = time.time()
    for i, (input, target, ske_joint, boxes) in enumerate(train_loader):
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
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        batch_time.update(time.time() - end)
        writer.add_scalar('train_data/loss', losses.val, epoch * len(train_loader) + i + 1)
        end = time.time()
        iteration = epoch * len(train_loader) + i
        if i % 10 == 0:
            writer.add_scalar('train_loss', losses.avg, iteration)
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
                lr=optimizer.param_groups[-1]['lr']))  # TODO
            logger.info(output + '\n')
            batch_time.reset()
            data_time.reset()


@torch.no_grad()
def val(model, val_loader, criterion, ske_criterion, epoch):
    batch_time = AverageMeter()
    rgb_losses = AverageMeter()
    ske_losses = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    data_time = AverageMeter()
    CM = metric.ConfusionMatrix(34)
    model.eval()
    end = time.time()
    for i, (input, target, ske_joint, boxes) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda()
        boxes = boxes.cuda()
        data_time.update(time.time() - end)
        output = model(input, ske_joint, boxes)
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
        writer.add_scalar('val_data/loss', losses.val, epoch * len(val_loader) + i + 1)
        if i % 10 == 0:
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
            if logger is not None:
                logger.info(output + '\n')
    recall = 0.0
    for p in CM.recall():
        recall += p
    recall_avg = recall / (args.num_class)

    if writer is not None:
        writer.add_scalar('loss/test', losses.avg, epoch)
        writer.add_scalar('acc/test_top1', top1.avg, epoch)
        writer.add_scalar('acc/test_top5', top5.avg, epoch)
    return recall_avg


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


def save_checkpoint(state, is_best):
    dir = args.root_model
    if not os.path.isdir(dir):
        os.makedirs(dir)
    filename = os.path.join(dir, "checkpoint.pth")
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth', 'best.pth'))


if __name__ == '__main__':
    main()
