import time

import torch.nn.parallel
import torch.optim
from ops.models import TSN
from ops.transforms import *
from ops.drive_dataset_with_keypoint import Drive as dataset
from tools.metric import ConfusionMatrix
from opts import parser

args = parser.parse_args()
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import logging

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

SEED = 777
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def load_model(checkpoint_path):
    model = TSN(args.num_class, args.num_segments, args.modality,
                base_model=args.arch,
                consensus_type=args.crop_fusion_type,
                img_feature_dim=args.img_feature_dim,
                pretrain=args.pretrain,
                is_shift=True, shift_div=8, shift_place='blockres',
                first = args.first,
                second = args.second,
                gcn_stride=args.gcn_stride,
                concat_layer=args.concat_layer,
                xyc = args.xyc,
                bn = args.bn,arch_cnn=args.arch_cnn,patch_size=args.patch_size
                )
    pretrained_dict = torch.load(checkpoint_path)
    state_dict = pretrained_dict['state_dict']
    epoch = pretrained_dict['epoch']
    model.cuda()
    model = nn.DataParallel(model)
    model.load_state_dict(state_dict,strict=False)
    return model, epoch


input_size = 224
if args.test_crops == 1:
    cropping = torchvision.transforms.Compose([
        GroupScale(256),
        GroupCenterCrop(input_size),
    ])
this_arch = 'resnet'
input_mean = [0.485, 0.456, 0.406]
input_std = [0.229, 0.224, 0.225]
test_transforms = torchvision.transforms.Compose([
    cropping,
    Stack(roll=(this_arch in ['BNInception', 'InceptionV3'])),
    ToTorchFormatTensor(div=(this_arch not in ['BNInception', 'InceptionV3'])),
    GroupNormalize(input_mean, input_std),
])


def get_logger(args, mode='test'):
    logger = logging.getLogger(__name__)
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler.setFormatter(formatter)
    handler.setLevel(0)
    logger.addHandler(handler)

    date = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    if not os.path.isdir(args.root_log):
        os.makedirs(args.root_log)
    logfile = os.path.join(args.root_log, 'val_test.log')
    file_handler = logging.FileHandler(logfile, mode='w')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger




def main():

    global logger
    logger = get_logger(args, 'test')
    if args.gcn_stride == 2:
        gcn_segments = 64
    else: gcn_segments = args.num_segments
    logger.info('runtime args\n{}\n\n'.format(args))
    logger.info('train set: {},val set: {}'.format(args.view, args.view))
    val_dataset = dataset(args.root, args.val_split, view=args.view, mode='eval', patch_size=args.patch_size,
                          num_segments=args.num_segments, gcn_segments=gcn_segments, transforms=test_transforms)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True
    )
    test_dataset = dataset(args.root, args.test_split, view=args.view, mode='test',patch_size=args.patch_size,
                          num_segments=args.num_segments, gcn_segments=gcn_segments, transforms=test_transforms)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True
    )
    model, epoch = load_model(args.model_path)
    val(model, epoch, val_dataloader)
    test(model, epoch, test_dataloader)


@torch.no_grad()
def val(model, epoch,val_dataloader):
    CM = ConfusionMatrix(args.num_class)
    top1 = AverageMeter()
    top5 = AverageMeter()
    rgb_losses = AverageMeter()
    ske_losses = AverageMeter()
    tot_loss = AverageMeter()

    model.eval()
    logger.info("The best model epoch is :{}".format(epoch))
    for i, (input,target,ske_joint,index) in enumerate(val_dataloader):
        batch_size = input.size(0)
        input = input.cuda()
        target = target.cuda()
        index = index.cuda()
        per_frame_logits = model(input, ske_joint,index)
        if type(per_frame_logits) is tuple:
            rgb_loss = F.cross_entropy(per_frame_logits[0], target)
            ske_loss = F.cross_entropy(per_frame_logits[1], target)
            rgb_losses.update(rgb_loss.item(), input.size(0))
            ske_losses.update(ske_loss.item(), input.size(0))
            loss = rgb_loss + ske_loss
            # measure accuracy and record loss
            prec1, prec5 = accuracy(per_frame_logits[0].data, target, topk=(1, 5))
            per_frame_logits = per_frame_logits[0]
        else:
            prec1, prec5 = accuracy(per_frame_logits.data, target, topk=(1, 5))
            loss = F.cross_entropy(per_frame_logits, target)

        CM.update(target, per_frame_logits)
        tot_loss.update(loss.item(), per_frame_logits.size(0))

        top1.update(prec1.item(), batch_size)
        top5.update(prec5.item(), batch_size)
        if i % args.print_freq == 0 or i == len(val_dataloader)-1:
            output = ('Val: [{0}/{1}]\t'
                      'Ske_Loss {ske_loss.val:.4f} ({ske_loss.avg:.4f})\t'
                      'RGB_Loss {rgb_loss.val:.4f} ({rgb_loss.avg:.4f})\t'
                      'Loss {tot_loss.val:.4f} ({tot_loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(val_dataloader)-1, tot_loss=tot_loss,ske_loss=ske_losses, rgb_loss=rgb_losses,
                top1=top1, top5=top5))
            logger.info(output)
    prec = 0.0
    for p in CM.precision():
        prec += p
    prec_avg = prec / args.num_class
    recall = 0.0
    for p in CM.recall():
        recall += p
    recall_avg = recall / args.num_class
    logger.info("Val Stage: class-wise precision: {}\nclass-wise recall: {}".format(CM.precision(), CM.recall()))
    logger.info(
        "Val View: {}, top1 acc: {:.2f}, top5 acc: {:.2f}, precision: {:.2f}, recall: {:.2f}".format(args.view.split('/')[-1], top1.avg,
                                                                                    top5.avg,prec_avg*100, recall_avg*100))

@torch.no_grad()
def test(model, epoch, test_dataloader):
    top1 = AverageMeter()
    top5 = AverageMeter()
    rgb_losses = AverageMeter()
    ske_losses = AverageMeter()
    tot_loss = AverageMeter()
    model.eval()
    logger.info("The best model epoch is :{}".format(epoch))
    CM = ConfusionMatrix(args.num_class)
    for i, (input, target,ske_joint,index) in enumerate(test_dataloader):
        batch_size = input.size(0)
        input = input.cuda()
        target = target.cuda()
        index = index.cuda()
        per_frame_logits = model(input, ske_joint,index)
        if type(per_frame_logits) is tuple:
            rgb_loss = F.cross_entropy(per_frame_logits[0], target)
            ske_loss = F.cross_entropy(per_frame_logits[1], target)
            rgb_losses.update(rgb_loss.item(), input.size(0))
            ske_losses.update(ske_loss.item(), input.size(0))
            loss = rgb_loss + ske_loss
            # measure accuracy and record loss
            prec1, prec5 = accuracy(per_frame_logits[0].data, target, topk=(1, 5))
            per_frame_logits = per_frame_logits[0]
        else:
            prec1, prec5 = accuracy(per_frame_logits.data, target, topk=(1, 5))
            loss = F.cross_entropy(per_frame_logits, target)
        CM.update(target, per_frame_logits)
        tot_loss.update(loss.item(), per_frame_logits.size(0))

        top1.update(prec1.item(), batch_size)
        top5.update(prec5.item(), batch_size)
        if i % args.print_freq == 0 or i == len(test_dataloader)-1:
            output = ('Test: [{0}/{1}]\t'
                      'Ske_Loss {ske_loss.val:.4f} ({ske_loss.avg:.4f})\t'
                      'RGB_Loss {rgb_loss.val:.4f} ({rgb_loss.avg:.4f})\t'
                      'Loss {tot_loss.val:.4f} ({tot_loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(test_dataloader)-1, tot_loss=tot_loss,ske_loss=ske_losses, rgb_loss=rgb_losses,
                top1=top1, top5=top5))
            logger.info(output)
    prec = 0.0
    for p in CM.precision():
        prec += p
    prec_avg = prec / args.num_class
    recall = 0.0
    for p in CM.recall():
        recall += p
    recall_avg = recall / args.num_class
    logger.info("Test Stage: class-wise precision: {}\nclass-wise recall: {}".format(CM.precision(), CM.recall()))
    logger.info(
            "Test View: {}, top1 acc: {:.2f}, top5 acc: {:.2f}, precision(34): {:.2f}, recall(34): {:.2f}".format(args.view.split('/')[-1], top1.avg,top5.avg,
                                                                         prec_avg*100, recall_avg*100))




if __name__ == '__main__':
    main()
