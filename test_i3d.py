import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from ops.utils import get_logger, AverageMeter, accuracy
from archs.i3d_model import I3D
from tools.metric import ConfusionMatrix
import random
from archs.fusion_i3d import fusion
from ops import videotransforms
from opts import parser
from ops.drive_dataset_with_keypoint_i3d import Drive as Dataset
SEED = 777
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

args = parser.parse_args()
test_transforms = transforms.Compose(
    [videotransforms.CenterCrop(args.input_size)]
)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
def load_model(checkpoint_path):
    if args.arch == 'i3d':
        model = I3D(num_classes=args.num_class)
    elif args.arch == 'i3d_all':
        model = fusion(first=args.first, second=args.second,
                       stride=args.gcn_stride, patch_size=args.patch_size,
                       concat_layer=args.concat_layer, xyc=args.xyc, bn=args.bn)
    pretrained_dict = torch.load(checkpoint_path)
    state_dict = pretrained_dict['state_dict']
    epoch = pretrained_dict['epoch']
    model.cuda()
    model = nn.DataParallel(model)
    model.load_state_dict(state_dict, strict=False)
    return model,epoch


def main():

    model,epoch = load_model(args.model_path)
    global logger
    logger = get_logger(args, 'test')
    val_dataset = Dataset(args.root, args.val_split, args.task, args.view, 'val', test_transforms)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True
    )
    test_dataset = Dataset(args.root, args.test_split, args.task, args.view, 'test', test_transforms)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True
    )
    val(model,epoch,val_dataloader)
    test(model,epoch,test_dataloader)


@torch.no_grad()
def val(model,epoch,val_dataloader):
    CM = ConfusionMatrix(34)
    top1 = AverageMeter()
    top5 = AverageMeter()
    rgb_losses = AverageMeter()
    ske_losses = AverageMeter()
    tot_loss = AverageMeter()

    model.eval()
    logger.info("The best model epoch is :{}".format(epoch))
    for i, (input, target, ske_joint, bbox) in enumerate(val_dataloader):
        batch_size = input.size(0)
        input = input.cuda()
        target = target.cuda()
        bbox = bbox.cuda()
        per_frame_logits = model(input, ske_joint, bbox)
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
        if i % 20 == 0 or i == len(val_dataloader) - 1:
            output = ('Val: [{0}/{1}]\t'
                      'Ske_Loss {ske_loss.val:.4f} ({ske_loss.avg:.4f})\t'
                      'RGB_Loss {rgb_loss.val:.4f} ({rgb_loss.avg:.4f})\t'
                      'Loss {tot_loss.val:.4f} ({tot_loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(val_dataloader) - 1, tot_loss=tot_loss, ske_loss=ske_losses, rgb_loss=rgb_losses,
                top1=top1, top5=top5))
            logger.info(output)
    prec = 0.0
    for p in CM.precision():
        prec += p
    prec_avg = prec / 33
    recall = 0.0
    for p in CM.recall():
        recall += p
    recall_avg = recall / 33
    logger.info("Val Stage: class-wise precision: {}\nclass-wise recall: {}".format(CM.precision(), CM.recall()))
    logger.info(
        "Val View: {}, top1 acc: {:.2f}, top5 acc: {:.2f}, precision: {:.2f}, recall: {:.2f}".format(
            args.view.split('/')[-1], top1.avg,
            top5.avg, prec_avg * 100, recall_avg * 100))


@torch.no_grad()
def test(model,epoch,test_dataloader):
    top1 = AverageMeter()
    top5 = AverageMeter()
    rgb_losses = AverageMeter()
    ske_losses = AverageMeter()
    tot_loss = AverageMeter()
    model.eval()
    logger.info("The best model epoch is :{}".format(epoch))
    CM = ConfusionMatrix(34)
    for i, (input, target, ske_joint, bbox) in enumerate(test_dataloader):
        batch_size = input.size(0)
        input = input.cuda()
        target = target.cuda()
        bbox = bbox.cuda()
        per_frame_logits = model(input, ske_joint, bbox)
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
        if i % 20 == 0 or i == len(test_dataloader) - 1:
            output = ('Test: [{0}/{1}]\t'
                      'Ske_Loss {ske_loss.val:.4f} ({ske_loss.avg:.4f})\t'
                      'RGB_Loss {rgb_loss.val:.4f} ({rgb_loss.avg:.4f})\t'
                      'Loss {tot_loss.val:.4f} ({tot_loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(test_dataloader) - 1, tot_loss=tot_loss, ske_loss=ske_losses, rgb_loss=rgb_losses,
                top1=top1, top5=top5))
            logger.info(output)
    prec = 0.0
    for p in CM.precision():
        prec += p
    prec_avg = prec / 34
    recall = 0.0
    for p in CM.recall():
        recall += p
    recall_avg = recall / 34
    logger.info("Test Stage: class-wise precision: {}\nclass-wise recall: {}".format(CM.precision(), CM.recall()))
    logger.info(
        "Test View: {}, top1 acc: {:.2f}, top5 acc: {:.2f}, precision(34): {:.2f}, recall(34): {:.2f}".format(
            args.view.split('/')[-1], top1.avg, top5.avg,
            prec_avg * 100, recall_avg * 100))


if __name__ == '__main__':
    main()
