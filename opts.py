import argparse
parser = argparse.ArgumentParser(description="PyTorch implementation of BPAI-Net")
parser.add_argument('--dataset', type=str,default='drive', choices=['drive', 'pcl'])
parser.add_argument('--modality', type=str, default='RGB',choices=['RGB', 'Flow'])

parser.add_argument('--train_list', type=str, default="")
parser.add_argument('--val_list', type=str, default="")
parser.add_argument('--root_path', type=str, default="")
parser.add_argument('--store_name', type=str, default="")
# ========================= Model Configs ==========================
parser.add_argument('--arch', type=str, default="BNInception")
parser.add_argument('--num_segments', type=int, default=8)
parser.add_argument('--consensus_type', type=str, default='avg')
parser.add_argument('--k', type=int, default=3)

parser.add_argument('--dropout', '--do', default=0.5, type=float,
                    metavar='DO', help='dropout ratio (default: 0.5)')
parser.add_argument('--loss_type', type=str, default="nll",
                    choices=['nll','focal'])
parser.add_argument('--img_feature_dim', default=256, type=int, help="the feature dimension for each frame")
parser.add_argument('--suffix', type=str, default=None)
parser.add_argument('--pretrain', type=str, default='imagenet')
parser.add_argument('--tune_from', type=str, default='', help='fine-tune from checkpoint')


# ========================= Learning Configs ==========================
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_type', default='step', type=str,
                    metavar='LRtype', help='learning rate type')
parser.add_argument('--lr_steps', default=[30], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--clip-gradient', '--gd', default=None, type=float,
                    metavar='W', help='gradient norm clipping (default: disabled)')
parser.add_argument('--no_partialbn', '--npb', default=False, action="store_true")

# ========================= Monitor Configs ==========================
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--eval-freq', '-ef', default=5, type=int,
                    metavar='N', help='evaluation frequency (default: 5)')


# ========================= Runtime Configs ==========================
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--gpus', type=str, default=0)
parser.add_argument('--flow_prefix', default="", type=str)
parser.add_argument('--root_log',type=str, default='log')
parser.add_argument('--root_model', type=str, default='/home/datasets/nigengqin_exps/tsm/checkpoint')

parser.add_argument('--shift', default=False, action="store_true", help='use shift for models')
parser.add_argument('--shift_div', default=8, type=int, help='number of div for shift (default: 8)')
parser.add_argument('--shift_place', default='blockres', type=str, help='place for shift (default: stageres)')

parser.add_argument('--temporal_pool', default=False, action="store_true", help='add temporal pooling')
parser.add_argument('--non_local', default=False, action="store_true", help='add non local block')

parser.add_argument('--dense_sample', default=False, action="store_true", help='use dense sample for video dataset')
parser.add_argument('--root', default='/home/nigengqin/drive/drive_dataset/', type=str)
parser.add_argument('--num_class', type=int,default=34)
parser.add_argument('--train_split', default='/home/nigengqin/drive/drive_dataset/annotations/activities_3s/*/+.chunks_90.split_0.train.csv', type=str)
parser.add_argument('--val_split', default='/home/nigengqin/drive/drive_dataset/annotations/activities_3s/*/+.chunks_90.split_0.val.csv', type=str)
parser.add_argument('--test_split', default='/home/nigengqin/drive/drive_dataset/annotations/activities_3s/*/+.chunks_90.split_0.test.csv', type=str)
parser.add_argument('--pcl_anno', type=str,default='annotations/3train3test--train2test/annotation123.json') # for pcldriver
parser.add_argument('--skeleton_json', type=str,default='/home/nigengqin/drive/video_pose/')
parser.add_argument('--task', default='midlevel')
parser.add_argument('--view', type=str, default='inner_mirror')
parser.add_argument('--first', type=str,default='1')
parser.add_argument('--second', type=str,default='layer4')
parser.add_argument('--mode',type=str)
parser.add_argument('--gcn_pretrained',type=str,default=None)
parser.add_argument('--gcn_stride',type=int,default=1)
parser.add_argument('--patch_size',type=int,default=3)
parser.add_argument('--spatial_size',type=int,default=28)
parser.add_argument('--roi',action="store_true",default=False)
parser.add_argument('--xyc',action="store_true",default=False)
parser.add_argument('--bn',action="store_true",default=False)
parser.add_argument('--concat_layer',type=int,default=5)
parser.add_argument('--arch_cnn', type=str, default='')
parser.add_argument('--debug',action='store_true',default=False)
parser.add_argument('--pcl_train_bus',nargs='+',type=int,default=[1,3,4])
parser.add_argument('--gcn_dropout',default=0,type=float)
parser.add_argument('--softmax_dim',default=-1,type=int)
parser.add_argument('-one_person',action="store_true",default=False)
#test
# may contain splits
parser.add_argument('--model_path',type=str)
parser.add_argument('--weights', type=str, default=None)
parser.add_argument('--twice_sample', default=False, action="store_true", help='use twice sample for ensemble')
parser.add_argument('--full_res', default=False, action="store_true",
                    help='use full resolution 256x256 for test as in Non-local I3D')

parser.add_argument('--test_crops', type=int, default=1)
parser.add_argument('--coeff', type=str, default=None)

# for true test
parser.add_argument('--test_list', type=str, default=None)
parser.add_argument('--csv_file', type=str, default=None)

parser.add_argument('--softmax', default=False, action="store_true", help='use softmax')

parser.add_argument('--max_num', type=int, default=-1)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--crop_fusion_type', type=str, default='avg')
parser.add_argument('--num_set_segments',type=int, default=1,help='TODO: select multiply set of n-frames from a video')
