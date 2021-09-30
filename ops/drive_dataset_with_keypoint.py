import os
import os.path
import numpy as np
import random
import csv
from tqdm import tqdm
import cv2
import torch
import torch.utils.data as data_utl
from numpy.random import randint
from PIL import Image
from opts import parser
from .transforms import GroupRandomHorizontalFlip as flip

args = parser.parse_args()

import json
import torch.nn.functional as F


def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)
    
    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3, 0, 1, 2]))


def load_rgb_frames(image_dir, vid, index):
    # try:
    #     img = cv2.imread(os.path.join(image_dir, vid, 'img_' + str(index).zfill(5) + '.jpg'))[:, :, [2, 1, 0]]
    # except TypeError:
    #     print(image_dir, vid, index)
    # # img = (img/255.)*2 - 1
    # return np.asarray(img, dtype=np.float32)
    return [Image.open(os.path.join(image_dir, vid, 'img_' + str(index).zfill(5) + '.jpg')).convert('RGB')]


def make_dataset(split_file, task, mode, view, patch_size=3):
    dataset = []
    if task == 'tasklevel':
        with open('/home/nigengqin/drive/drive_dataset/annotations/activities_3s/first_level.csv') as f:
            reader = csv.reader(f)
            label_list = [row[1] for row in reader]
    elif task == 'midlevel':
        with open('/home/nigengqin/drive/drive_dataset/annotations/activities_3s/second_level.csv') as f:
            reader = csv.reader(f)
            label_list = [row[1] for row in reader]
    elif task == 'objectlevel':
        with open('/home/nigengqin/drive/drive_dataset/annotations/activities_3s/third_level_action.csv') as f:
            reader = csv.reader(f)
            label_list = [row[1] for row in reader]

    # prepare for getting skeleton info
    keypoint_index = [1, 0, 15, 16, 2, 3, 4, 5, 6, 7, 8, 9, 12]
    video_pose_path = args.skeleton_json
    node_list = ['neck', 'nose', 'lEye', 'rEye', 'lShoulder', 'lElbow', 'lWrist', 'rShoulder', 'rElbow',
                 'rWrist', 'midHip', 'lHip', 'rHip']
    # node_list = [2,3,4, 6,7,8, 10,11,12, 18,19,20, 26,27,28, 34,35,36, 42,43,44, 54,55,56, 58,59,60, 70,71,72, 74,75,76, 82,83,84, 90,91,92]

    if mode in ['eval', 'test']:
        print('eval view: ', view)
    elif mode == 'train':
        print('train view：', view)

    pose_json = {}
    json_path = ''
    view_name = view.split('/')[-1]  # kinect color bug
    view_file = split_file.replace('*', view_name)
    view_file = view_file.replace('+', task)
    interval = (patch_size - 1) / 2
    with open(view_file, 'r') as f:
        data = csv.reader(f)
        for row in tqdm(data):  # every row is a annotation about a action in one video
            if row[1] == 'file_id':
                continue
            vid = view + '/' + row[1]
            start = int(row[3])
            end = int(row[4])
            label = label_list.index(row[5])
            if not pose_json or vid not in json_path:
                json_path = os.path.join(video_pose_path, vid,
                                         'video.json')  # put in if statement, so that each video only read json one time
                f = open(json_path, 'r')
                pose_json = json.load(f)
                f.close()
            skeleton = []
            boxes_list = []
            for frame_idx in range(start, end + 1):
                frame_skeleton = []
                boxes = []
                if pose_json[frame_idx - 1]['img_name'] != 'img_' + str(frame_idx).zfill(5):
                    print('image index error')
                people = pose_json[frame_idx - 1]['keypoints']['people']

                if people:
                    pose_keypoints_2d = people[0]['pose_keypoints_2d']
                    for i in keypoint_index:  # get predefined 13 points
                        i = i * 3
                        x = pose_keypoints_2d[i]
                        y = pose_keypoints_2d[i + 1]
                        c = pose_keypoints_2d[i + 2]
                        # frame_skeleton.append(float(x))
                        # frame_skeleton.append(float(y))
                        # roi
                        frame_skeleton.append(float(x))
                        frame_skeleton.append(float(y))
                        if args.xyc:
                            frame_skeleton.append(c)

                        if x == 0 or y == 0:
                            x = 0.5
                            y = 0.5
                        x = round(x * args.spatial_size)
                        y = round(y * args.spatial_size)
                        x1 = x - interval
                        y1 = y - interval
                        x2 = x + interval
                        y2 = y + interval
                        box = [x1 - 0.5, y1 - 0.5, x2 + 0.5, y2 + 0.5]
                        boxes.append(box)

                else:  # if no keypoint, just append 0
                    for i in range(len(keypoint_index)):
                        x = 0
                        y = 0
                        c = 0
                        frame_skeleton.append(float(x))
                        frame_skeleton.append(float(y))
                        if args.xyc:
                            frame_skeleton.append(c)
                        # frame_skeleton.append(float(x))
                        # frame_skeleton.append(float(y))

                        if x == 0 or y == 0:
                            x = 0.5
                            y = 0.5
                        x = round(x * args.spatial_size)
                        y = round(y * args.spatial_size)
                        x1 = x - interval
                        y1 = y - interval
                        x2 = x + interval
                        y2 = y + interval
                        box = [x1 - 0.5, y1 - 0.5, x2 + 0.5, y2 + 0.5]
                        boxes.append(box)
                skeleton.append(frame_skeleton)  # skeleton shape: num_frames*26
                boxes_list.append(boxes)
            dataset.append((view, vid, start, end, label, skeleton, boxes_list))
            if args.debug:
                if len(dataset) == 200:
                    break
    return dataset


class Drive(data_utl.Dataset):

    def __init__(self, root, split_file, view, patch_size, task='midlevel', mode='train', transforms=None,
                 num_segments=8, gcn_segments=64):
        self.root = root
        self.num_segments = num_segments
        self.gcn_segments = gcn_segments
        self.new_length = 1
        self.data = make_dataset(split_file, task, mode, view, patch_size)
        self.split_file = split_file
        self.transforms = transforms
        self.mode = mode
        self.patch_size = patch_size
        self.flip_ske_tensor = torch.zeros(gcn_segments, 13, 3) if args.xyc else torch.zeros(gcn_segments, 13, 2)
        self.flip_ske_tensor[:, :, 0] = 1
        self.flip_box_tensor = torch.zeros(gcn_segments, 13, 4)
        self.flip_box_tensor[:, :, 0] = args.spatial_size - args.patch_size
        self.flip_box_tensor[:, :, 2] = args.spatial_size - args.patch_size

    def _sample_indices(self, start, end, target, num_segments):
        assert type(target) == int, 'target should be a int'
        num_frames = end - start + 1
        average_duration = (num_frames) // num_segments
        if average_duration > 0:
            chosen_list = np.multiply(list(range(num_segments)), average_duration) + randint(
                average_duration)  # ,size=self.num_segments)
        # elif num_frames > self.num_segments:
        #     offsets = np.sort(randint(num_frames - self.new_length + 1, size=self.num_segments))
        else:
            end = num_segments - num_frames
            chosen_list = list(range(num_frames)) + [num_frames - 1] * end

        label_list = len(chosen_list) * [target]  # each chosen frame has a label
        return chosen_list, label_list  # for this two list, different video should be same length

    def _get_val_indices(self, start, end, target, num_segments):
        assert type(target) == int, 'target should be a int'
        num_frames = end - start + 1
        if num_frames > num_segments:
            tick = (num_frames) / float(num_segments)
            chosen_list = np.array(
                [int(tick / 2.0 + tick * x) for x in range(num_segments)])  # get middle frame of each segments
        else:
            end = num_segments - num_frames
            chosen_list = list(range(num_frames)) + [num_frames - 1] * end

        label_list = len(chosen_list) * [target]  # each chosen frame has a label
        return chosen_list, label_list  # for this two list, different video should be same length

    def __getitem__(self, index):

        view, vid, start, end, label, ske, boxes_list = self.data[index]
        boxes = np.array(boxes_list).astype(np.float)
        boxes = torch.tensor(boxes)
        ske = np.array(ske).astype(np.float32)
        if args.xyc:
            ske = torch.tensor(ske.reshape((end - start + 1, 13, 3)))
        else:
            ske = torch.tensor(ske.reshape((end - start + 1, 13, 2)))
        if ske.shape[0] < self.num_segments:
            ske = list(ske)
            ske = torch.cat([i.unsqueeze(0) for i in ske] + (self.num_segments - len(ske)) * [ske[-1].unsqueeze(0)],
                            dim=0)
        # correct sampling method
        if self.mode == 'train':
            segment_indices, labels = self._sample_indices(start, end, label, self.num_segments)

        elif self.mode in ['eval', 'test']:
            segment_indices, labels = self._get_val_indices(start, end, label, self.num_segments)
        else:
            raise Exception('mode error!')
        segment_indices.sort()
        ske = ske[segment_indices, :, :]
        boxes = boxes[segment_indices, :]

        images = list()
        for seg_ind in segment_indices:
            p = int(seg_ind) + start
            for i in range(self.new_length):
                imgs = load_rgb_frames(self.root, vid, p)
                images.extend(imgs)
        # concat at channel dimension
        # images = np.concatenate(images, axis=2)  # 256*256*3
        process_data = self.transforms(images)  # torch.Size([24, 224, 224])
        return process_data, torch.from_numpy(np.array(label)), ske.permute(2, 0, 1).unsqueeze(3), boxes

    def __len__(self):
        return len(self.data)
