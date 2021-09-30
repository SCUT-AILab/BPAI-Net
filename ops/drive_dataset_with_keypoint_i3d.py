import os
import os.path
import numpy as np
import random
import csv
import cv2
import torch
import torch.utils.data as data_utl
from tqdm import tqdm
from opts import parser
args = parser.parse_args()
import json
from PIL import Image
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


def load_rgb_frames(image_dir, vid, num_frame):
    frames = []
    for i in num_frame:
        if i == -1:
            img = np.zeros((256, 252, 3))
        else:
            # print(image_dir, vid)
            try:
                #img = cv2.imread(os.path.join(image_dir, vid, 'img_' + str(i).zfill(5) + '.jpg'))[:, :, [2, 1, 0]]
                img = Image.open(os.path.join(image_dir, vid, 'img_' + str(i).zfill(5) + '.jpg')).convert('RGB')
                img = np.asarray(img,dtype=np.float32)
            except TypeError:
                print(image_dir, vid, i)
            # img = (img/255.)*2 - 1
        frames.append(img)
    return np.asarray(frames, dtype=np.float32)


def make_dataset(split_file, task, mode, view, patch_size=7):
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

    if mode in ['eval','test']:
        print('eval view: ', view)
    elif mode == 'train':
        print('train view：', view)

    pose_json = {}
    json_path = ''
    view_name = view.split('/')[-1]  # kinect color bug
    view_file = split_file.replace('*', view_name)
    view_file = view_file.replace('+', task)
    interval = (patch_size - 1)/2
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

                if people:  # not None means there are extracted keypoints
                    pose_keypoints_2d = people[0]['pose_keypoints_2d']
                    for i in keypoint_index:  # get predefined 13 points
                        i = i * 3
                        # y,x,bug
                        x = pose_keypoints_2d[i]
                        y = pose_keypoints_2d[i + 1]
                        c = pose_keypoints_2d[i + 2]
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
                        box = [x1-0.5, y1-0.5, x2+0.5, y2+0.5]
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
                        if x==0 or y == 0:
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

            dataset.append((view, vid, start, end, label, skeleton,boxes_list))
            if args.debug:
                if len(dataset) == 200:
                    break
    return dataset




class Drive(data_utl.Dataset):
    def __init__(self, root, split_file,task,view, mode='train', transforms=None,num_segments=64,gcn_segments=32):
        self.root = root
        self.num_segments = num_segments
        self.gcn_segments = gcn_segments
        view = view.split('/')[-1]
        self.data = make_dataset(split_file, task, mode, view)
        self.split_file = split_file
        self.transforms = transforms
        self.mode = mode
    def _sample_indices(self, start, end):
        num_frames = end - start + 1
        if end - start < self.num_segments:
            index_list = list(range(start, end + 1)) + [end] * (self.num_segments - num_frames)
        else:
            random_start = random.randint(start, end - 64)
            index_list = list(range(random_start, random_start + 64))
        gcn_index_list = np.linspace(3,59,8,dtype=int).tolist()
        gcn_list = [index_list[i]-start for i in gcn_index_list]
        return index_list,gcn_list

    def _get_val_indices(self, start, end):
        num_frames = end - start + 1
        if end - start < self.num_segments:
            index_list = list(range(start, end + 1)) + [end] * (self.num_segments - num_frames)
        else:
            select_start = (num_frames - self.num_segments) // 2 + start
            select_end = select_start + self.num_segments
            index_list = list(range(select_start, select_end))
        gcn_index_list = np.linspace(3, 59, 8,dtype=int).tolist()
        gcn_list = [index_list[i]-start for i in gcn_index_list]
        return index_list,gcn_list

    def __getitem__(self, index):

        view, vid, start, end, label, ske, boxes_list = self.data[index]
        boxes = np.array(boxes_list).astype(np.float)
        boxes = torch.tensor(boxes)
        ske = np.array(ske).astype(np.float32)
        if args.xyc:
            ske = torch.tensor(ske.reshape((end - start + 1, 13, 3)))
        else:
            ske = torch.tensor(ske.reshape((end - start + 1, 13, 2)))

        # correct sampling method
        if self.mode == 'train':
            image_indices, ske_indices = self._sample_indices(start, end)

        elif self.mode in ['val', 'test']:
            image_indices, ske_indices = self._get_val_indices(start, end)
        else:
            raise Exception('mode error!')

        ske = ske[ske_indices, :, :]  # till this line, get the needed skeleton point for training
        boxes = boxes[ske_indices,:]

        imgs = load_rgb_frames(self.root, vid, image_indices)
        imgs = self.transforms(imgs)
        return video_to_tensor(imgs), torch.from_numpy(np.array(label)), ske.permute(2, 0, 1).unsqueeze(3),boxes

    def __len__(self):
        return len(self.data)
