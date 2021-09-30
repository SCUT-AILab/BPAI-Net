
import os
import math
import json
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, Sampler
import torchvision
import torchvision.transforms as transforms
from time import sleep
from ops.drive_dataset_with_keypoint import video_to_tensor, load_rgb_frames
import numpy as np
from numpy.random import randint
from opts import parser
from PIL import Image

args = parser.parse_args()
import json
from tqdm import tqdm


# from tools.attention_tools import *

class SegmentSampler(Sampler):
    '''
    for training, randomly select one frame in every segment.
    for validation, evenly(one frame per second) select frames in every segment.
    '''

    def __init__(self, data_source, train, batch_size):
        super(SegmentSampler, self).__init__()
        self.data_source = data_source
        self.train = train
        self.batch_size = batch_size
        self.segment_indexes = [list()
                                for _ in range(self.data_source.segment_count)]
        self.fpses = [None, ] * self.data_source.segment_count

        for i, data_pair in enumerate(self.data_source.data_pairs):
            *_, metadata = data_pair
            self.segment_indexes[metadata['segment_count']].append(i)
            self.fpses[metadata['segment_count']] = metadata['fps']
        self.segment_indexes = list(
            filter(lambda item: len(item) != 0, self.segment_indexes))  # 这两个filter把原来的空元素去掉
        self.fpses = list(filter(lambda item: item is not None, self.fpses))
        assert len(self.segment_indexes) == len(self.fpses)

        # if in validation mode, we can generate the index off-line
        if not train:
            index = []
            for fps, segment_index in zip(self.fpses, self.segment_indexes):
                interval = fps
                for i in range(0, len(segment_index), interval):
                    if interval // 2 + i < len(segment_index):
                        index.append(segment_index[interval // 2 + i])
                    else:
                        index.append(
                            segment_index[(len(segment_index) - i) // 2 + i])
            self.index = index

    def __iter__(self):
        if self.train:
            index = []
            for segment_index in self.segment_indexes:
                index.append(random.choice(segment_index))
            random.shuffle(index)
            return iter(index)
        else:
            return iter(self.index)

    def __len__(self):
        if self.train:
            return len(self.fpses)
        else:
            return len(self.index)


def _pts2index_impl(pts, fps):  # get the position of this frame
    fraction, integer = math.modf(pts)
    return int(fps * integer + fraction * 100)


def is_rgb(metadata):
    return metadata['type'] == 0


def is_infrared(metadata):
    return metadata['type'] == 1


def is_high_quality(metadata):
    return metadata['quality'] == 0


def is_daytime(metadata):
    return metadata['lighting'] == 0


def is_night(metadata):
    return metadata['lighting'] == 1


def is_in_classes(metadata, labels):
    return metadata['label'] in labels


def is_not_in_classes(metadata, labels):
    return metadata['label'] not in labels


'''
每一行表示一个video segment的信息
{driver: {
    subset: train/test,
    fps: 该segment的帧率
    lighting: 白天/黑夜
    type: RGB/光流
    annotations: {
        segment: 开始结束时刻，例如16.34-18.24：表示第16秒后的第34帧开始，到第18秒后的第24帧结束
        quality: 视频片段质量高低
        label: 动作类别，目前总8类
        }
    }
}
'''


class BusDeriverDataset3D(Dataset):
    # 3D dataset to get 1st frames of every 8 frames-segment in each action video
    # for ghostnet

    #     anno2index = {
    #         '1-a-1': 0,  # Normal Driving
    #         '1-b-1': 0,  # Normal Driving
    #         '1-c-1': 0,  # Normal Driving
    #         '2-a-1': 1,  # Drinking or Eating (2-a-1 denotes Drinking)
    #         '2-b-1': 1,  # Drinking or Eating (2-b-1 denotes Drinking)
    #         '3-a-1': 1,  # Drinking or Eating (3-a-1 denotes Eating)
    #         '3-b-1': 1,  # Drinking or Eating (3-a-1 denotes Eating)
    #         '4-a-1': 2,  # Playing on the Phone
    #         '4-b-1': 2,  # Playing on the Phone
    #         '5-a-1': 3,  # Calling
    #         '5-b-1': 3,  # Calling
    #         '6-a-1': 4,  # Looking Sideways
    #         '6-b-1': 4,  # Looking Sideways
    #         '7-a-1': 5,  # Fighting (7-a-1 denotes fighting for the steering wheel)
    #         '8-a-1': 5,  # Fighting (8-a-1 denotes pulling the driver)
    #         '9-a-1': 6,  # Fixing Hair
    #         '10-a-1': 7,  # Driving with Single Hand
    #         # '11-a-1': 8,  # Fatigue Driving
    #     }

    #     index2name = {
    #         0: 'Normal_Driving',
    #         1: 'Drinking_or_Eating',
    #         2: 'Playing_on_the_Phone',
    #         3: 'Calling',
    #         4: 'Looking_Sideways',
    #         5: 'Fighting',
    #         6: 'Fixing_Hair',
    #         7: 'Driving_with_Single_Hand',
    #         #         8: 'Fatigue_Driving'
    #     }

    #     # 11 classes
    #     anno2index = {
    #         '1-a-1': 0,  # Normal Driving
    #         '1-b-1': 0,  # Normal Driving
    #         '1-c-1': 0,  # Normal Driving
    #         '2-a-1': 1,  # Drinking (2-a-1 denotes Drinking)
    #         '2-b-1': 1,  # Drinking (2-b-1 denotes Drinking)
    #         '3-a-1': 2,  # Eating (3-a-1 denotes Eating)
    #         '3-b-1': 2,  # Eating (3-b-1 denotes Eating)
    #         '4-a-1': 3,  # Playing the Phone
    #         '4-b-1': 3,  # Playing the Phone
    #         '5-a-1': 4,  # Calling
    #         '5-b-1': 4,  # Calling
    #         '6-a-1': 5,  # Turning Around
    #         '6-b-1': 5,  # Turning Around
    #         '7-a-1': 6,  # grabbing the steering wheel
    #         '8-a-1': 7,  # pulling the driver
    #         '9-a-1': 8,  # Fixing Hair
    #         '10-a-1': 9,  # Driving with Single Hand
    #         '11-a-1': 10,  # Fatigue Driving, Yawning
    #     }

    # index2name = {
    #     0: 'Normal_Driving',
    #     1: 'Drinking',
    #     2: 'Eating',
    #     3: 'Playing_the_Phone',
    #     4: 'Calling',
    #     5: 'Turning_Around',
    #     6: 'pulling the driver',
    #     7: 'grabbing for the steering wheel',
    #     8: 'Fixing_Hair',
    #     9: 'Driving_with_Single_Hand',
    #     10: 'Fatigue_Driving'
    # }

    #     # 11 classes
    #     anno2index = {
    #         '1-a': 0,  # Normal Driving
    #         '1-b': 0,  # Normal Driving
    #         '1-c': 0,  # Normal Driving
    #         '2-a': 1,  # Drinking (2-a-1 denotes Drinking)
    #         '2-b': 1,  # Drinking (2-b-1 denotes Drinking)
    #         '3-a': 2,  # Eating (3-a-1 denotes Eating)
    #         '3-b': 2,  # Eating (3-b-1 denotes Eating)
    #         '4-a': 3,  # Playing the Phone
    #         '4-b': 3,  # Playing the Phone
    #         '5-a': 4,  # Calling
    #         '5-b': 4,  # Calling
    #         '6-a': 5,  # Turning Around
    #         '6-b': 5,  # Turning Around
    #         '7-a': 6,  # pulling the driver
    #         '8-a': 7,  # grabbing the steering wheel
    #         '9-a': 8,  # Fixing Hair
    #         '10-a': 9,  # Driving with Single Hand
    #         '11-a': 10,  # Fatigue Driving, Yawning
    #     }

    index2name = {
        0: 'Normal_Driving',
        1: 'Drinking_with_Left_Hand pre',
        2: 'Drinking_with_Left_Hand',
        3: 'Drinking_with_Left_Hand post',
        4: 'Drinking_with_Right_Hand pre',
        5: 'Drinking_with_Right_Hand',
        6: 'Drinking_with_Right_Hand post',
        7: 'Eating_with_Left_Hand pre',
        8: 'Eating_with_Left_Hand',
        9: 'Eating_with_Left_Hand post',
        10: 'Eating_with_Right_Hand pre',
        11: 'Eating_with_Right_Hand',
        12: 'Eating_with_Right_Hand post',
        13: 'Playing the Phone_with_Left_Hand pre',
        14: 'Playing the Phone_with_Left_Hand',
        15: 'Playing the Phone_with_Left_Hand post',
        16: 'Playing the Phone_with_Right_Hand pre',
        17: 'Playing the Phone_with_Right_Hand',
        18: 'Playing the Phone_with_Right_Hand post',
        19: 'Calling_with_Left_Hand pre',
        20: 'Calling_with_Left_Hand',
        21: 'Calling_with_Left_Hand post',
        22: 'Calling_with_Right_Hand pre',
        23: 'Calling_with_Right_Hand',
        24: 'Calling_with_Right_Hand post',
        25: 'Turning Left pre',
        26: 'Turning Left',
        27: 'Turning Left post',
        28: 'Turning Right pre',
        29: 'Turning Right',
        30: 'Turning Right post',
        31: 'pulling the driver',
        32: 'grabbing the steering wheel',
        33: 'Fixing Hair pre',
        34: 'Fixing Hair',
        35: 'Fixing Hair post',
        36: 'Driving with Single Hand pre',
        37: 'Driving with Single Hand',
        38: 'Driving_with_Single_Hand post',
        39: 'Fatigue_Driving'
    }

    anno2index = {
        '1-a-1': 0,  # Normal Driving
        '1-b-1': 0,  # Normal Driving
        '1-c-1': 0,  # Normal Driving
        '2-a-0': 1,  # Drinking_with_Left_Hand pre
        '2-a-1': 2,  # Drinking_with_Left_Hand
        '2-a-2': 3,  # Drinking_with_Left_Hand post
        '2-b-0': 4,  # Drinking_with_Right_Hand pre
        '2-b-1': 5,  # Drinking_with_Right_Hand
        '2-b-2': 6,  # Drinking_with_Right_Hand post
        '3-a-0': 7,  # Eating_with_Left_Hand pre
        '3-a-1': 8,  # Eating_with_Left_Hand
        '3-a-2': 9,  # Eating_with_Left_Hand post
        '3-b-0': 10,  # Eating_with_Right_Hand pre
        '3-b-1': 11,  # Eating_with_Right_Hand
        '3-b-2': 12,  # Eating_with_Right_Hand post
        '4-a-0': 13,  # Playing the Phone_with_Left_Hand pre
        '4-a-1': 14,  # Playing the Phone_with_Left_Hand
        '4-a-2': 15,  # Playing the Phone_with_Left_Hand post
        '4-b-0': 16,  # Playing the Phone_with_Right_Hand pre
        '4-b-1': 17,  # Playing the Phone_with_Right_Hand
        '4-b-2': 18,  # Playing the Phone_with_Right_Hand post
        '5-a-0': 19,  # Calling_with_Left_Hand pre
        '5-a-1': 20,  # Calling_with_Left_Hand
        '5-a-2': 21,  # Calling_with_Left_Hand post
        '5-b-0': 22,  # Calling_with_Right_Hand pre
        '5-b-1': 23,  # Calling_with_Right_Hand
        '5-b-2': 24,  # Calling_with_Right_Hand post
        '6-a-0': 25,  # Turning Left pre
        '6-a-1': 26,  # Turning Left
        '6-a-2': 27,  # Turning Left post
        '6-b-0': 28,  # Turning Right pre
        '6-b-1': 29,  # Turning Right
        '6-b-2': 30,  # Turning Right post
        '7-a-1': 31,  # pulling the driver
        '8-a-1': 32,  # grabbing the steering wheel
        '9-a-0': 33,  # Fixing Hair pre
        '9-a-1': 34,  # Fixing Hair
        '9-a-2': 35,  # Fixing Hair post
        '10-a-0': 36,  # Driving with Single Hand pre
        '10-a-1': 37,  # Driving with Single Hand
        '10-a-2': 38,  # Driving with Single Hand post
        '11-a-1': 39,  # Fatigue Driving, Yawning
    }

    def __init__(self, root, anno_path, train, filters,
                 transforms=None, target_transforms=None, n_frames=8, gcn_segments=64, interval=0, writer=None,
                 random_shift=True, test_mode=False, patch_size=7):
        self.root = root
        self.anno_path = anno_path
        self.train = train
        self.filters = filters
        self.transforms = transforms
        self.target_transforms = target_transforms

        self.segment_count = 0
        self.data_pairs = []

        self.num_segments = n_frames  # awkward name, means num_frames
        self.gcn_segments = gcn_segments
        self.interval = interval
        self.writer = writer
        self.random_shift = random_shift
        self.test_mode = test_mode

        self.patch_size = patch_size
        # prepare for getting skeleton info
        self.keypoint_index = [1, 0, 15, 16, 2, 3, 4, 5, 6, 7, 8, 9, 12]
        self.video_pose_path = args.skeleton_json
        self.node_list = ['neck', 'nose', 'lEye', 'rEye', 'lShoulder', 'lElbow', 'lWrist', 'rShoulder', 'rElbow',
                          'rWrist', 'midHip', 'lHip', 'rHip']
        # node_list = [2,3,4, 6,7,8, 10,11,12, 18,19,20, 26,27,28, 34,35,36, 42,43,44, 54,55,56, 58,59,60, 70,71,72, 74,75,76, 82,83,84, 90,91,92]

        self.new_length = 1
        self._parse_annotation()

        # Here, just to be sure, we check all the frames whether exist in the disk
        for data_pair in self.data_pairs:
            view, driver, start, end, label, ske, boxes_list, metadata = data_pair

            for index in range(start, end):
                path = os.path.join(self.root, driver, f"img_{index:05d}.jpg")
                if not os.path.exists(path):
                    raise FileNotFoundError(f"image {path} is not found in the disk.")

        # we are going to filter some frames here
        print('before: ', len(self.data_pairs))
        for f in filters:
            self.data_pairs = [data_pair for data_pair in self.data_pairs if f(data_pair[-1])]
        print('after: ', len(self.data_pairs))
        self.data_pairs = list(self.data_pairs)

    # from ngq
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

    def _parse_annotation(self):
        with open(self.anno_path, 'r') as f:
            self.annotations = json.load(f)

        drivers = sorted(self.annotations.keys())
        # devide into training set and validation set
        # subset = 'training' if self.train else 'testing'
        # drivers = [
        #     driver for driver in drivers if self.annotations[driver]['subset'] == subset]
        bus_id = [1, 3, 4] if self.train else [2]
        drivers = [driver for driver in drivers if self.annotations[driver]['busID'] in bus_id]

        pose_json = {}
        json_path = ''
        view_file = self.anno_path
        interval = (self.patch_size - 1) / 2
        box_coordinate = (0,0,0,0) # (x1,y1,x2,y2)
        for driver in tqdm(drivers): # e.g. b1-rgb/33-b1-d1-3
            #print(driver)
            if args.debug:
                if self.segment_count >200:
                    break

            if 'rgb' not in driver:
                continue
            if 'b1' in driver:
                box_coordinate = (220, 150, 520, 450)
            elif 'b2' in driver:
                box_coordinate = (330, 130, 570, 370)
            elif 'b3' in driver:
                box_coordinate = (320, 120, 600, 400)
            elif 'b4' in driver:
                box_coordinate = (300, 100, 580, 380)
            else:
                raise NotImplementedError('{} is wrong'.format(driver))
            fps = round(self.annotations[driver]['fps'])
            if fps == 0 or fps is None:
                raise ValueError(f"fps should not be {fps}")
            lighting = self.annotations[driver]['lighting']
            type_ = self.annotations[driver]['type']
            for item in self.annotations[driver]['annotations']:
                start_pts, end_pts = item['segment']
                quality = item['quality']
                # Note that pts=16.34 indicates the 34-th frame after 16-th second
                start_index = _pts2index_impl(start_pts, fps)
                end_index = _pts2index_impl(end_pts, fps)

                # first try
                # some process due to the mistakes in annotation file
                if start_index > end_index:
                    start_index = end_index - (start_index - end_index)
                total_ims = len((os.listdir('{}/{}'.format(self.root, driver))))
                if end_index+1 > total_ims:
                    gap = np.abs(end_index - total_ims)
                    if start_index > gap:
                        start_index -= gap
                    end_index -= gap
                # second try
                # if start_index >= end_index or end_index-start_index < self.n_frames:
                #     continue

                assert  start_index <= end_index, 'fps={}, start={}, end={}, this {}/{}-{} wrong'.format(fps, start_index, end_index, driver, start_pts, end_pts)


                label = item['label']
                # we only rserver the label anno. in 'BusDeriverDataset.anno2index.keys'
                # this is also a kind of filter
                if label not in BusDeriverDataset3D.anno2index.keys():
                    continue
                label = BusDeriverDataset3D.anno2index[label]
                view = str(driver)
                vid = view
                if not pose_json or vid not in json_path:
                    json_path = os.path.join(self.video_pose_path, vid,
                                             'video.json')  # put in if statement, so that each video only read json one time
                    if os.path.exists(json_path):
                        f = open(json_path, 'r')
                        pose_json = json.load(f)
                        f.close()
                    else:
                        f = None
                skeleton = []
                boxes_list = []
                if f is not None:
                    for frame_idx in range(start_index+1, end_index+1):
                        frame_skeleton = []
                        boxes = []
                        if pose_json[frame_idx - 1]['img_name'] != 'img_' + str(frame_idx).zfill(5):
                            print('image index error')
                        people = pose_json[frame_idx - 1]['keypoints']['people']
                        # print(people)
                        if people:  # not None means there are extracted keypoints
                            nums_in_box = [0 for x in range(len(people))]
                            for person_id in range(len(people)):
                                pose_keypoints_2d = people[person_id]['pose_keypoints_2d']
                                for i in self.keypoint_index:  # get predefined 13 points
                                    i = i * 3
                                    # y,x,bug
                                    x = pose_keypoints_2d[i] * 854
                                    y = pose_keypoints_2d[i + 1] * 480
                                    if box_coordinate[0] <= x <= box_coordinate[2] and box_coordinate[1] <= y <= box_coordinate[3]:
                                        nums_in_box[person_id] += 1
                            max_index = nums_in_box.index(max(nums_in_box))
                            pose_keypoints_2d = people[max_index]['pose_keypoints_2d']  # assumed driver
                            j = 0
                            for i in self.keypoint_index:  # get predefined 13 points
                                i = i * 3
                                x = pose_keypoints_2d[i]
                                y = pose_keypoints_2d[i + 1]
                                # roi
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
                            if not args.one_person:
                                if label in [6, 7]:
                                    c_max = 0
                                    people_max = 0
                                    for person_id in range(len(people)):
                                        if person_id == max_index:
                                            continue
                                        other_pose_keypoints_2d = people[person_id]['pose_keypoints_2d']
                                        c_sum = 0
                                        for i in self.keypoint_index:  # get predefined 13 points
                                            i = i * 3
                                            # y,x,bug
                                            c_sum += other_pose_keypoints_2d[i+2]
                                        if c_sum > c_max:
                                            c_max = c_sum
                                            people_max = person_id # get interploated person
                                    interpolated_person = people[people_max]['pose_keypoints_2d']
                                    for i in self.keypoint_index:  # change c of 13 points to 0
                                        i = i * 3
                                        interpolated_person[i+2] = 0
                                    pose_keypoints_2d = list(np.array(pose_keypoints_2d) - np.array(interpolated_person))

                            j = 0
                            for i in self.keypoint_index:  # get predefined 13 points
                                i = i * 3
                                # y,x,bug
                                x = pose_keypoints_2d[i]
                                y = pose_keypoints_2d[i + 1]
                                c = pose_keypoints_2d[i + 2]
                                # frame_skeleton.append(float(x))
                                # frame_skeleton.append(float(y))
                                # roi
                                # if args.keypoint_mean:
                                #     if x == 0 or y == 0:
                                #         x = self.x_mean[j]
                                #         y = self.y_mean[j]
                                #     j += 1
                                # print(x,y)
                                frame_skeleton.append(float(x))
                                frame_skeleton.append(float(y))
                                if args.xyc:
                                    frame_skeleton.append(c)
                        #                                 if x == 0 or y == 0:
                        #                                     x = 0.5
                        #                                     y = 0.5
                        #                                 x = round(x * args.spatial_size)
                        #                                 y = round(y * args.spatial_size)
                        #                                 x1 = x - interval
                        #                                 y1 = y - interval
                        #                                 x2 = x + interval
                        #                                 y2 = y + interval
                        #                                 box = [x1 - 0.5, y1 - 0.5, x2 + 0.5, y2 + 0.5]
                        #                                 boxes.append(box)

                        else:  # if no keypoint, just append 0
                            for i in range(len(self.keypoint_index)):
                                x = 0
                                y = 0
                                c = 0
                                # frame_skeleton.append(float(x))
                                # frame_skeleton.append(float(y))
                                # if args.keypoint_mean:
                                #     x = self.x_mean[i]
                                #     y = self.y_mean[i]
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
                        skeleton.append(frame_skeleton)  # skeleton shape: num_frames*26
                        boxes_list.append(boxes)
                # # print(start_index, end_index)
                self.data_pairs.append(
                    (view, driver, start_index+1, end_index, label, skeleton, boxes_list, dict(lighting=lighting, type=type_, fps=fps,
                                                    start_pts=start_pts, end_pts=end_pts,
                                                    start_index=start_index, end_index=end_index,
                                                    quality=quality, label=label, segment_count=self.segment_count)))

                self.segment_count += 1
                # if args.debug:
                #     if self.segment_count == 1:
                #         break

    def __getitem__(self, index):
        view, driver, start, end, label, ske, boxes_list, metadata = self.data_pairs[index]

        boxes = np.array(boxes_list).astype(np.float)
        boxes = torch.tensor(boxes)
        ske = np.array(ske).astype(np.float32)
        if args.xyc:
            ske = torch.tensor(ske.reshape((end - start + 1, 13, 3)))
        else:
            ske = torch.tensor(ske.reshape((end - start + 1, 13, 2)))
        if ske.shape[0] < self.num_segments:
            ske = list(ske)
            ske = torch.cat([i.unsqueeze(0) for i in ske] + (self.num_segments - len(ske)) * [ske[-1].unsqueeze(0)],dim=0)
        # correct sampling method
        if self.train:
            segment_indices, labels = self._sample_indices(start, end, label, self.num_segments)
        else:
            segment_indices, labels = self._get_val_indices(start, end, label, self.num_segments)

        segment_indices.sort()

        if self.num_segments == self.gcn_segments:
            ske = ske[segment_indices, :, :]  # till this line, get the needed skeleton point for training
            boxes = boxes[segment_indices, :]
        else:
            if self.train:
                gcn_segment_indices, labels = self._sample_indices(start, end, label, self.gcn_segments)

            else:
                gcn_segment_indices, labels = self._get_val_indices(start, end, label, self.gcn_segments)

            gcn_segment_indices.sort()
            ske = ske[gcn_segment_indices, :, :]
            boxes = boxes[gcn_segment_indices, :]
        # add a function creating heatmap based on ske
        heatmap = torch.ones(0)

        images = list()
        for seg_ind in segment_indices:
            p = int(seg_ind) + start
            for i in range(self.new_length):
                imgs = load_rgb_frames(self.root, driver, p)
                images.extend(imgs)
        process_data = self.transforms(images) #torch.Size([3*num_frames, 224, 224])
        # one of b1/b2/b3/b4

        # not have skeleton
        #         heatmap = torch.ones([8, 13, 7, 7])
        #         ske = torch.ones([152, 13, 2])
        #         boxes = torch.ones([8, 13, 4])

        return process_data, torch.from_numpy(np.array(label)), ske.permute(2, 0, 1).unsqueeze(3), boxes


    def __len__(self):
        return len(self.data_pairs)

if __name__ == '__main__':
    test_dataset = BusDeriverDataset3D(
        root=args.root,
        anno_path='annotation(2)(1).json',
        train=True,
        filters=[is_high_quality],
        transforms=None,
        n_frames=args.num_segments, gcn_segments=8,
        interval=0
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True
    )