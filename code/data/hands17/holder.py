import os
import numpy as np
from colour import Color
import cv2
import random
from multiprocessing.dummy import Pool as ThreadPool
# import multiprocessing
# from multiprocessing import Manager as ThreadManager
import ops as dataops
import io as dataio


class hands17holder:
    """ Pose class for Hands17 dataset """

    # dataset info
    data_dir = ''
    out_dir = ''
    training_images = ''
    frame_images = ''
    training_cropped = ''
    training_annot_origin = ''
    training_annot_cleaned = ''
    training_annot_train = ''
    training_annot_test = ''
    training_annot_predict = ''
    frame_bbox = ''

    # num_training = int(957032)
    num_training = int(992)
    # num_training = int(96)
    tt_split = int(64)
    range_train = np.zeros(2, dtype=np.int)
    range_test = np.zeros(2, dtype=np.int)

    # cropped & resized training images
    image_size = [640, 480]
    crop_size = 96
    min_z = 1
    max_z = 3333
    max_distance = 9999  # max distance set to 10m
    # camera info
    focal = (475.065948, 475.065857)
    centre = (315.944855, 245.287079)
    # fx = 475.065948
    # fy = 475.065857
    # cx = 315.944855
    # cy = 245.287079

    # joints description
    join_name = [
        'Wrist',
        'TMCP', 'IMCP', 'MMCP', 'RMCP', 'PMCP',
        'TPIP', 'TDIP', 'TTIP',
        'IPIP', 'IDIP', 'ITIP',
        'MPIP', 'MDIP', 'MTIP',
        'RPIP', 'RDIP', 'RTIP',
        'PPIP', 'PDIP', 'PTIP'
    ]

    join_num = 21
    join_type = ('W', 'T', 'I', 'M', 'R', 'P')
    join_color = (
        # Color('cyan'),
        Color('black'),
        Color('magenta'),
        Color('blue'),
        Color('lime'),
        Color('yellow'),
        Color('red')
    )
    join_id = (
        (1, 6, 7, 8),
        (2, 9, 10, 11),
        (3, 12, 13, 14),
        (4, 15, 16, 17),
        (5, 18, 19, 20)
    )
    bone_id = (
        ((0, 1), (1, 6), (6, 11), (11, 16)),
        ((0, 2), (2, 7), (7, 12), (12, 17)),
        ((0, 3), (3, 8), (8, 13), (13, 18)),
        ((0, 4), (4, 9), (9, 14), (14, 19)),
        ((0, 5), (5, 10), (10, 15), (15, 20))
    )
    bbox_color = Color('orange')

    def remove_out_frame_annot(self):
        self.num_training = int(0)
        with open(self.training_annot_cleaned, 'w') as writer, \
                open(self.training_annot_origin, 'r') as reader:
            for annot_line in reader.readlines():
                _, pose_raw = dataio.parse_line_annot(annot_line)
                pose2d = dataops.raw_to_2d(pose_raw, self.centre, self.focal)
                if 0 > np.min(pose2d):
                    continue
                if 0 > np.min(self.image_size - pose2d):
                    continue
                writer.write(annot_line)
                self.num_training += 1

    def shuffle_split(self):
        with open(self.training_annot_cleaned, 'r') as source:
            lines = source.readlines()
        random.shuffle(lines)
        with open(self.training_annot_train, 'w') as f:
            for line in lines[self.range_train[0]:self.range_train[1]]:
                f.write(line)
        with open(self.training_annot_test, 'w') as f:
            for line in lines[self.range_test[0]:self.range_test[1]]:
                # name = re.search(r'(image_D\d+\.png)', line).group(1)
                # shutil.move(
                #     os.path.join(self.training_cropped, name),
                #     os.path.join(self.evaluate_cropped, name))
                f.write(line)

    def get_rect_crop_resize(self, annot_line):
        """
            Returns:
                p3z_crop: projected 2d coordinates, and original z on the 3rd column
        """
        img_name, pose_raw, rescen = dataio.parse_line_annot(annot_line)
        img = dataio.read_image(
            os.path.join(self.training_images, img_name))
        pose2d = dataops.raw_to_2d(
            pose_raw, self.centre, self.focal)
        rect = dataops.get_rect3(
            pose_raw,
            self.centre, self.focal,
            self.image_size, 0.25)
        crop_size = self.crop_size
        rs = crop_size / rect[1, 1]
        rescen = np.append(rs, rect[0, :])
        p2d_crop = (pose2d - rect[0, :]) * rs
        p3z_crop = np.hstack((
            p2d_crop, np.array(pose_raw[:, 2].reshape(-1, 1)) * rs
        ))
        img_crop = img[
            int(np.floor(rect[0, 1])):int(np.ceil(rect[0, 1] + rect[1, 1])),
            int(np.floor(rect[0, 0])):int(np.ceil(rect[0, 0] + rect[1, 0]))
        ]
        # try:
        # img_crop_resize = spmisc.imresize(
        #     img_crop, (crop_size, crop_size), interp='bilinear')
        # img_crop_resize = spndim.interpolation.zoom(img_crop, rs)
        # img_crop_resize = img_crop_resize[0:crop_size, 0:crop_size]
        img_crop_resize = cv2.resize(
            img_crop, (crop_size, crop_size))
        # except:
        #     print(np.hstack((pose_raw, pose2d)))
        # print(np.max(img_crop), np.max(img_crop_resize), img_crop_resize.shape)

        return img_name, img_crop_resize, p3z_crop, rescen

    def crop_resize_save(self, annot_line, messages=None):
        img_name, img_crop, p3z_crop, rescen = self.get_rect_crop_resize(
            annot_line)
        img_crop[self.min_z > img_crop] = self.max_distance
        img_crop[self.max_z < img_crop] = self.max_distance
        dataio.save_image(
            os.path.join(self.training_cropped, img_name),
            img_crop
        )
        # self.draw_hist_random(self.training_cropped, img_name)
        out_list = np.append(p3z_crop.flatten(), rescen.flatten()).flatten()
        crimg_line = ''.join("%12.4f" % x for x in out_list)
        pose_l = img_name + crimg_line + '\n'
        if messages is not None:
            messages.put(pose_l)
        return pose_l

    def crop_resize_training_images(self):
        if not os.path.exists(self.training_cropped):
            os.makedirs(self.training_cropped)
        with open(self.training_annot_cropped, 'w') as crop_writer:
            with open(self.training_annot_shuffled, 'r') as fanno:
                for line_number, annot_line in enumerate(fanno):
                    # opts = {
                    #     'class_inst': self,
                    #     'annot_line': annot_line
                    # }
                    pose_l = self.crop_resize_save(annot_line)
                    crop_writer.write(pose_l)
                    # break

    class mt_queue_writer:
        @staticmethod
        def listener(file_name, messages):
            with open(file_name, 'w') as writer:
                while 1:
                    m = messages.get()
                    if m == '\n':
                        break
                    print(m)
                    writer.write(m)

    def crop_resize_training_images_mt(self):
        if not os.path.exists(self.training_cropped):
            os.makedirs(self.training_cropped)
        # thread_manager = ThreadManager()
        # messages = thread_manager.Queue()
        # thread_pool = ThreadPool(multiprocessing.cpu_count() + 2)
        # watcher = thread_pool.apply_async(
        #     mt_queue_writer.listener,
        #     (self.training_annot_cropped, messages))
        # jobs = []
        # with open(self.training_annot_shuffled, 'r') as fanno:
        #     annot_line = fanno.readline()
        #     job = thread_pool.apply_async(
        #         hands17holder.crop_resize_save, (annot_line, opts.crop_size, messages))
        #     jobs.append(job)
        # for job in jobs:
        #     job.get()
        # messages.put('\n')
        # thread_pool.close()
        thread_pool = ThreadPool()
        with open(self.training_annot_shuffled, 'r') as fanno:
            annot_list = [line for line in fanno if line]
        with open(self.training_annot_cropped, 'w') as writer:
            for result in thread_pool.map(self.crop_resize_save, annot_list):
                writer.write(result)
        thread_pool.close()
        thread_pool.join()

    def init_data(self, rebuild=False):
        if rebuild or (not os.path.exists(self.training_annot_cleaned)):
            self.remove_out_frame_annot()
            print('data cleaned, using: {}'.format(
                self.training_annot_cleaned))
        else:
            self.num_training = int(sum(
                1 for line in open(self.training_annot_cleaned, 'r')))
        print('total number of images: {:d}'.format(
            self.num_training))

        portion = int(self.num_training / self.tt_split)
        self.range_train[0] = int(0)
        self.range_train[1] = int(portion * (self.tt_split - 1))
        self.range_test[0] = self.range_train[1]
        self.range_test[1] = self.num_training
        print('splitted data: {} training, {} test.'.format(
            self.range_train, self.range_test))

        if (rebuild or (not os.path.exists(self.training_annot_train)) or
                (not os.path.exists(self.training_annot_test))):
            self.shuffle_split()
        print('images are splitted out for evaluation: {:d} portions'.format(
            self.tt_split))

        # # if rebuild:  # just over-write, this detete operation is slow
        # #     if os.path.exists(self.training_cropped):
        # #         shutil.rmtree(self.training_cropped)
        # if (rebuild or (not os.path.exists(self.training_annot_cropped)) or
        #         (not os.path.exists(self.training_cropped))):
        #     print('running cropping code (be patient) ...')
        #     # time_s = timer()
        #     # self.crop_resize_training_images()
        #     # print('single tread time: {:.4f}'.format(timer() - time_s))
        #     time_s = timer()
        #     self.crop_resize_training_images_mt()
        #     print('multiprocessing time: {:.4f}'.format(timer() - time_s))
        # print('using cropped and resized images: {}'.format(
        #     self.training_cropped))

    def __init__(self, args):
        self.data_dir = args.data_dir
        self.out_dir = args.out_dir
        self.crop_size = args.crop_size

        self.training_images = os.path.join(self.data_dir, 'training/images')
        self.frame_images = os.path.join(self.data_dir, 'frame/images')
        self.training_cropped = os.path.join(self.out_dir, 'cropped')
        self.training_annot_origin = os.path.join(
            self.data_dir, 'training/Training_Annotation.txt')
        self.training_annot_cleaned = os.path.join(
            self.out_dir, 'annotation.txt')
        self.training_annot_train = os.path.join(
            self.out_dir, 'training_train.txt')
        self.training_annot_test = os.path.join(
            self.out_dir, 'training_test.txt')
        self.training_annot_predict = os.path.join(
            self.out_dir, 'training_predict.txt')
        self.frame_bbox = os.path.join(self.data_dir, 'frame/BoundingBox.txt')
