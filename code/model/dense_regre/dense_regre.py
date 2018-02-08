import os
import numpy as np
import tensorflow as tf
import progressbar
import h5py
import matplotlib.pyplot as mpplot
from cv2 import resize as cv2resize
from args_holder import args_holder
from model.base_regre import base_regre
from utils.iso_boxes import iso_cube


class dense_regre(base_regre):
    class batch_allot(object):
        def __init__(self, model_inst):
            self.batch_size = model_inst.batch_size
            self.crop_size = model_inst.crop_size
            self.hmap_size = model_inst.hmap_size
            self.out_dim = model_inst.out_dim
            self.num_channel = model_inst.num_channel
            self.num_appen = model_inst.num_appen
            batch_data = {
                'batch_index': np.empty(
                    shape=(self.batch_size, 1), dtype=np.int32),
                'batch_frame': np.empty(
                    shape=(
                        self.batch_size,
                        self.crop_size, self.crop_size,
                        self.num_channel),
                    # dtype=np.float32),
                    dtype=float),
                'batch_hmap2': np.empty(
                    shape=(
                        self.batch_size,
                        self.hmap_size, self.hmap_size,
                        self.out_dim),
                    # dtype=np.float32)
                    dtype=float),
                'batch_hmap3': np.empty(
                    shape=(
                        self.batch_size,
                        self.hmap_size, self.hmap_size,
                        self.out_dim),
                    # dtype=np.float32)
                    dtype=float),
                'batch_uomap': np.empty(
                    shape=(
                        self.batch_size,
                        self.hmap_size, self.hmap_size,
                        self.out_dim * 3),
                    # dtype=np.float32)
                    dtype=float),
                'batch_resce': np.empty(
                    shape=(self.batch_size, self.num_appen),
                    # dtype=np.float32),
                    dtype=float),
            }
            self.batch_bytes = \
                batch_data['batch_index'].nbytes + batch_data['batch_frame'].nbytes + \
                batch_data['batch_hmap2'].nbytes + batch_data['batch_hmap3'].nbytes + \
                batch_data['batch_uomap'].nbytes + batch_data['batch_resce'].nbytes
            self.batch_beg = 0

        def allot(self, store_size=-1):
            from psutil import virtual_memory
            store_cap_mult = (virtual_memory().total >> 2) // self.batch_bytes
            store_cap = store_cap_mult * self.batch_size
            if 0 > store_size:
                self.store_size = store_cap
            else:
                self.store_size = min(store_cap, store_size)
            self.store_bytes = self.store_size * self.batch_bytes / self.batch_size
            self.store_beg = 0
            self.batch_index = np.empty(
                shape=(self.store_size, 1), dtype=np.int32)
            self.batch_frame = np.empty(
                shape=(
                    self.store_size,
                    self.crop_size, self.crop_size,
                    self.num_channel),
                # dtype=np.float32)
                dtype=float)
            self.batch_hmap2 = np.empty(
                shape=(store_size, self.hmap_size, self.hmap_size, self.out_dim),
                # dtype=np.float32)
                dtype=float)
            self.batch_hmap3 = np.empty(
                shape=(store_size, self.hmap_size, self.hmap_size, self.out_dim),
                # dtype=np.float32)
                dtype=float)
            self.batch_uomap = np.empty(
                shape=(store_size, self.hmap_size, self.hmap_size, self.out_dim * 3),
                # dtype=np.float32)
                dtype=float)
            self.batch_resce = np.empty(
                shape=(self.store_size, self.num_appen),
                # dtype=np.float32)
                dtype=float)

    @staticmethod
    def get_trainer(args, new_log):
        from train import train_dense_regre
        return train_dense_regre(args, new_log)

    def __init__(self, args):
        super(dense_regre, self).__init__(args)
        self.batch_allot = dense_regre.batch_allot
        self.num_appen = 4
        self.hmap_size = 32

    def provider_worker(self, line, image_dir, caminfo):
        img_name, pose_raw = self.data_module.io.parse_line_annot(line)
        img = self.data_module.io.read_image(os.path.join(image_dir, img_name))
        img_crop_resize, resce = self.data_module.ops.crop_resize_pca(
            img, pose_raw, caminfo)
        resce3 = resce[0:4]
        cube = iso_cube()
        cube.load(resce3)
        hmap2 = self.data_module.ops.raw_to_heatmap2(
            pose_raw, cube, self.hmap_size, caminfo
        )
        _, hmap3, uomap = self.data_module.ops.raw_to_offset(
            img, pose_raw, cube, self.hmap_size, caminfo
        )
        index = self.data_module.io.imagename2index(img_name)
        return (index, np.expand_dims(img_crop_resize, axis=-1),
                hmap2, hmap3, uomap, resce)

    @staticmethod
    def put_worker(
        args, image_dir, model_inst,
            caminfo, data_module, batchallot):
        bi = args[0]
        line = args[1]
        index, frame, hmap2, hmap3, uomap, resce = model_inst.provider_worker(
            line, image_dir, caminfo)
        batchallot.batch_index[bi, :] = index
        batchallot.batch_frame[bi, ...] = frame
        # batchallot.batch_poses[bi, :] = poses
        batchallot.batch_hmap2[bi, :] = hmap2
        batchallot.batch_hmap3[bi, :] = hmap3
        batchallot.batch_uomap[bi, :] = uomap
        batchallot.batch_resce[bi, :] = resce

    def fetch_batch(self, fetch_size=None):
        if fetch_size is None:
            fetch_size = self.batch_size
        batch_end = self.batch_beg + fetch_size
        if batch_end >= self.store_size:
            self.batch_beg = batch_end
            batch_end = self.batch_beg + fetch_size
            self.split_end -= self.store_size
        # print(self.batch_beg, batch_end, self.split_end)
        if batch_end >= self.split_end:
            return None
        self.batch_data = {
            'batch_index': self.store_file['index'][self.batch_beg:batch_end, ...],
            'batch_frame': self.store_file['frame'][self.batch_beg:batch_end, ...],
            'batch_hmap2': self.store_file['hmap2'][self.batch_beg:batch_end, ...],
            'batch_hmap3': self.store_file['hmap3'][self.batch_beg:batch_end, ...],
            'batch_uomap': self.store_file['uomap'][self.batch_beg:batch_end, ...],
            'batch_resce': self.store_file['resce'][self.batch_beg:batch_end, ...]
        }
        self.batch_beg = batch_end
        return self.batch_data

    def prepare_data(self, thedata, args,
                     batchallot, file_annot, name_appen):
        num_line = int(sum(1 for line in file_annot))
        file_annot.seek(0)
        batchallot.allot(num_line)
        store_size = batchallot.store_size
        num_stores = int(np.ceil(float(num_line) / store_size))
        self.logger.debug(
            'preparing data [{}]: {:d} lines (producing {:.4f} GB for store size {:d}) ...'.format(
                self.__class__.__name__, num_line,
                float(batchallot.store_bytes) / (2 << 30),
                store_size))
        timerbar = progressbar.ProgressBar(
            maxval=num_stores,
            widgets=[
                progressbar.Percentage(),
                ' ', progressbar.Bar('=', '[', ']'),
                ' ', progressbar.ETA()]
        ).start()
        crop_size = self.crop_size
        hmap_size = self.hmap_size
        out_dim = self.out_dim
        num_channel = self.num_channel
        num_appen = self.num_appen
        with h5py.File(os.path.join(self.prepare_dir, name_appen), 'w') as h5file:
            h5file.create_dataset(
                'index',
                (num_line, 1),
                compression='lzf',
                dtype=np.int32
            )
            h5file.create_dataset(
                'frame',
                (num_line,
                    crop_size, crop_size,
                    num_channel),
                chunks=(1,
                        crop_size, crop_size,
                        num_channel),
                compression='lzf',
                # dtype=np.float32)
                dtype=float)
            h5file.create_dataset(
                'hmap2',
                (num_line, hmap_size, hmap_size, out_dim),
                compression='lzf',
                # dtype=np.float32)
                dtype=float)
            h5file.create_dataset(
                'hmap3',
                (num_line, hmap_size, hmap_size, out_dim),
                compression='lzf',
                # dtype=np.float32)
                dtype=float)
            h5file.create_dataset(
                'uomap',
                (num_line, hmap_size, hmap_size, out_dim * 3),
                compression='lzf',
                # dtype=np.float32)
                dtype=float)
            h5file.create_dataset(
                'resce',
                (num_line, num_appen),
                compression='lzf',
                # dtype=np.float32)
                dtype=float)
            bi = 0
            store_beg = 0
            while True:
                resline = self.data_module.provider.puttensor_mt(
                    file_annot, self.put_worker, self.image_dir,
                    self, thedata, self.data_module, batchallot
                )
                if 0 > resline:
                    break
                h5file['index'][store_beg:store_beg + resline, ...] = \
                    batchallot.batch_index[0:resline, ...]
                h5file['frame'][store_beg:store_beg + resline, ...] = \
                    batchallot.batch_frame[0:resline, ...]
                # h5file['poses'][store_beg:store_beg + resline, ...] = \
                #     batchallot.batch_poses[0:resline, ...]
                h5file['hmap2'][store_beg:store_beg + resline, ...] = \
                    batchallot.batch_hmap2[0:resline, ...]
                h5file['hmap3'][store_beg:store_beg + resline, ...] = \
                    batchallot.batch_hmap3[0:resline, ...]
                h5file['uomap'][store_beg:store_beg + resline, ...] = \
                    batchallot.batch_uomap[0:resline, ...]
                h5file['resce'][store_beg:store_beg + resline, ...] = \
                    batchallot.batch_resce[0:resline, ...]
                timerbar.update(bi)
                bi += 1
                store_beg += resline
        timerbar.finish()

    def draw_random(self, thedata, args):
        with h5py.File(self.appen_train, 'r') as h5file:
            store_size = h5file['index'].shape[0]
            frame_id = np.random.choice(store_size)
            # frame_id = 0
            img_id = h5file['index'][frame_id, 0]
            frame_h5 = np.squeeze(h5file['frame'][frame_id, ...], -1)
            poses_h5 = h5file['poses'][frame_id, ...].reshape(-1, 3)
            resce_h5 = h5file['resce'][frame_id, ...]

        print('[{}] drawing image #{:d} ...'.format(self.name_desc, img_id))
        print(np.min(frame_h5), np.max(frame_h5))
        print(np.histogram(frame_h5, range=(1e-4, np.max(frame_h5))))
        print(np.min(poses_h5, axis=0), np.max(poses_h5, axis=0))
        print(resce_h5)
        resce3 = resce_h5[0:4]
        resce2 = resce_h5[4:7]
        mpplot.subplots(nrows=2, ncols=2, figsize=(2 * 5, 2 * 5))

        mpplot.subplot(2, 2, 3)
        mpplot.gca().set_title('test storage read')
        # resize the cropped resion for eazier pose drawing in the commen frame
        sizel = np.floor(resce2[0]).astype(int)
        resce_cp = np.copy(resce2)
        resce_cp[0] = 1
        mpplot.imshow(
            cv2resize(frame_h5, (sizel, sizel)),
            cmap='bone')
        pose_raw = args.data_ops.local_to_raw(poses_h5, resce3)
        args.data_draw.draw_pose2d(
            thedata,
            args.data_ops.raw_to_2d(pose_raw, thedata, resce_cp)
        )

        mpplot.subplot(2, 2, 4)
        mpplot.gca().set_title('test output')
        img_name = args.data_io.index2imagename(img_id)
        img = args.data_io.read_image(os.path.join(self.image_dir, img_name))
        mpplot.imshow(img, cmap='bone')
        pose_raw = self.yanker(
            poses_h5, resce_h5, self.caminfo)
        args.data_draw.draw_pose2d(
            thedata,
            args.data_ops.raw_to_2d(pose_raw, thedata)
        )
        rect = iso_rect()
        rect.load(resce2)
        rect.draw()

        mpplot.subplot(2, 2, 1)
        mpplot.gca().set_title('test input #{:d}'.format(img_id))
        annot_line = args.data_io.get_line(
            thedata.training_annot_cleaned, img_id)
        img_name, pose_raw = args.data_io.parse_line_annot(annot_line)
        img = args.data_io.read_image(os.path.join(self.image_dir, img_name))
        mpplot.imshow(img, cmap='bone')
        args.data_draw.draw_pose2d(
            thedata,
            args.data_ops.raw_to_2d(pose_raw, thedata))

        mpplot.subplot(2, 2, 2)
        mpplot.gca().set_title('test storage write')
        img_name, frame, poses, resce = self.provider_worker(
            annot_line, self.image_dir, thedata)
        frame = np.squeeze(frame, axis=-1)
        poses = poses.reshape(-1, 3)
        if (
                (1e-4 < np.linalg.norm(frame_h5 - frame)) or
                (1e-4 < np.linalg.norm(poses_h5 - poses))
        ):
            print(np.linalg.norm(frame_h5 - frame))
            print(np.linalg.norm(poses_h5 - poses))
            print('ERROR - h5 storage corrupted!')
        resce3 = resce[0:4]
        resce2 = resce[4:7]
        sizel = np.floor(resce2[0]).astype(int)
        resce_cp = np.copy(resce2)
        resce_cp[0] = 1
        mpplot.imshow(
            cv2resize(frame, (sizel, sizel)),
            cmap='bone')
        pose_raw = args.data_ops.local_to_raw(poses, resce3)
        args.data_draw.draw_pose2d(
            thedata,
            args.data_ops.raw_to_2d(pose_raw, thedata, resce_cp)
        )

        mpplot.tight_layout()
        mpplot.savefig(os.path.join(
            args.predict_dir,
            'draw_{}.png'.format(self.name_desc)))
        mpplot.show()
        print('[{}] drawing image #{:d} - done.'.format(
            self.name_desc, img_id))


if __name__ == "__main__":
    # python dense_regre.py --max_epoch=1 --batch_size=16 --model_name=base_regre
    with args_holder() as argsholder:
        argsholder.parse_args()
        argsholder.create_instance()
        regresor = dense_regre(argsholder.args)
        regresor.train()
        regresor.evaluate()
