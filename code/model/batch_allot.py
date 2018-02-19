import numpy as np
from psutil import virtual_memory


class batch_index(object):
    def __init__(self, data_inst, store_size=-1):
        import multiprocessing
        self.store_size = multiprocessing.cpu_count() * (2 << 4)
        if 0 < store_size:
            self.store_size = min(self.store_size, store_size)
        self.data_inst = data_inst
        self.create_fn = {
            'index': self.create_index,
        }

    def create_index(self, filepack, h5file_name, num_line):
        join_num = self.data_inst.join_num
        h5file = filepack.write_h5(h5file_name)
        h5file.create_dataset(
            'index',
            (num_line,),
            maxshape=(num_line,),
            compression='lzf',
            dtype='i4'
        )
        h5file.create_dataset(
            'poses',
            (num_line, join_num, 3),
            maxshape=(num_line, join_num, 3),
            compression='lzf',
            dtype='f4')
        h5file.create_dataset(
            'resce',
            (num_line, 4),
            maxshape=(num_line, 4),
            compression='lzf',
            dtype='f4')
        batch_data = {
            'valid': np.full(
                (self.store_size,),
                True),
            'index': np.empty(
                shape=(self.store_size,),
                dtype='f4'),
            'poses': np.empty(
                shape=(self.store_size, join_num, 3),
                dtype='f4'),
            'resce': np.empty(
                shape=(self.store_size, 4),
                dtype='f4')
        }
        return h5file, batch_data

    def resize(self, h5file, num_line):
        join_num = self.data_inst.join_num
        h5file['index'].resize((num_line,))
        h5file['poses'].resize((num_line, join_num, 3))
        h5file['resce'].resize((num_line, 4))


class batch_crop2(object):
    def __init__(self, model_inst, store_size=-1):
        import multiprocessing
        self.store_size = multiprocessing.cpu_count() * (2 << 4)
        if 0 < store_size:
            self.store_size = min(self.store_size, store_size)
        self.model_inst = model_inst
        # self.crop_size = model_inst.crop_size
        # out_dim = model_inst.join_num
        # self.entry = {
        #     'crop2': np.empty(
        #         shape=(
        #             self.store_size,
        #             self.crop_size, self.crop_size),
        #         dtype='f4'),
        #     'pose_c': np.empty(
        #         shape=(self.store_size, out_dim),
        #         dtype='f4'),
        # }
        # self.store_bytes = 0
        # for _, v in self.entry.items():
        #     self.store_bytes += v.nbytes
        self.create_fn = {
            'crop2': self.create_crop2,
            'pose_c': self.create_pose_c,
        }

    def create_crop2(self, filepack, h5file_name, num_line):
        crop_size = self.model_inst.crop_size
        h5file = filepack.write_h5(h5file_name)
        h5file.create_dataset(
            'crop2',
            (num_line,
                crop_size, crop_size),
            chunks=(1,
                    crop_size, crop_size),
            compression='lzf',
            dtype='f4')
        batch_data = np.empty(
            shape=(
                self.store_size,
                crop_size, crop_size),
            dtype='f4')
        return h5file['crop2'], batch_data

    def create_pose_c(self, filepack, h5file_name, num_line):
        out_dim = self.model_inst.join_num * 3
        h5file = filepack.write_h5(h5file_name)
        h5file.create_dataset(
            'pose_c',
            (num_line, out_dim),
            compression='lzf',
            dtype='f4')
        batch_data = np.empty(
            shape=(self.store_size, out_dim),
            dtype='f4')
        return h5file['pose_c'], batch_data


class batch_clean(batch_crop2):
    def __init__(self, model_inst, store_size=-1):
        super(batch_clean, self).__init__(model_inst, store_size)
        self.create_fn['clean'] = self.create_clean

    def create_clean(self, filepack, h5file_name, num_line):
        crop_size = self.model_inst.crop_size
        h5file = filepack.write_h5(h5file_name)
        h5file.create_dataset(
            'clean',
            (num_line,
                crop_size, crop_size),
            chunks=(1,
                    crop_size, crop_size),
            compression='lzf',
            dtype='f4')
        batch_data = np.empty(
            shape=(
                self.store_size,
                crop_size, crop_size),
            dtype='f4')
        return h5file['clean'], batch_data


class batch_ortho3(batch_clean):
    def __init__(self, model_inst, store_size=-1):
        super(batch_ortho3, self).__init__(model_inst, store_size)
        self.create_fn['ortho3'] = self.create_ortho3

    def create_ortho3(self, filepack, h5file_name, num_line):
        crop_size = self.model_inst.crop_size
        h5file = filepack.write_h5(h5file_name)
        h5file.create_dataset(
            'ortho3',
            (num_line,
                crop_size, crop_size,
                3),
            chunks=(1,
                    crop_size, crop_size,
                    3),
            compression='lzf',
            dtype='f4')
        batch_data = np.empty(
            shape=(
                self.store_size,
                crop_size, crop_size,
                3),
            dtype='f4')
        return h5file['ortho3'], batch_data


class batch_conv3(batch_clean):
    def __init__(self, model_inst, store_size=-1):
        super(batch_conv3, self).__init__(model_inst, store_size)
        self.create_fn['pcnt3'] = self.create_pcnt3

    def create_pcnt3(self, filepack, h5file_name, num_line):
        crop_size = self.model_inst.crop_size
        h5file = filepack.write_h5(h5file_name)
        h5file.create_dataset(
            'pcnt3',
            (num_line,
                crop_size, crop_size, crop_size),
            chunks=(1,
                    crop_size, crop_size, crop_size),
            compression='lzf',
            dtype='f4')
        batch_data = np.empty(
            shape=(
                self.store_size,
                crop_size, crop_size, crop_size),
            dtype='f4')
        return h5file['pcnt3'], batch_data


class batch_truncd(batch_conv3):
    def __init__(self, model_inst, store_size=-1):
        super(batch_truncd, self).__init__(model_inst, store_size)
        self.create_fn['truncd'] = self.create_truncd

    def create_truncd(self, filepack, h5file_name, num_line):
        crop_size = self.model_inst.crop_size
        h5file = filepack.write_h5(h5file_name)
        h5file.create_dataset(
            'truncd',
            (num_line,
                crop_size, crop_size, crop_size),
            chunks=(1,
                    crop_size, crop_size, crop_size),
            compression='lzf',
            dtype='f4')
        batch_data = np.empty(
            shape=(
                self.store_size,
                crop_size, crop_size, crop_size),
            dtype='f4')
        return h5file['truncd'], batch_data


class batch_tsdf3(batch_conv3):
    def __init__(self, model_inst, store_size=-1):
        super(batch_tsdf3, self).__init__(model_inst, store_size)
        self.create_fn['tsdf3'] = self.create_tsdf3

    def create_tsdf3(self, filepack, h5file_name, num_line):
        crop_size = self.model_inst.crop_size
        h5file = filepack.write_h5(h5file_name)
        h5file.create_dataset(
            'tsdf3',
            (num_line,
                crop_size, crop_size, crop_size,
                3),
            chunks=(1,
                    crop_size, crop_size, crop_size,
                    3),
            compression='lzf',
            dtype='f4')
        batch_data = np.empty(
            shape=(
                self.store_size,
                crop_size, crop_size, crop_size,
                3),
            dtype='f4')
        return h5file['tsdf3'], batch_data


class batch_vxhit(batch_conv3):
    def __init__(self, model_inst, store_size=-1):
        super(batch_vxhit, self).__init__(model_inst, store_size)
        self.create_fn['vxhit'] = self.create_vxhit
        self.create_fn['pose_hit'] = self.create_pose_hit
        self.create_fn['pose_lab'] = self.create_pose_lab

    def create_vxhit(self, filepack, h5file_name, num_line):
        crop_size = self.model_inst.crop_size
        h5file = filepack.write_h5(h5file_name)
        h5file.create_dataset(
            'vxhit',
            (num_line,
                crop_size, crop_size, crop_size),
            chunks=(1,
                    crop_size, crop_size, crop_size),
            compression='lzf',
            dtype='f4')
        batch_data = np.empty(
            shape=(
                self.store_size,
                crop_size, crop_size, crop_size),
            dtype='f4')
        return h5file['vxhit'], batch_data

    def create_pose_hit(self, filepack, h5file_name, num_line):
        hmap_size = self.model_inst.hmap_size
        out_dim = self.model_inst.join_num
        h5file = filepack.write_h5(h5file_name)
        h5file.create_dataset(
            'pose_hit',
            (num_line,
                hmap_size, hmap_size, hmap_size,
                out_dim),
            chunks=(1,
                    hmap_size, hmap_size, hmap_size,
                    out_dim),
            compression='lzf',
            dtype='f4')
        batch_data = np.empty(
            shape=(
                self.store_size,
                hmap_size, hmap_size, hmap_size,
                out_dim),
            dtype='f4')
        return h5file['pose_hit'], batch_data

    def create_pose_lab(self, filepack, h5file_name, num_line):
        out_dim = self.model_inst.join_num
        h5file = filepack.write_h5(h5file_name)
        h5file.create_dataset(
            'pose_lab',
            (num_line, out_dim),
            chunks=(1, out_dim),
            compression='lzf',
            dtype='f4')
        batch_data = np.empty(
            shape=(
                self.store_size, out_dim),
            dtype='f4')
        return h5file['pose_lab'], batch_data


class batch_vxoff(batch_vxhit):
    def __init__(self, model_inst, store_size=-1):
        super(batch_vxoff, self).__init__(model_inst, store_size)
        self.create_fn['vxoff'] = self.create_vxoff

    def create_vxoff(self, filepack, h5file_name, num_line):
        hmap_size = self.model_inst.hmap_size
        out_dim = self.model_inst.join_num
        h5file = filepack.write_h5(h5file_name)
        h5file.create_dataset(
            'vxoff',
            (num_line,
                hmap_size, hmap_size, hmap_size,
                out_dim * 3),
            chunks=(1,
                    hmap_size, hmap_size, hmap_size,
                    out_dim * 3),
            compression='lzf',
            dtype='f4')
        batch_data = np.empty(
            shape=(
                self.store_size,
                hmap_size, hmap_size, hmap_size,
                out_dim * 3),
            dtype='f4')
        return h5file['vxoff'], batch_data


class batch_vxudir(batch_vxhit):
    def __init__(self, model_inst, store_size=-1):
        super(batch_vxudir, self).__init__(model_inst, store_size)
        self.create_fn['vxudir'] = self.create_vxudir

    def create_vxudir(self, filepack, h5file_name, num_line):
        hmap_size = self.model_inst.hmap_size
        out_dim = self.model_inst.join_num
        h5file = filepack.write_h5(h5file_name)
        h5file.create_dataset(
            'vxudir',
            (num_line,
                hmap_size, hmap_size, hmap_size,
                out_dim * 4),
            chunks=(1,
                    hmap_size, hmap_size, hmap_size,
                    out_dim * 4),
            compression='lzf',
            dtype='f4')
        batch_data = np.empty(
            shape=(
                self.store_size,
                hmap_size, hmap_size, hmap_size,
                out_dim * 4),
            dtype='f4')
        return h5file['vxudir'], batch_data
