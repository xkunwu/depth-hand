import numpy as np
from psutil import virtual_memory


class batch_allot(object):
    def __init__(self, model_inst, store_size=-1):
        import multiprocessing
        self.store_size = multiprocessing.cpu_count() * (2 << 10)
        if 0 < store_size:
            self.store_size = min(self.store_size, store_size)
        self.crop_size = model_inst.crop_size
        self.out_dim = model_inst.out_dim
        self.entry = {
            'crop2': np.empty(
                shape=(
                    self.store_size,
                    self.crop_size, self.crop_size),
                dtype=float),
            'pose_c': np.empty(
                shape=(self.store_size, self.out_dim),
                dtype=float),
        }
        self.store_bytes = 0
        for _, v in self.entry.items():
            self.store_bytes += v.nbytes
        self.create_fn = {
            'crop2': self.create_crop2,
            'pose_c': self.create_pose_c,
        }

    def create_crop2(self, filepack, h5file_name, num_line):
        h5file = filepack.write_h5(h5file_name)
        h5file.create_dataset(
            'crop2',
            (num_line,
                self.crop_size, self.crop_size),
            chunks=(1,
                    self.crop_size, self.crop_size),
            compression='lzf',
            # dtype=np.float32)
            dtype=float)
        return h5file['crop2']

    def create_pose_c(self, filepack, h5file_name, num_line):
        h5file = filepack.write_h5(h5file_name)
        h5file.create_dataset(
            'pose_c',
            (num_line, self.out_dim),
            compression='lzf',
            # dtype=np.float32)
            dtype=float)
        return h5file['pose_c']
# class batch_allot(object):
#     def __init__(self, model_inst):
#         self.batch_size = model_inst.batch_size
#         self.crop_size = model_inst.crop_size
#         self.out_dim = model_inst.out_dim
#         self.num_channel = model_inst.num_channel
#         self.num_appen = model_inst.num_appen
#         batch_data = {
#             'batch_index': np.empty(
#                 shape=(self.batch_size, 1), dtype=np.int32),
#             'batch_frame': np.empty(
#                 shape=(
#                     self.batch_size,
#                     self.crop_size, self.crop_size,
#                     self.num_channel),
#                 # dtype=np.float32),
#                 dtype=float),
#             'batch_poses': np.empty(
#                 shape=(self.batch_size, self.out_dim),
#                 # dtype=np.float32),
#                 dtype=float),
#             'batch_resce': np.empty(
#                 shape=(self.batch_size, self.num_appen),
#                 # dtype=np.float32),
#                 dtype=float),
#         }
#         self.batch_bytes = \
#             batch_data['batch_index'].nbytes + batch_data['batch_frame'].nbytes + \
#             batch_data['batch_poses'].nbytes + batch_data['batch_resce'].nbytes
#         self.batch_beg = 0
#
#     def allot(self, store_size=-1):
#         store_cap_mult = (virtual_memory().total >> 2) // self.batch_bytes
#         store_cap = store_cap_mult * self.batch_size
#         if 0 > store_size:
#             self.store_size = store_cap
#         else:
#             self.store_size = min(store_cap, store_size)
#         self.store_bytes = self.store_size * self.batch_bytes / self.batch_size
#         self.store_beg = 0
#         self.batch_index = np.empty(
#             shape=(self.store_size, 1), dtype=np.int32)
#         self.batch_frame = np.empty(
#             shape=(
#                 self.store_size,
#                 self.crop_size, self.crop_size,
#                 self.num_channel),
#             # dtype=np.float32)
#             dtype=float)
#         self.batch_poses = np.empty(
#             shape=(self.store_size, self.out_dim),
#             # dtype=np.float32)
#             dtype=float)
#         self.batch_resce = np.empty(
#             shape=(self.store_size, self.num_appen),
#             # dtype=np.float32)
#             dtype=float)


class batch_allot_conv3(batch_allot):
    def __init__(self, model_inst):
        self.batch_size = model_inst.batch_size
        self.crop_size = model_inst.crop_size
        self.out_dim = model_inst.out_dim
        self.num_channel = model_inst.num_channel
        self.num_appen = model_inst.num_appen
        batch_data = {
            'batch_index': np.empty(
                shape=(self.batch_size, 1), dtype=np.int32),
            'batch_frame': np.empty(
                shape=(
                    self.batch_size,
                    self.crop_size, self.crop_size, self.crop_size,
                    self.num_channel),
                # dtype=np.float32),
                dtype=float),
            'batch_poses': np.empty(
                shape=(self.batch_size, self.out_dim),
                # dtype=np.float32),
                dtype=float),
            'batch_resce': np.empty(
                shape=(self.batch_size, self.num_appen),
                # dtype=np.float32),
                dtype=float),
        }
        self.batch_bytes = \
            batch_data['batch_index'].nbytes + batch_data['batch_frame'].nbytes + \
            batch_data['batch_poses'].nbytes + batch_data['batch_resce'].nbytes
        self.batch_beg = 0

    def allot(self, store_size=-1):
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
                self.crop_size, self.crop_size, self.crop_size,
                self.num_channel),
            # dtype=np.float32)
            dtype=float)
        self.batch_poses = np.empty(
            shape=(self.store_size, self.out_dim),
            # dtype=np.float32)
            dtype=float)
        self.batch_resce = np.empty(
            shape=(self.store_size, self.num_appen),
            # dtype=np.float32)
            dtype=float)


class batch_allot_hmap(object):
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
            'batch_poses': np.empty(
                shape=(self.batch_size, self.out_dim * 3),
                # dtype=np.float32),
                dtype=float),
            'batch_hmap2': np.empty(
                shape=(
                    self.batch_size,
                    self.hmap_size, self.hmap_size,
                    self.out_dim),
                # dtype=np.float32)
                dtype=float),
            'batch_olmap': np.empty(
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
            batch_data['batch_poses'].nbytes + batch_data['batch_resce'].nbytes + \
            batch_data['batch_hmap2'].nbytes + batch_data['batch_olmap'].nbytes + \
            batch_data['batch_uomap'].nbytes
        self.batch_beg = 0

    def allot(self, store_size=-1):
        from psutil import virtual_memory
        # use up to 1/4 total memory
        store_cap_mult = (virtual_memory().total >> 2) // self.batch_bytes
        # maximal entries in a processing cycle
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
        self.batch_poses = np.empty(
            shape=(self.store_size, self.out_dim * 3),
            # dtype=np.float32)
            dtype=float)
        self.batch_hmap2 = np.empty(
            shape=(self.store_size, self.hmap_size, self.hmap_size, self.out_dim),
            # dtype=np.float32)
            dtype=float)
        self.batch_olmap = np.empty(
            shape=(self.store_size, self.hmap_size, self.hmap_size, self.out_dim),
            # dtype=np.float32)
            dtype=float)
        self.batch_uomap = np.empty(
            shape=(self.store_size, self.hmap_size, self.hmap_size, self.out_dim * 3),
            # dtype=np.float32)
            dtype=float)
        self.batch_resce = np.empty(
            shape=(self.store_size, self.num_appen),
            # dtype=np.float32)
            dtype=float)


class batch_allot_vxhit(object):
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
                    self.crop_size, self.crop_size, self.crop_size,
                    self.num_channel),
                # dtype=np.float32),
                dtype=float),
            'batch_poses': np.empty(
                shape=(self.batch_size, self.out_dim * 3),
                # dtype=np.float32),
                dtype=float),
            # 'batch_vxhit': np.empty(
            #     shape=(
            #         self.batch_size,
            #         self.hmap_size, self.hmap_size, self.hmap_size,
            #         self.out_dim),
            #     # dtype=np.float32)
            #     dtype=float),
            'batch_vxhit': np.empty(
                shape=(
                    self.batch_size,
                    self.out_dim),
                # dtype=np.float32)
                dtype=float),
            'batch_resce': np.empty(
                shape=(self.batch_size, self.num_appen),
                # dtype=np.float32),
                dtype=float),
        }
        self.batch_bytes = \
            batch_data['batch_index'].nbytes + batch_data['batch_frame'].nbytes + \
            batch_data['batch_poses'].nbytes + batch_data['batch_resce'].nbytes + \
            batch_data['batch_vxhit'].nbytes
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
                self.crop_size, self.crop_size, self.crop_size,
                self.num_channel),
            # dtype=np.float32)
            dtype=float)
        self.batch_poses = np.empty(
            shape=(self.store_size, self.out_dim * 3),
            # dtype=np.float32)
            dtype=float)
        # self.batch_vxhit = np.empty(
        #     shape=(
        #         self.store_size,
        #         self.hmap_size, self.hmap_size, self.hmap_size,
        #         self.out_dim),
        #     # dtype=np.float32)
        #     dtype=float)
        self.batch_vxhit = np.empty(
            shape=(
                self.store_size,
                self.out_dim),
            # dtype=np.float32)
            dtype=float)
        self.batch_resce = np.empty(
            shape=(self.store_size, self.num_appen),
            # dtype=np.float32)
            dtype=float)


class batch_allot_vxdir(object):
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
                    self.crop_size, self.crop_size, self.crop_size,
                    self.num_channel),
                # dtype=np.float32),
                dtype=float),
            'batch_poses': np.empty(
                shape=(self.batch_size, self.out_dim * 3),
                # dtype=np.float32),
                dtype=float),
            # 'batch_vxhit': np.empty(
            #     shape=(
            #         self.batch_size,
            #         self.hmap_size, self.hmap_size, self.hmap_size,
            #         self.out_dim),
            #     # dtype=np.float32)
            #     dtype=float),
            'batch_vxhit': np.empty(
                shape=(
                    self.batch_size,
                    self.out_dim),
                # dtype=np.float32)
                dtype=float),
            'batch_olmap': np.empty(
                shape=(
                    self.batch_size,
                    self.hmap_size, self.hmap_size, self.hmap_size,
                    self.out_dim),
                # dtype=np.float32)
                dtype=float),
            'batch_uomap': np.empty(
                shape=(
                    self.batch_size,
                    self.hmap_size, self.hmap_size, self.hmap_size,
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
            batch_data['batch_poses'].nbytes + batch_data['batch_resce'].nbytes + \
            batch_data['batch_vxhit'].nbytes + batch_data['batch_olmap'].nbytes + \
            batch_data['batch_uomap'].nbytes
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
                self.crop_size, self.crop_size, self.crop_size,
                self.num_channel),
            # dtype=np.float32)
            dtype=float)
        self.batch_poses = np.empty(
            shape=(self.store_size, self.out_dim * 3),
            # dtype=np.float32)
            dtype=float)
        # self.batch_vxhit = np.empty(
        #     shape=(
        #         self.store_size,
        #         self.hmap_size, self.hmap_size, self.hmap_size,
        #         self.out_dim),
        #     # dtype=np.float32)
        #     dtype=float)
        self.batch_vxhit = np.empty(
            shape=(
                self.store_size,
                self.out_dim),
            # dtype=np.float32)
            dtype=float)
        self.batch_olmap = np.empty(
            shape=(
                self.store_size,
                self.hmap_size, self.hmap_size, self.hmap_size,
                self.out_dim),
            # dtype=np.float32)
            dtype=float)
        self.batch_uomap = np.empty(
            shape=(
                self.store_size,
                self.hmap_size, self.hmap_size, self.hmap_size,
                self.out_dim * 3),
            # dtype=np.float32)
            dtype=float)
        self.batch_resce = np.empty(
            shape=(self.store_size, self.num_appen),
            # dtype=np.float32)
            dtype=float)