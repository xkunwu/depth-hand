import numpy as np
from psutil import virtual_memory


class batch_allot(object):
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
                    self.crop_size, self.crop_size,
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
                self.crop_size, self.crop_size,
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
