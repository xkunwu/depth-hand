import numpy as np
from psutil import virtual_memory


class batch_allot(object):
    def __init__(self, batch_size, image_size, out_dim, num_channel, num_appen):
        self.batch_size = batch_size
        self.image_size = image_size
        self.out_dim = out_dim
        self.num_channel = num_channel
        self.num_appen = num_appen
        batch_data = {
            'batch_index': np.empty(
                shape=(batch_size, 1), dtype=np.int32),
            'batch_frame': np.empty(
                shape=(
                    batch_size,
                    image_size, image_size,
                    num_channel),
                # dtype=np.float32),
                dtype=float),
            'batch_poses': np.empty(
                shape=(batch_size, out_dim),
                # dtype=np.float32),
                dtype=float),
            'batch_resce': np.empty(
                shape=(batch_size, num_appen),
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
                self.image_size, self.image_size,
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

    def fetch_store(self):
        if self.store_beg >= self.file_size:
            return False
        store_end = min(
            self.store_beg + self.store_size,
            self.file_size
        )
        self.store_size = store_end - self.store_beg
        self.batch_index = self.store_file['index'][self.store_beg:store_end, ...]
        self.batch_frame = self.store_file['frame'][self.store_beg:store_end, ...]
        self.batch_poses = self.store_file['poses'][self.store_beg:store_end, ...]
        self.batch_resce = self.store_file['resce'][self.store_beg:store_end, ...]
        self.store_beg = store_end
        self.batch_beg = 0
        return True

    def assign(self, store_file):
        self.store_file = store_file
        self.file_size = self.store_file['index'].shape[0]
        self.store_size = min(
            self.file_size,
            ((virtual_memory().total >> 1) // self.batch_bytes) * self.batch_size
        )
        self.store_beg = 0
        self.fetch_store()

    def fetch_batch(self):
        # if self.batch_beg >= self.store_size:
        #     if not self.fetch_store():
        #         return None
        # batch_end = min(
        #     self.batch_beg + self.batch_size,
        #     self.store_size
        # )
        batch_end = self.batch_beg + self.batch_size
        if batch_end >= self.store_size:
            if not self.fetch_store():
                return None
            batch_end = self.batch_beg + self.batch_size
            if batch_end >= self.store_size:
                return None
        batch_data = {
            'batch_index': self.batch_index[self.batch_beg:batch_end, ...],
            'batch_frame': self.batch_frame[self.batch_beg:batch_end, ...],
            'batch_poses': self.batch_poses[self.batch_beg:batch_end, ...],
            'batch_resce': self.batch_resce[self.batch_beg:batch_end, ...]
        }
        self.batch_beg = batch_end
        return batch_data


class batch_allot_conv3(batch_allot):
    def __init__(self, batch_size, image_size, out_dim, num_channel, num_appen):
        self.batch_size = batch_size
        self.image_size = image_size
        self.out_dim = out_dim
        self.num_channel = num_channel
        self.num_appen = num_appen
        batch_data = {
            'batch_index': np.empty(
                shape=(batch_size, 1), dtype=np.int32),
            'batch_frame': np.empty(
                shape=(
                    batch_size,
                    image_size, image_size, image_size,
                    num_channel),
                # dtype=np.float32),
                dtype=float),
            'batch_poses': np.empty(
                shape=(batch_size, out_dim),
                # dtype=np.float32),
                dtype=float),
            'batch_resce': np.empty(
                shape=(batch_size, num_appen),
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
                self.image_size, self.image_size, self.image_size,
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

    # def fetch_store(self):
    #     if self.store_beg >= self.file_size:
    #         return False
    #     store_end = min(
    #         self.store_beg + self.store_size,
    #         self.file_size
    #     )
    #     self.store_size = store_end - self.store_beg
    #     self.batch_index = self.store_file['index'][self.store_beg:store_end, ...]
    #     self.batch_frame = self.store_file['frame'][self.store_beg:store_end, ...]
    #     self.batch_poses = self.store_file['poses'][self.store_beg:store_end, ...]
    #     self.batch_resce = self.store_file['resce'][self.store_beg:store_end, ...]
    #     self.store_beg = store_end
    #     self.batch_beg = 0
    #     return True
    #
    # def assign(self, store_file):
    #     self.store_file = store_file
    #     self.file_size = self.store_file['index'].shape[0]
    #     self.store_size = min(
    #         self.file_size,
    #         ((virtual_memory().total >> 1) // self.batch_bytes) * self.batch_size
    #     )
    #     self.store_beg = 0
    #     self.fetch_store()
    #
    # def fetch_batch(self):
    #     batch_end = self.batch_beg + self.batch_size
    #     if batch_end >= self.store_size:
    #         if not self.fetch_store():
    #             return None
    #         batch_end = self.batch_beg + self.batch_size
    #         if batch_end >= self.store_size:
    #             return None
    #     batch_data = {
    #         'batch_index': self.batch_index[self.batch_beg:batch_end, ...],
    #         'batch_frame': self.batch_frame[self.batch_beg:batch_end, ...],
    #         'batch_poses': self.batch_poses[self.batch_beg:batch_end, ...],
    #         'batch_resce': self.batch_resce[self.batch_beg:batch_end, ...]
    #     }
    #     self.batch_beg = batch_end
    #     return batch_data


class batch_allot_loc2(batch_allot):
    def __init__(self, batch_size, image_size, out_dim, num_channel, num_appen):
        self.batch_size = batch_size
        self.image_size = image_size
        self.out_dim = out_dim
        self.num_channel = num_channel
        self.num_appen = num_appen
        batch_data = {
            'batch_index': np.empty(
                shape=(batch_size, 1), dtype=np.int32),
            'batch_frame': np.empty(
                shape=(
                    batch_size,
                    image_size, image_size,
                    num_channel),
                # dtype=np.float32),
                dtype=float),
            'batch_poses': np.empty(
                shape=(batch_size, out_dim),
                # dtype=np.float32),
                dtype=float),
            'batch_resce': np.empty(
                shape=(batch_size, num_appen),
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
                self.image_size, self.image_size,
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

    # def fetch_store(self):
    #     if self.store_beg >= self.file_size:
    #         return False
    #     store_end = min(
    #         self.store_beg + self.store_size,
    #         self.file_size
    #     )
    #     self.store_size = store_end - self.store_beg
    #     self.batch_index = self.store_file['index'][self.store_beg:store_end, ...]
    #     self.batch_frame = self.store_file['frame'][self.store_beg:store_end, ...]
    #     self.batch_poses = self.store_file['poses'][self.store_beg:store_end, ...]
    #     self.batch_resce = self.store_file['resce'][self.store_beg:store_end, ...]
    #     self.store_beg = store_end
    #     self.batch_beg = 0
    #     return True
    #
    # def assign(self, store_file):
    #     self.store_file = store_file
    #     self.file_size = self.store_file['index'].shape[0]
    #     self.store_size = min(
    #         self.file_size,
    #         ((virtual_memory().total >> 1) // self.batch_bytes) * self.batch_size
    #     )
    #     self.store_beg = 0
    #     self.fetch_store()
    #
    # def fetch_batch(self):
    #     # if self.batch_beg >= self.store_size:
    #     #     if not self.fetch_store():
    #     #         return None
    #     # batch_end = min(
    #     #     self.batch_beg + self.batch_size,
    #     #     self.store_size
    #     # )
    #     batch_end = self.batch_beg + self.batch_size
    #     if batch_end >= self.store_size:
    #         if not self.fetch_store():
    #             return None
    #         batch_end = self.batch_beg + self.batch_size
    #         if batch_end >= self.store_size:
    #             return None
    #     batch_data = {
    #         'batch_index': self.batch_index[self.batch_beg:batch_end, ...],
    #         'batch_frame': self.batch_frame[self.batch_beg:batch_end, ...],
    #         'batch_poses': self.batch_poses[self.batch_beg:batch_end, ...],
    #         'batch_resce': self.batch_resce[self.batch_beg:batch_end, ...]
    #     }
    #     self.batch_beg = batch_end
    #     return batch_data
