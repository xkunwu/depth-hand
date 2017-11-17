import h5py


class break_with(object):
    class Break(Exception):
      """Break out of the with statement"""

    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self.value.__enter__()

    def __exit__(self, etype, value, traceback):
        error = self.value.__exit__(etype, value, traceback)
        if etype == self.Break:
            return True
        return error


class file_pack:
    def __init__(self):
        self.files = {}

    def write_h5(self, name):
        if name in self.files:
            self.files[name].close()
        file = h5py.File(name, 'w')
        self.files[name] = file
        return file

    def write_file(self, name):
        if name in self.files:
            self.files[name].close()
        file = open(name, 'w')
        self.files[name] = file
        return file

    def push_h5(self, name):
        if name not in self.files:
            file = h5py.File(name, 'r')
            self.files[name] = file
        return self.files[name]

    def push_file(self, name):
        if name not in self.files:
            file = open(name, 'r')
            self.files[name] = file
        return self.files[name]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for _, file in self.files.items():
            file.close()
