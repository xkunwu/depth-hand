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

    def push_h5(self, name, rw='r'):
        file = h5py.File(name, rw)
        self.files[name] = file
        return file

    def push_file(self, name, rw='r'):
        file = open(name, rw)
        self.files[name] = file
        return file

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for _, file in self.files.items():
            file.close()
