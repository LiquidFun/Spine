import importlib

import torch


def get_object_from_string(full_path):
    """
    Given a string of format 'module.submodule.function_name',
    imports the module and returns the function object.
    """
    module_path, function_name = full_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    function = getattr(module, function_name)
    return function


class ProxyClass(torch.nn.Module):
    def __init__(self, cls, *args, **kwargs):
        super().__init__()
        func = get_object_from_string(cls)
        self._cls = func(*args, **kwargs)
        self.__class__.__name__ = type(self._cls).__name__

    def __getattr__(self, name):
        if name == "_cls":
            return super().__getattr__(name)
        return getattr(self._cls, name)

    def __call__(self, *args, **kwargs):
        return self._cls(*args, **kwargs)

    def __getitem__(self, item):
        return self._cls[item]
