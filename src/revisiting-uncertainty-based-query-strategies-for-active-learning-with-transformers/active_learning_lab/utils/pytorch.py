import torch

from contextlib import contextmanager


# inspired by:
#     https://discuss.pytorch.org
#     /t/is-there-anything-wrong-with-setting-default-tensor-type-to-cuda/27949/4
@contextmanager
def default_tensor_type(tensor_type):
    default_type = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    torch.set_default_tensor_type(tensor_type)
    yield
    torch.set_default_tensor_type(default_type)
