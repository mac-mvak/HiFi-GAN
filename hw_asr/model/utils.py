import torch


def _conv_shape_transform1d(L_in, kernel_size,
                            padding= 0,
                            dilation=1, 
                            stride=1, **batch):
    ans = torch.floor((L_in + 2 * padding - dilation * (kernel_size - 1)-1)/stride + 1).int()
    return ans


def _conv_shape_transform2d(L_in, kernel_size,
                            padding=torch.tensor([0,0]),
                            dilation=torch.tensor([1, 1]), 
                            stride=torch.tensor([1, 1]), dim=0, **batch):
    kernel_size = kernel_size[dim]
    padding = padding[dim]
    dilation = dilation[dim]
    stride = stride[dim]
    ans = torch.floor((L_in + 2 * padding - dilation * (kernel_size - 1)-1)/stride + 1).int()
    return ans




