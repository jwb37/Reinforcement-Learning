import torch
import imageio

def numpy_greyscale_to_tensor(n):
    n = n / 255.0
    t = torch.tensor(n, dtype=torch.float)
    t = t.unsqueeze(0)
    t = t.unsqueeze(0)
    return t

def numpy_rgb_to_tensor(n):
    n = n.dot([0.298, 0.587, 0.114])
    return numpy_greyscale_to_tensor(n)

def tensor_to_numpy_greyscale(t):
    t = t.detach()
    t = t[0,0,:,:] * 255.0
    t = t.transpose(0,1)
    return t.cpu().numpy().astype('uint8')

def tensor_to_numpy_rgb(t):
    t = t.detach()
    t = t[0,:,:,:] * 255.0
    t = torch.movedim(t, 0, -1)
    t = t.transpose(0,1)
    return t.cpu().numpy().astype('uint8')

def save_tensor_as_image(t, path):
    array = tensor_to_numpy_greyscale(t)
    imageio.imwrite(path, array)

def save_tensor_list_as_video(l, path):
    if l[0].shape[1] == 3:
        conv_func = tensor_to_numpy_rgb
    elif l[0].shape[1] == 1:
        conv_func = tensor_to_numpy_greyscale

    l = [conv_func(t) for t in l]
    imageio.mimwrite(path, l, 'GIF-PIL')

