import confs,os,sys,shutil
from skimage.io import imread, imsave
import numpy as np
import torch
import random
import os,sys,math,functools,inspect
from pathlib import Path
import PIL
from einops import rearrange

def debug_imsave(path__rel_to__path_4debug,arr):
    #
    if isinstance(path__rel_to__path_4debug,Path):
        path__rel_to__path_4debug=str(path__rel_to__path_4debug)
    assert isinstance(path__rel_to__path_4debug,str)
    #
    if isinstance(arr,PIL.Image.Image):
        # Convert PIL image to NumPy array
        # arr = np.array(arr)
        arr = np.asarray(arr)
    if isinstance(arr,torch.Tensor):
        arr = np.array(arr.cpu())
    assert isinstance(arr,np.ndarray)
    #
    full_path=os.path.join(confs.path_4debug,path__rel_to__path_4debug)
    os.makedirs(os.path.dirname(full_path),exist_ok=1)
    print(f"[debug_imsave]saving...",end=" ",flush = True)
    imsave(full_path,arr)
    print(f"save to \"{full_path}\"")


def save_Xchw(path__rel_to__path_4debug,    Xchw,):
    if isinstance(Xchw,torch.Tensor):
        chw=Xchw.cpu().numpy()
    while(chw.ndim>3):
        chw=chw[0]
    img_arr = chw[:3]
    img_arr=rearrange(img_arr,'c h w -> h w c')
    #norm to 0-1
    img_arr=(img_arr-img_arr.min())/(img_arr.max()-img_arr.min())
    if 1:
        if isinstance(path__rel_to__path_4debug,Path):
            path__rel_to__path_4debug=str(path__rel_to__path_4debug)
        assert isinstance(path__rel_to__path_4debug,str)
        full_path=os.path.join(confs.path_4debug,path__rel_to__path_4debug)
        os.makedirs(os.path.dirname(full_path),exist_ok=1)
    imsave(full_path, img_arr)
    
def print_tensor_info(a,sep='\n',p=1):#print Tensor info
    if p<1:
        if random.random()>=p:
            return
    print(a.shape,end=sep)
    print(a.mean(),end=sep)
    print(a.var(),end=sep)
    print(a.min(),end=sep)
    print(a.max())
    
ppp=print_tensor_info