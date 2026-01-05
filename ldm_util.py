from debug_util import *
from einops import rearrange
import confs
from torchvision import transforms
import torch
# from miscellaneous import ModelGetters
import numpy as np
import funcy as fc
from PIL import Image
def load_image_(img:np.ndarray):
    if not confs.LOOKAT.look_at_type=='WARP':
        assert img.shape==(256,256,3),img.shape
    height, width=256,256
    # img = np.asarray(img, dtype=np.float32) / 255.
    if 1:# from zero123 preprocess
        img=Image.fromarray(img)
        img=img.resize([height, width], Image.Resampling.LANCZOS)
        img=np.array(img)
    img = img / 255.
    assert img.max() < 1.01
    assert img.min() > -0.01
    # img = transforms.ToTensor()(img).unsqueeze(0).to('cuda:0')  #seems to perform hw3->3hw
    img = torch.Tensor(img).unsqueeze(0).to('cuda:0')
    img = img * 2 - 1
    # img = transforms.functional.resize(img, [height, width])
    assert img.shape==(1,height,width,3)
    return img



def get_pose(pose):
    p1 = pose[..., 0:1]
    p2 = torch.sin(pose[..., 1:2])
    p3 = torch.cos(pose[..., 1:2])
    p4 = pose[..., 2:]

    return torch.cat([p1, p2, p3, p4], dim=-1)
def _get_sds_loss(
    model, 
    cond_images:torch.Tensor, # bIV*bsz,256,256,3
    target_images:torch.Tensor, # bIV*bsz,256,256,3
    pose, # bIV,3
    ts_range, 
    bsz:int, bIV:int,
    noise=None,q0_noisePr=None,
) -> torch.Tensor: # bIV,
    batch = {}
    
    if confs.cache_4_ldm_get_input.enable and confs.cache_4_ldm_get_input.I_am_sure_that_cache_has_built_over:
        if 0:
            batch['image_target'] = None
            batch['image_cond'] = None
    else:
        batch['image_target'] = target_images
        batch['image_cond'] = cond_images
    
    _pose = get_pose(pose  ) # bIV,3
    assert _pose.shape[0]==bIV,'Please adjust confs.bIV'
    _pose = _pose.repeat_interleave(bsz,dim=0) # bIV*bsz,3
    batch['T'] = _pose

    noise = noise.repeat(bIV,1,1,1) # bsz,... -> bIV*bsz,...
    
    mx = ts_range[1]
    mn = ts_range[0]
    ts=np.arange(mn, mx, (mx-mn) / bsz)
    ts = torch.from_numpy(ts).float()
    ts = ts.repeat(bIV,) # bsz, -> bIV*bsz,
        
    losses, _ = model.shared_step(
        batch, ts=ts, noise=noise,q0_noisePr=q0_noisePr,
        target_is_z_insteadof_rgb=False,
    )
    return losses

def get_sds_loss(
    l_pose:list[list], # bIV,3.  # [ [theta, azimuth, radius],[theta, azimuth, radius],.... ]
    model, 
    cond_images:torch.Tensor, # bIV*bsz,256,256,3
    target_images:torch.Tensor, # bIV*bsz,256,256,3
    num_repeat:int, bIV:int,
    noise=None,q0_noisePr=None,
):
    pose1 = torch.tensor(l_pose, device=model.device, dtype=torch.float32)
    #
    pose2 = torch.empty_like(pose1)
    pose2[:, 0] = -pose1[:, 0]            # -theta
    pose2[:, 1] = np.pi * 2 - pose1[:, 1]  # 2π - azimuth
    pose2[:, 2] = -pose1[:, 2]            # -radius
    
    # only suuport half now.
    
    losses1 = _get_sds_loss(
        model, cond_images, target_images, pose2, confs.CalScore_ts_range, num_repeat, bIV,
        noise=noise,q0_noisePr=q0_noisePr,
    )
    if confs.CalScore_half:
        losses2 = 0
    else:
        confs.cache_4_ldm_get_input.swap_cur()
        losses2 = _get_sds_loss(
            model, target_images, cond_images, pose1, confs.CalScore_ts_range, num_repeat, bIV,
            noise=noise,
        )
    losses = losses1 + losses2
    # print(f"{losses=}", )
    return losses

def get_sds_loss_requireGrad(
    l_pose:list[list[torch.Tensor]], # bIV,3.  # [ [theta, azimuth, radius],[theta, azimuth, radius],.... ]
    model, 
    cond_images:torch.Tensor, # bIV*bsz,256,256,3
    target_images:torch.Tensor, # bIV*bsz,256,256,3
    num_repeat:int, bIV:int,
    noise=None,
):
    # pose1 = torch.tensor(l_pose, device=model.device, dtype=torch.float32)
    pose1 = torch.stack([torch.stack(_l) for _l in l_pose]).to(model.device)
    #
    pose2 = torch.stack([
        -pose1[:, 0], 
        np.pi * 2 - pose1[:, 1], 
        -pose1[:, 2]
    ], dim=1)
    
    assert pose1.requires_grad
    assert pose2.requires_grad
    
    losses1 = _get_sds_loss(
        model, cond_images, target_images, pose2, confs.Lr.ts_range, num_repeat, bIV,
        noise=noise,
    )
    # print(f"{losses1=}", )
    confs.cache_4_ldm_get_input.swap_cur()
    losses2 = _get_sds_loss(
        model, target_images, cond_images, pose1, confs.Lr.ts_range, num_repeat, bIV,
        noise=noise,
    )
    # print(f"{losses2=}", )
    losses = losses1 + losses2
    return losses
#-------------END----------------------