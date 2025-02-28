from debug_util import *
from einops import rearrange
import root_config
from torchvision import transforms
import torch
from zero123.zero1.scy_run1 import main_run,main_run_inter
from miscellaneous import ModelGetters
import numpy as np
import funcy as fc
def get_loss(latent0,latent1,):
    def DDPM__get_loss( pred:torch.Tensor, target:torch.Tensor,  ):
        """
        copy from
        """
        loss_type='l2'
        mean=False
        if isinstance(pred,np.ndarray):
            pred=torch.from_numpy(pred)
        if isinstance(target,np.ndarray):
            target=torch.from_numpy(target)
        if loss_type == 'l1':
            loss:torch.Tensor = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")
        return loss
    if root_config.LATENT_TYPE=='A':
        if not root_config.scy_ddim_inter:
            assert latent0.shape==(4,32,32,)
            assert latent1.shape==(4,32,32,)
            ret= DDPM__get_loss(latent0,latent1)
            ret=ret.mean()
            ret=ret.cpu().item()
            return ret
        else:
            assert latent0.shape[1:] == (4, 32, 32,)
            assert latent1.shape[1:] == (4, 32, 32,)
            inter = DDPM__get_loss(latent0, latent1)
            ret = inter.mean()
            ret = ret.cpu().item()
            return ret,inter
    else:
        raise ValueError(root_config.LATENT_TYPE)
def get_generation(input_image,l_xyz:list):
    models = ModelGetters.ModelsGetter.set_param( )
    xs = []
    ys = []
    zs = []
    for i, (x, y, z) in enumerate(l_xyz):
        xs.append(x)
        ys.append(y)
        zs.append(z)
    lBB_cfg=root_config.lBB
    if  not (root_config.scy_ddim_inter or lBB_cfg.enable):
        _, _, _, _, _, _, ret = main_run(
            models, 'cuda:0', None, 'angles_gen',
            xs, ys, zs, raw_im=input_image,
            n_samples=1,
            ddim_steps=root_config.ddim_steps__get_generation,
            batch_sample=True,
            no_decoder=True,
        )
        return ret
    else:
        if lBB_cfg.enable:
            ib_2_inter_features:list =[]
            for ib in range(lBB_cfg.ddim_bsz):
                #inter_features: n t z h w
                final_feature, inter_features, inter_index = main_run_inter(
                    models, 'cuda:0',
                    xs, ys, zs, raw_im=input_image,
                    n_samples=1,
                    ddim_steps=root_config.ddim_steps__get_generation,
                    batch_sample=True,
                    no_decoder=True,
                )
                ib_2_inter_features.append(inter_features)
            ib_2_inter_features:torch.Tensor  =torch.tensor(  ib_2_inter_features )# ddim_bsz, n t z h w
            if 1:# refer to redmi learn/chunk1.py
                temp:tuple= ib_2_inter_features.chunk(lBB_cfg.latentBB_bsz)
                temp:list = [x.mean(axis=0) for x in temp]
                ib_2_inter_features:torch.Tensor   = torch.stack(temp)# latentBB_bsz, n t z h w
            if 1:
                assert ib_2_inter_features.shape == (lBB_cfg.latentBB_bsz, root_config.NUM_REF, len(lBB_cfg.ddim_steps),
                                                     4, 32, 32), ib_2_inter_features.shape
                inter_features = rearrange(ib_2_inter_features, 'b n t z h w -> n t b z h w')
                assert inter_features.shape == (root_config.NUM_REF, len(lBB_cfg.ddim_steps), lBB_cfg.latentBB_bsz,
                                                4, 32, 32), ib_2_inter_features.shape
            return inter_features
        else:
            final_feature ,inter_features,inter_index= main_run_inter(
                models, 'cuda:0',
                xs, ys, zs, raw_im=input_image,
                n_samples=1,
                ddim_steps=root_config.ddim_steps__get_generation,
                batch_sample=True,
                no_decoder=True,
            )
        return inter_features
#-----------------------------copy from idpose
from PIL import Image
def load_image_(img:np.ndarray):
    if not root_config.LOOKAT.look_at_type=='WARP':
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
    
    if root_config.cache_4_ldm_get_input.enable and root_config.cache_4_ldm_get_input.I_am_sure_that_cache_has_built_over:
        if 0:
            batch['image_target'] = None
            batch['image_cond'] = None
    else:
        batch['image_target'] = target_images
        batch['image_cond'] = cond_images
    
    _pose = get_pose(pose  ) # bIV,3
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
        scy_target_is_z_insteadof_rgb=False,
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
    
    losses = _get_sds_loss(
        model, cond_images, target_images, pose2, root_config.CalScoreB_ts_range, num_repeat, bIV,
        noise=noise,q0_noisePr=q0_noisePr,
    )
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
        model, cond_images, target_images, pose2, root_config.Lr.ts_range, num_repeat, bIV,
        noise=noise,
    )
    # print(f"{losses1=}", )
    root_config.cache_4_ldm_get_input.swap_cur()
    losses2 = _get_sds_loss(
        model, target_images, cond_images, pose1, root_config.Lr.ts_range, num_repeat, bIV,
        noise=noise,
    )
    # print(f"{losses2=}", )
    losses = losses1 + losses2
    return losses
#-------------END----------------------