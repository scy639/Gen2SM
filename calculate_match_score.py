from debug_util import *
import math
import methodtools

import numpy as np

import ldm_util
import confs
# from LatentDatabase import *
import torch
from gen6d.Gen6D.pipeline import    Estimator4Co3dEval
from ldm_util import get_sds_loss,get_sds_loss_requireGrad
import funcy as fc
from imports import *


import json
import misc_util
from pathlib import Path


def get_totalScore(i2score: list, i2weight: list):
    assert len(i2score) == len(i2weight)
    ret = 0
    for score, weight in zip(i2score, i2weight):
        ret += score * weight
    ret /= sum(i2weight)
    return ret
@fc.print_durations( threshold=0.01)
def calculate_Set_Matching_Score(estimator0: Estimator4Co3dEval, image1, q1_elev_deg, shift_azim,
        noise, ldm_model,
        l_score_from_cache4lB=None,
        #
        image1_path_4_get_input_cache=None,
        ):
    """
    for search stage
    ldb must have been built
    """
    if confs.cache_4_ldm_get_input.enable:
        assert image1_path_4_get_input_cache
    if   l_score_from_cache4lB:
        l_score_=l_score_from_cache4lB[:]
    else:
        if 1: # confs.lBB.enable:
            db0 = estimator0.sparseViewPoseEstimator.ref_database
            len0 = len(db0.get_img_ids()  )
        l_score_ = []
        i2weight = []
        l_i=[]
        # l_i+=[db0.ID_zero123InputImage]
        l_i+=range(0,len0,confs.CalScore__db0_iterate_interval )
        if 1: # confs.BATCH_multiple_iv:
            bIV = confs.bIV
            bsz = confs.Ls_bsz
            if 1:
                cond_images = image1.repeat(bsz*bIV, 1, 1, 1)
            l_l_i = [l_i[i:i + bIV] for i in range(0, len(l_i), bIV)]
            del l_i
            for _l_i in l_l_i:
                l_pose = []
                cur_target_id = []
                target_images = torch.empty( (bIV,256,256,3) )
                for ii,i in enumerate(_l_i):
                    # I0gi简称i; q1 warp又称image1, 简称q1
                    elev_i,azim_i,_,_,dis0=db0.get_elev_az_many_relaToBase(str(i))
                    elev_iToq1=q1_elev_deg-elev_i
                    if 1:
                        q0_elev,q0_az=pose_util.R_or_pose__2__elev_azim_inDeg_A( np.array(db0.get_zero123Input_info()['pose'] ) )
                        assert math.isclose(q0_az,0),q0_az
                    az_iToq1=shift_azim-azim_i
                    elev_iToq1=np.deg2rad(elev_iToq1)
                    az_iToq1=np.deg2rad(az_iToq1)
                    I0giRGB_tensor = db0.get_image_B(i) # TODO 优化。
                    target_images[ii] = I0giRGB_tensor
                    cur_target_id.append( db0.get_image_path(i) )
                    l_pose.append([-elev_iToq1, az_iToq1, 0])
                target_images = target_images.repeat_interleave(bsz,dim=0) #必须加dim,否则use flattened input, ret a flat output
                cur_target_id = '|'.join(cur_target_id)
                confs.cache_4_ldm_get_input.set_cur(image1_path_4_get_input_cache, cur_target_id, )
                with torch.no_grad():
                    losses = get_sds_loss(
                        l_pose,
                        ldm_model,
                        cond_images, target_images,
                        confs.Ls_bsz, bIV=bIV,
                        noise=noise,
                    )
                for ii,i in enumerate(_l_i):
                    loss = losses[ii]
                    loss=loss.item()
                    if 1:
                        elev_i,azim_i,elev_rToi,az_rToi,_=db0.get_elev_az_many_relaToBase_B(str(i))
                        #shift_azim: az_w0Tow1 ~= az_rToq
                    score=-loss
                    l_score_.append(score)
                    weight=0#TODO dev ing
                    i2weight.append(weight)
            confs.cache_4_ldm_get_input.I_am_sure_that_cache_has_built_over=True
            confs.cache_4_ldm_get_input.I_am_sure_that_cache_has_built_over=True
    i2weight = [1] * len(l_score_)
    # print(f"{l_score_=}")
    totalScore = get_totalScore(l_score_, i2weight)
    return totalScore,l_score_

from miscellaneous import ModelGetters
@fc.print_durations()
@torch.enable_grad()
def refine(
        img_or_path,
        pose_or_elev_az_deg,
        estimator0: Estimator4Co3dEval,
        cache4Lr,
        image1_path_4_get_input_cache=None,
        noise=None,
        # ldm_model=Global.ldm_model,
        learning_rate=5.0,
        num_iter=None,#不要写=confs.xxx这又是那个坑，karg默认值会在一开始就被计算
        decay_interval=None,
):
    """
    for refine stage
    """
    if confs.cache_4_ldm_get_input.enable:
        assert image1_path_4_get_input_cache
    if 1:
        ModelGetters.ModelsGetter.set_param()
        ldm_model = ModelGetters.ModelsGetter.get_B()
        ldm_model = ldm_model['turncam']
    if noise is None:
        noise = np.random.randn(confs.Lr.Lr_latentB_bsz, 4, 32, 32)
        noise = torch.tensor(noise, dtype=ldm_model.dtype, device=ldm_model.device)
    if isinstance(img_or_path,str) or isinstance(img_or_path,Path):
        img = imread(img_or_path)
    else:
        img=img_or_path
    if isinstance(pose_or_elev_az_deg,np.ndarray):
        # q1_azim_deg其实就是B里的shift_azim
        q1_elev_deg, q1_azim_deg=pose_util.R_or_pose__2__elev_azim_inDeg_A(pose_or_elev_az_deg)
    else:
        q1_elev_deg, q1_azim_deg=pose_or_elev_az_deg
    img = ldm_util.load_image_(img)
    #q1 的
    device = 'cuda:0'
    q1_elev_deg=torch.tensor(q1_elev_deg,dtype=torch.float32, requires_grad=True, device=device)
    q1_azim_deg=torch.tensor(q1_azim_deg,dtype=torch.float32, requires_grad=True, device=device)
    radius = torch.tensor(0.0, requires_grad=True, device=device)
    if 0:
        param_list= [q1_elev_deg,q1_azim_deg,radius]
        optimizer = torch.optim.SGD(param_list, lr=learning_rate)
    else:
        param_groups = [
            {
                'params': [q1_elev_deg,q1_azim_deg],
                'lr':confs.Lr.elev_az_learning_rate,
            },
            {
                'params': [radius],
                'lr': 1.,
            },
        ]
        optimizer = torch.optim.SGD(param_groups,   )
    #----------------------------
    db0 = estimator0.sparseViewPoseEstimator.ref_database
    len0 = len(db0.get_img_ids())
    loss_traj = []
    pose_traj = [(q1_elev_deg.item(),q1_azim_deg.item(),radius.item()), ]
    for i_iter in range(num_iter):
        elev_az_r_deg_loss=cache4Lr.i_iter_2_elev_az_r_deg_loss.get(i_iter,None)
        if elev_az_r_deg_loss is None:
            sum_loss=0
            optimizer.zero_grad()
            if 1:
                _start_i=0
                if confs.Lr.db0_iterate_strategy.roll:
                    _start_i=i_iter % confs.Lr.db0_iterate_strategy.interval
            l_i = range(_start_i, len0, confs.Lr.db0_iterate_strategy.interval)
            if 1: # confs.BATCH_multiple_iv:
                bIV = confs.bIV
                bsz = confs.Lr.Lr_latentB_bsz
                if 1:
                    cond_images = img.repeat(bsz*bIV, 1, 1, 1)
                l_l_i = [l_i[i:i + bIV] for i in range(0, len(l_i), bIV)]
                assert l_l_i[0]==l_l_i[-1],'Please adjust confs.bIV'
                del l_i
                for _l_i in l_l_i:
                    assert len(_l_i)==bIV,'Please adjust confs.bIV'
                    l_pose = []
                    cur_target_id = []
                    target_images = torch.empty( (bIV,256,256,3) )
                    for ii,i in enumerate(_l_i):
                        # I0gi简称i; q1 warp又称image1, 简称q1
                        elev_i,azim_i,_,_,dis0=db0.get_elev_az_many_relaToBase(str(i))
                        elev_i=torch.tensor(elev_i, device=device)
                        azim_i=torch.tensor(azim_i, device=device)
                        elev_iToq1=q1_elev_deg-elev_i
                        az_iToq1=q1_azim_deg-azim_i
                        elev_iToq1 = torch.deg2rad_(elev_iToq1)
                        az_iToq1 = torch.deg2rad_(az_iToq1)
                        rela_pose=[-elev_iToq1,az_iToq1,radius]
                        I0giRGB_tensor = db0.get_image_B(i)
                        target_images[ii] = I0giRGB_tensor
                        cur_target_id.append( db0.get_image_path(i) )
                        l_pose.append(rela_pose)
                    target_images = target_images.repeat_interleave(bsz,dim=0) #必须加dim,否则use flattened input, ret a flat output
                    cur_target_id = '|'.join(cur_target_id)
                    confs.cache_4_ldm_get_input.set_cur(image1_path_4_get_input_cache, cur_target_id, )
                    losses = get_sds_loss_requireGrad(
                        l_pose,
                        ldm_model,
                        cond_images, target_images,
                        confs.Lr.Lr_latentB_bsz, 
                        bIV=bIV,
                        noise=noise,
                    )
                    loss = losses.sum()
                    sum_loss+=loss.item()
                    loss.backward()
            optimizer.step()
            elev_az_r_deg_loss=(q1_elev_deg.item(), q1_azim_deg.item(), radius.item(),sum_loss)
            cache4Lr.set(i_iter,elev_az_r_deg_loss)
        else:
            pass
        loss_traj.append(elev_az_r_deg_loss[3])
        pose_traj.append(elev_az_r_deg_loss[:3]  )
        print(elev_az_r_deg_loss)
        if (i_iter+1)%decay_interval==0:
            print('lr decay')
            for _i in range(len(optimizer.param_groups)):
                optimizer.param_groups[_i]['lr']/=2
    # print(f"{loss_traj=}")
    # print(f"{pose_traj=}")
    inter = {
        'loss_traj': loss_traj,
        'pose_traj': pose_traj,
        'pose_traj_no_initialPose': pose_traj[1:],
    }
    # return q1_elev_deg.item(),q1_azim_deg.item(),inter
    return *(elev_az_r_deg_loss[:2]),inter

