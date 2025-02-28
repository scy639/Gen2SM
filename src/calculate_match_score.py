from debug_util import *
import math

import numpy as np

import ldm_util
import root_config
from LatentDatabase import *
import torch
from gen6d.Gen6D.pipeline import    Estimator4Co3dEval
from ldm_util import pairwise_loss,pairwise_loss_requireGrad
from ldm_util import get_sds_loss,get_sds_loss_requireGrad
import funcy as fc
from imports import *


import json
import misc_util
from pathlib import Path


class NoisePr_Database_4lBB:#简称ndb
    def __init__(self, ldb):
        folder=Path(root_config.path_data)/'ndb' /root_config.ndb_idSuffix/f"{Global.curPairOnlyQ0Str}"
        folder.mkdir(exist_ok=True,parents=True)
        self.folder=folder
        self.ldb:LatentDatabase=ldb
    def get_path(self, index, ts: float)->Path:
        assert isinstance(ts,float)
        ts_str = str(ts)
        assert len(ts_str) < 6, ts_str
        folder=self.folder   /ts_str
        folder.mkdir(exist_ok=True)
        ret = folder   / f"{index}.npy"
        return ret
    @methodtools.lru_cache()
    def get(self,index,ts:float):
        assert isinstance(ts,float)
        ret=self.get_path(index,ts)
        if not ret.exists():
            self.pr_noise_given_ldb(  (ts,) )
        ret=np.load(ret)
        ret = torch.tensor(ret, device='cuda:0')
        return ret
    @fc.print_durations()
    def pr_noise_given_ldb(
            self,ddim_steps:tuple,
    ):
        """
        limg:bsz,...
        """
        ldb: LatentDatabase=self.ldb
        lBB_cfg = root_config.lBB
        if lBB_cfg.dn_q0_bsz != lBB_cfg.latentBB_bsz:
            raise NotImplementedError#实现的话直接for,不要组batch. 只是1 step，不用多久的
        ldm_model = ModelGetters.ModelsGetter.get_B()['turncam']
        #
        base = ldb.get__zero123Input_info()
        img_path__in_ref = base['img_path__in_ref']#img0_warp
        img0_warp_tensor=imread(img_path__in_ref)
        img0_warp_tensor = ldm_util.load_image_(img0_warp_tensor)  # cond
        l_xyz = base['other_info']['l_xyz']
        for i_ts, ts in enumerate(ddim_steps):
            for i, (x, y, z) in enumerate(l_xyz):
                limg=ldb.get_latentImg(i,i_ts)
                noisePr = ldm_util.pr_noise(# latentBB_bsz,...
                    [x, y, z], ldm_model, img0_warp_tensor, limg,
                    (ts, ts + 0.01),
                    lBB_cfg.latentBB_bsz,
                    noise=None,
                )
                path = self.get_path(i,ts)
                np.save(path, noisePr.cpu())
def get_totalScore_A(i2score: list, i2weight: list):
    assert len(i2score) == len(i2weight)
    ret = 0
    for score, weight in zip(i2score, i2weight):
        ret += score * weight
    ret /= sum(i2weight)
    return ret
class Calculate_Set_Match_Score:
    @staticmethod
    def A(
        ldb0:LatentDatabase,ldb1:LatentDatabase,shift_azim,KNN:int=1,
        lossCache=None,
    ):
        """
        ldb must have been built
        actually it do not return
        """
        
        assert KNN>0
        len0=len(ldb0)
        len1=len(ldb1)
        l_score_=[]
        for j in range(len1):
            limg_j=ldb1.get_latentImg(j)
            elev_j,azim_j=ldb1.get_elev_azim(j)
            azim_j+=shift_azim
            if 1:
                indexs_after_selector_filter = []
                if 1:
                    lack = KNN
                    if lack > 0:
                        # 在self.scy_ref_poses中找min_number个离pose_4_min_number最近的pose
                        distances = []
                        for i in range(len0):
                            if 0:
                                pass
                            else:
                                elev_, azim_ = ldb0.get_elev_azim(i)
                                az_diff = (azim_j - azim_) % 360
                                if az_diff > 180:
                                    az_diff = 360 - az_diff
                                dis = (elev_j - elev_) ** 2 + az_diff ** 2
                                # dis= math.sqrt(dis)
                                # print(f"{elev_=} {azim_=} {center_elev=} {center_azim=} {dis=}")
                            distances.append(dis)
                        distances = np.array(distances)
                        indexs_after_selector_filter += distances.argsort()[:lack].tolist()  # 按照距离排序，然后取最小的 lack 个
                # print(f"{indexs_after_selector_filter=}")
                assert len(indexs_after_selector_filter) > 0
            l_score__=[]
            for i in indexs_after_selector_filter:
                limg_i=ldb0.get_latentImg(i)
                if root_config.enable____LossCache4___search__index__ddim_inter:
                    # k=(ldb0.id,ldb1.id,i,j)
                    # k=f"{ldb0.id}  {ldb1.id}  {shift_azim}  {i}  {j}"
                    k=f"{shift_azim}  {i}  {j}"
                    if lossCache is None:
                        v=None
                    else:
                        v=lossCache.dic.get(k,None)
                    if v is None:
                        _,v=get_loss(limg_i,limg_j)
                        assert v.shape == (10, 4,32,32),v.shape
                        v = v.mean([1,2,3])
                        v = v.cpu()
                        lossCache.set(k,v)
                    assert v.shape == (10, ),v.shape
                    v=v[root_config.index__ddim_inter__when_get_loss,...]
                    loss = v.mean()
                    loss=loss.item()
                    score = -loss
                else:
                    score,_ = get_loss(limg_i, limg_j)
                    score = -score
                l_score__.append(score)
            l_score_.append(max(l_score__)   )
        i2weight = [1] * len(l_score_)
        totalScore = get_totalScore_A(l_score_, i2weight)
        return totalScore
    @staticmethod
    @fc.print_durations( threshold=0.01)
    def B(estimator0: Estimator4Co3dEval, image1, q1_elev_deg, shift_azim,
          noise, ldm_model,
          l_score_from_cache4lB=None,
          #
          image1_path_4_get_input_cache=None,
          ldb0:LatentDatabase=None,
          ndb0:NoisePr_Database_4lBB=None,
          cache4lB_B=None,
          ):
        """
        ldb must have been built
        """
        if root_config.cache_4_ldm_get_input.enable:
            assert image1_path_4_get_input_cache
        if   l_score_from_cache4lB:
            l_score_=l_score_from_cache4lB[:]
        else:
            if root_config.lBB.enable:
                db0: LatentDatabase =ldb0
                len0=len(ldb0)
            else:
                db0 = estimator0.sparseViewPoseEstimator.ref_database
                len0 = len(db0.get_img_ids()  )
            l_score_ = []
            i2weight = []
            l_i=[]
            # l_i+=[db0.ID_zero123InputImage]
            l_i+=range(0,len0,root_config.CalScoreB__db0_iterate_interval )
            if root_config.BATCH_multiple_iv:
                bIV = root_config.bIV
                bsz = root_config.latentB_bsz
                if 1:
                    cond_images = image1.repeat(bsz*bIV, 1, 1, 1)
                l_l_i = [l_i[i:i + bIV] for i in range(0, len(l_i), bIV)]
                del l_i
                if root_config.lBB.enable:
                    raise NotImplementedError
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
                    root_config.cache_4_ldm_get_input.set_cur(image1_path_4_get_input_cache, cur_target_id, )
                    with torch.no_grad():
                        losses = get_sds_loss(
                            l_pose,
                            ldm_model,
                            cond_images, target_images,
                            root_config.latentB_bsz, bIV=bIV,
                            noise=noise,
                        )
                    for ii,i in enumerate(_l_i):
                        loss = losses[ii]
                        loss=loss.item()
                        if 1:
                            elev_i,azim_i,elev_rToi,az_rToi,_=db0.get_elev_az_many_relaToBase_B(str(i))
                            #shift_azim: az_w0Tow1 ~= az_rToq
                            cache4lB_B.set( q_elev=q1_elev_deg,az_w0Tow1=shift_azim ,
                                            elev_i=elev_i,elev_ri=elev_rToi,az_ri=az_rToi ,
                                            elev_qi=elev_iToq1,az_qi=-az_iToq1,
                                            loss=loss  )
                        score=-loss
                        l_score_.append(score)
                        weight=0#TODO dev ing
                        i2weight.append(weight)
                root_config.cache_4_ldm_get_input.I_am_sure_that_cache_has_built_over=True
            else:
                for i in l_i:
                    # I0gi简称i; q1 warp又称image1, 简称q1
                    # I0giRGB=db0.get_image(i)
                    elev_i,azim_i,_,_,dis0=db0.get_elev_az_many_relaToBase(str(i))
                    elev_iToq1=q1_elev_deg-elev_i
                    if 1:
                        q0_elev,q0_az=pose_util.R_or_pose__2__elev_azim_inDeg_A( np.array(db0.get_zero123Input_info()['pose'] ) )
                        assert math.isclose(q0_az,0),q0_az
                    az_iToq1=shift_azim-azim_i
                    elev_iToq1=np.deg2rad(elev_iToq1)
                    az_iToq1=np.deg2rad(az_iToq1)
                    lBB_cfg=root_config.lBB
                    if lBB_cfg.enable:
                        if i==4:
                            print(4)
                        i_ts_2_loss=[]
                        for i_ts in range(len(lBB_cfg.ddim_steps  )):
                            latentImg_i = ldb0.get_latentImg(i,i_ts )
                            q0_noisePr = ndb0.get(i, lBB_cfg.q0_ddpm_steps[i_ts]  )
                            if 1:
                                latentImg_i=latentImg_i.clone()
                                q0_noisePr=q0_noisePr.clone()
                            root_config.cache_4_ldm_get_input.set_cur(f"{db0.id} {i} {i_ts}", image1_path_4_get_input_cache)
                            with torch.no_grad():
                                loss = pairwise_loss(
                                    [-elev_iToq1, az_iToq1, 0],
                                    ldm_model,
                                    latentImg_i, image1,
                                    ( lBB_cfg.q1_ddpm_steps[i_ts], lBB_cfg.q1_ddpm_steps[i_ts] +0.01,), lBB_cfg.latentBB_bsz,
                                    noise='639',
                                    q0_noisePr=q0_noisePr,
                                    scy_target_is_z_insteadof_rgb=True,
                                )
                            i_ts_2_loss.append(loss)
                        i_ts_2_loss=[loss*weight for loss,weight in zip(i_ts_2_loss,lBB_cfg.i_ts_2_weight)]
                        loss=sum(i_ts_2_loss)
                        loss=loss/sum(lBB_cfg.i_ts_2_weight)
                    else:
                        # I0giRGB_tensor=ldm_util.load_image_(I0giRGB)
                        I0giRGB_tensor = db0.get_image_B(i)
                        root_config.cache_4_ldm_get_input.set_cur(db0.get_image_path(i), image1_path_4_get_input_cache)
                        with torch.no_grad():
                            loss = pairwise_loss(
                                [-elev_iToq1, az_iToq1, 0],
                                ldm_model,
                                I0giRGB_tensor, image1,
                                root_config.CalScoreB_ts_range, root_config.latentB_bsz,
                                noise=noise,
                            )
                    loss=loss.item()
                    if 1:
                        elev_i,azim_i,elev_rToi,az_rToi,_=db0.get_elev_az_many_relaToBase_B(str(i))
                        #shift_azim: az_w0Tow1 ~= az_rToq
                        cache4lB_B.set( q_elev=q1_elev_deg,az_w0Tow1=shift_azim ,
                                    elev_i=elev_i,elev_ri=elev_rToi,az_ri=az_rToi ,
                                    elev_qi=elev_iToq1,az_qi=-az_iToq1,
                                    loss=loss  )
                    score=-loss
                    l_score_.append(score)
                    weight=0#TODO dev ing
                    i2weight.append(weight)
                root_config.cache_4_ldm_get_input.I_am_sure_that_cache_has_built_over=True
        if root_config.set_match_weight_strtegy=='A':
            i2weight = [1] * len(l_score_)
        else:
            pass
        # print(f"{l_score_=}")
        totalScore = get_totalScore_A(l_score_, i2weight)
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
        num_iter=None,#不要写=root_config.xxx这又是那个坑，karg默认值会在一开始就被计算
        decay_interval=None,
):
    if root_config.cache_4_ldm_get_input.enable:
        assert image1_path_4_get_input_cache
    if 1:
        ModelGetters.ModelsGetter.set_param()
        ldm_model = ModelGetters.ModelsGetter.get_B()
        ldm_model = ldm_model['turncam']
    if noise is None:
        noise = np.random.randn(root_config.Lr.Lr_latentB_bsz, 4, 32, 32)
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
                'lr':root_config.Lr.elev_az_learning_rate,
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
                if root_config.Lr.db0_iterate_strategy.roll:
                    _start_i=i_iter % root_config.Lr.db0_iterate_strategy.interval
            l_i = range(_start_i, len0, root_config.Lr.db0_iterate_strategy.interval)
            if root_config.BATCH_multiple_iv:
                bIV = root_config.bIV
                bsz = root_config.latentB_bsz
                if 1:
                    cond_images = img.repeat(bsz*bIV, 1, 1, 1)
                l_l_i = [l_i[i:i + bIV] for i in range(0, len(l_i), bIV)]
                del l_i
                for _l_i in l_l_i:
                    l_pose = []
                    cur_target_id = []
                    target_images = torch.empty( (bIV,256,256,3) )
                    for ii,i in enumerate(_l_i):
                        # I0gi简称i; q1 warp又称image1, 简称q1
                        elev_i,azim_i,_,_,dis0=db0.get_elev_az_many_relaToBase(str(i))
                        elev_i=torch.tensor(elev_i, device=device)
                        azim_i=torch.tensor(azim_i, device=device)
                        elev_iToq1=q1_elev_deg-elev_i
                        if 0:
                            q0_elev,q0_az=pose_util.R_or_pose__2__elev_azim_inDeg_A( np.array(db0.get_zero123Input_info()['pose'] ) )
                            assert math.isclose(q0_az,0),q0_az
                        az_iToq1=q1_azim_deg-azim_i
                        elev_iToq1 = torch.deg2rad_(elev_iToq1)
                        az_iToq1 = torch.deg2rad_(az_iToq1)
                        rela_pose=[-elev_iToq1,az_iToq1,radius]
                        I0giRGB_tensor = db0.get_image_B(i) # TODO 优化。
                        target_images[ii] = I0giRGB_tensor
                        cur_target_id.append( db0.get_image_path(i) )
                        l_pose.append(rela_pose)
                    target_images = target_images.repeat_interleave(bsz,dim=0) #必须加dim,否则use flattened input, ret a flat output
                    cur_target_id = '|'.join(cur_target_id)
                    root_config.cache_4_ldm_get_input.set_cur(image1_path_4_get_input_cache, cur_target_id, )
                    losses = get_sds_loss_requireGrad(
                        l_pose,
                        ldm_model,
                        cond_images, target_images,
                        root_config.Lr.Lr_latentB_bsz, 
                        bIV=bIV,
                        noise=noise,
                    )
                    loss = losses.sum()
                    sum_loss+=loss.item()
                    loss.backward()
            else:
                for i in l_i:
                    # I0gi简称i; q1 warp又称image1, 简称q1
                    # I0giRGB = db0.get_image(i)
                    elev_i, azim_i, _, _, _ = db0.get_elev_az_many_relaToBase(str(i))
                    elev_i=torch.tensor(elev_i, device=device)
                    azim_i=torch.tensor(azim_i, device=device)
                    elev_iToq1 = q1_elev_deg - elev_i
                    if 0:
                        q0_elev, q0_az = pose_util.R_or_pose__2__elev_azim_inDeg_A(np.array(db0.get_zero123Input_info()['pose']))
                        assert math.isclose(q0_az, 0), q0_az
                    az_iToq1 = q1_azim_deg - azim_i
                    elev_iToq1 = torch.deg2rad_(elev_iToq1)
                    az_iToq1 = torch.deg2rad_(az_iToq1)
                    rela_pose=[-elev_iToq1,az_iToq1,radius]
                    # I0giRGB_tensor = ldm_util.load_image_(I0giRGB)
                    I0giRGB_tensor=db0.get_image_B(i)
                    root_config.cache_4_ldm_get_input.set_cur(db0.get_image_path(i), image1_path_4_get_input_cache)
                    loss = pairwise_loss_requireGrad(rela_pose,
                                        ldm_model,
                                        I0giRGB_tensor, img,
                                        root_config.Lr.ts_range, root_config.Lr.Lr_latentB_bsz,
                                        noise=noise)
                    sum_loss+=loss.item()
                    loss.backward()
                # root_config.cache_4_ldm_get_input.I_am_sure_that_cache_has_built_over=True
                # sum_loss.backward()
            if 0:
                for i in param_list:
                    print(f"{i.grad=}")
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

