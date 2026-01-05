"""
term:
Ls: latent search, ie the search stage
Lr: latent refine, ie the refine stage
"""
import methodtools
import numpy as np

import elev_util
import image_util
import ldm_util
import misc_util
import pose_util
import confs
from gen6d.Gen6D.pipeline import run4gen6d_main
from infer_pair import *
import calculate_match_score
from miscellaneous import ModelGetters
from ldm_util import *

class Cache4Ls: # Ls: latent search, ie the search stage
    @staticmethod
    def get_file_name_C(elev_deg1:float):  # _C: version C
        return f"{Global.curPairStr}-elev1={round(elev_deg1,2)}-{confs.cache4Ls_idSuffix}.json"
    def __init__(self,file_name):
        self.path=Path(confs.path_data)/'Cache4lB'/file_name
        self.load()
    def set(self,shift_az,l_score):
        self.shiftAzim_2_l_score[shift_az]=l_score
    def load(self,):
        path=self.path
        if path.exists():
            with open(path, "r") as f:
                dic = json.load(f)
            dic=misc_util.dic_key_str_2_int(dic)
        else:
            dic={}
        self.shiftAzim_2_l_score=dic
    def dump(self,):
        path=self.path
        path.parent.mkdir(exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.shiftAzim_2_l_score, f)
class Cache4Lr: # Lr: latent refine, ie the refine stage
    @staticmethod
    def get_file_name_A(elev_deg1_before_Lr:float,az_deg_before_Lr:float):
        return f"{Global.curPairStr}-{round(elev_deg1_before_Lr,2)},{round(az_deg_before_Lr,2)}-{confs.cache4Lr_idSuffix}.json"
    def __init__(self,file_name):
        self.path=Path(confs.path_data)/'Cache4Lr'/file_name
        self.load()
    def set(self,i_iter,elev_az_r_deg_loss):
        assert len(elev_az_r_deg_loss)==4
        assert i_iter not in self.i_iter_2_elev_az_r_deg_loss
        self.i_iter_2_elev_az_r_deg_loss[i_iter]=elev_az_r_deg_loss
    def load(self,):
        path=self.path
        if path.exists():
            with open(path, "r") as f:
                dic = json.load(f)
            Lr_cfg=dic['Lr_cfg']
            Lr_cfg=misc_util.dic_list_2_tuple_nested(Lr_cfg)
            i_iter_2_elev_az_r_deg_loss=dic['i_iter_2_elev_az_r_deg_loss']
            i_iter_2_elev_az_r_deg_loss=misc_util.dic_key_str_2_int__nested(i_iter_2_elev_az_r_deg_loss)
            assert Lr_cfg==confs.Lr.get_fields_in_dic()
        else:
            i_iter_2_elev_az_r_deg_loss={}
        self.i_iter_2_elev_az_r_deg_loss=i_iter_2_elev_az_r_deg_loss
    def dump(self,):
        path=self.path
        path.parent.mkdir(exist_ok=True)
        dic = dict(
            Lr_cfg=confs.Lr.get_fields_in_dic(),
            i_iter_2_elev_az_r_deg_loss=self.i_iter_2_elev_az_r_deg_loss,
        )
        with open(path, "w") as f:
            json.dump(dic, f)


def update__shiftAzim_2_totalScore(
        l_shift_azim,
        shiftAzim_2_totalScore,#will be updated
        #
        cache4Ls: Cache4Ls,#will be updated
        estimator0,
        img1_warp_tensor,
        elev_deg1,
        noise,
        ldm_model,
        image1_path_4_get_input_cache,
):
    print(f"{l_shift_azim=}")
    for shift_azim in l_shift_azim:
        shift_azim = shift_azim % 360
        assert isinstance(shift_azim, int)
        if shift_azim in shiftAzim_2_totalScore:
            continue
        totalScore, l_score = calculate_match_score.calculate_Set_Matching_Score(
            estimator0,
            img1_warp_tensor,
            elev_deg1,
            shift_azim,
            noise,
            ldm_model,

            cache4Ls.shiftAzim_2_l_score.get(shift_azim, None),
            # None,

            image1_path_4_get_input_cache=image1_path_4_get_input_cache,
        )
        print(f"{shift_azim=} {totalScore=}")
        assert totalScore > -1e5, totalScore  # !=-inf
        cache4Ls.set(shift_azim, l_score)
        shiftAzim_2_totalScore[shift_azim] = totalScore
    best_shiftAzim = sorted(shiftAzim_2_totalScore, key=lambda k: shiftAzim_2_totalScore[k], reverse=True)[0]
    return best_shiftAzim

@fc.print_durations()
def infer_in_latent_space(# search stage+refine stage
        estimator0,
        estimator1,# just for get ...
        id0: str,
        id1: str,
        K0,
        K1,
        image0_path,
        image1_path,
        bbox0: list,  # x1,y1,x2,y2
        bbox1: list,
        #
        R_GT_opencv=None,
        elev_deg0=None,
        elev_deg1=None,
):
    # -------------step 1 cal mat
    #
    img0_warp_path = get_path_after_warp(image0_path)
    img0_warp, pose0_rect, K0_warp = look_at_wrapper_wrapper(image0_path, bbox0, K0, save_path=img0_warp_path)
    if 'GT_elev0' in Global.anything: # if GT elev of img 0 is provided
        confs.force_elev = elev_deg0
    else:
        confs.force_elev = None
    pose_w0Tor = rq____path__2__pose_wjTorq(image0_path, estimator0, bbox0, K0,)#(w0: after rectify ipr
    confs.force_elev = None  # recover
    #
    img1_warp_path = get_path_after_warp(image1_path)
    img1_warp, pose1_rect, K1_warp = look_at_wrapper_wrapper(image1_path, bbox1, K1, save_path=img1_warp_path)
    if 1:
        noise = np.random.randn(confs.Ls_bsz, 4, 32, 32)
    if 1:
        assert 'GT_elev1' not in Global.anything
        # m1  TIP method ( PR elev method1, '_m1'
        R, t,relative_pose,inter= gen6d_imgPaths2relativeRt_B(
            estimator=estimator0,
            K0=K0,
            K1=K1,
            image0_path=image0_path,
            image1_path=image1_path,
            bbox0=bbox0,
            bbox1=bbox1,
        )
        if confs.CONSIDER_IPR or confs.CONSIDER_Q1_IPR:
            degree_counterClockwise1=Global.degree_counterClockwise1_channel.get()
            img1_warp = image_util.rotate_C(img1_warp, [degree_counterClockwise1], )[0]
            img1_warp_path = f"{img1_warp_path}.rot{round(degree_counterClockwise1)}.png"
            imsave(img1_warp_path,img1_warp)
            print(f"{img1_warp_path=}")
        else:
            degree_counterClockwise1=0
        elevDeg_m1 = pose_util.R_or_pose__2__elev_azim_inDeg_B(inter['pose1'])[0]
        Global.pairs_result.set_cur_dic('elevDeg_m1', elevDeg_m1)
        if confs.DirectlyUseE2VGresult_if_gen6d_selector_max_score_larger_than.enable:
            if Global.pairs_result.get_cur_dic( 'selector-max-score'  )>\
            confs.DirectlyUseE2VGresult_if_gen6d_selector_max_score_larger_than.thres:
                return R, t,relative_pose,inter
        del R, t,relative_pose,inter
        elev_deg1=elevDeg_m1
        confs.force_elev = elevDeg_m1
    """
    else:
        if 'GT_elev1' in Global.anything:
            confs.force_elev = elev_deg1
        else:
            confs.force_elev = None
    pose_w1Toq = rq____path__2__pose_wjTorq(image1_path, estimator1, bbox1, K1, )
    confs.force_elev = None#recover
    """
    img1_warp_tensor=load_image_(img1_warp)
    debug_imsave(f'load_image_/{os.path.basename(img1_warp_path)  }',img1_warp_tensor[0])
    if 1:
        ModelGetters.ModelsGetter.set_param()
        ldm_model=ModelGetters.ModelsGetter.get_B()
        ldm_model=ldm_model['turncam']
        #
        noise = torch.tensor(noise, dtype=ldm_model.dtype, device=ldm_model.device)
    else:#fast debug
        ldm_model=None
    if 1:#ssD
        i_elev_2_elev=[elev_deg1-30,elev_deg1,elev_deg1+30]
        i_elev_2_elev=list(   filter(   lambda x: -5<x<88,  i_elev_2_elev)    )
        i_elev_2_best_shiftAzim=[]
        i_elev_2_socre=[]
        for elev_ in i_elev_2_elev:
            cache4Ls = Cache4Ls(Cache4Ls.get_file_name_C(elev_)  )
            shiftAzim_2_totalScore = {}
            best_shiftAzim= update__shiftAzim_2_totalScore(
                range(0, 360, 45),
                shiftAzim_2_totalScore,  # will be updated
                #
                cache4Ls,  # will be updated
                estimator0,
                img1_warp_tensor,
                elev_,
                noise,
                ldm_model,
                img1_warp_path,
            )
            cache4Ls.dump()
            score=shiftAzim_2_totalScore[best_shiftAzim]
            i_elev_2_best_shiftAzim.append(best_shiftAzim)
            i_elev_2_socre.append(score)
        # best_shiftAzim=i_elev_2_best_shiftAzim[ i_elev_2_socre.index(  max(i_elev_2_socre)  )   ]
        elev_deg1 = i_elev_2_elev[i_elev_2_socre.index(max(i_elev_2_socre))]
        print(f"{i_elev_2_elev=}")
        print(f"{i_elev_2_best_shiftAzim=}")
        print(f"{i_elev_2_socre=}")
        print(f"{elev_deg1=}")
    fn_cache4lB =Cache4Ls.get_file_name_C(elev_deg1)  
    cache4Ls = Cache4Ls(fn_cache4lB)
    if 1:
        shiftAzim_2_totalScore = {}
        def wrap(l_shift_azim):
            return update__shiftAzim_2_totalScore(
                l_shift_azim,
                shiftAzim_2_totalScore,  # will be updated
                #
                cache4Ls,  # will be updated
                estimator0,
                img1_warp_tensor,
                elev_deg1,
                noise,
                ldm_model,
                img1_warp_path,
            )
        best_shiftAzim= wrap(range(0, 360, 45))
        best_shiftAzim= wrap(range(best_shiftAzim-45, best_shiftAzim+45, 15))
        best_shiftAzim= wrap(range(best_shiftAzim-15, best_shiftAzim+15, 5))


        # print(f"{shiftAzim_2_totalScore=}")
        print(f"{best_shiftAzim=}")
        cache4Ls.dump( )
    _l_elev = [elev_deg1]
    _l_score = [shiftAzim_2_totalScore[best_shiftAzim]]
    del shiftAzim_2_totalScore
    del cache4Ls
    if 1:
        def wrap2(elev_deg1_):
            shiftAzim_2_totalScore={}
            cache4Ls = Cache4Ls(Cache4Ls.get_file_name_C(elev_deg1_)  )
            _=update__shiftAzim_2_totalScore(
                (best_shiftAzim,),
                shiftAzim_2_totalScore,  # will be updated
                #
                cache4Ls,  # will be updated
                estimator0,
                img1_warp_tensor,
                elev_deg1_,
                noise,
                ldm_model,
                img1_warp_path,
            )
            cache4Ls.dump()
            assert len(shiftAzim_2_totalScore)==1
            return shiftAzim_2_totalScore[best_shiftAzim]
        if 0:
            if elev_deg1>8:
                elev=elev_deg1-10
                _l_elev.append(elev)
                _l_score.append(wrap2(elev))
            if elev_deg1<78:
                elev = elev_deg1 + 10
                _l_elev.append(elev)
                _l_score.append(wrap2(elev))
        else:#ssD2: ssD but when finally search elev interval is 15 instead of 10
            if elev_deg1>-3+15:
                elev=elev_deg1-15
                _l_elev.append(elev)
                _l_score.append(wrap2(elev))
            if elev_deg1<88-15:
                elev = elev_deg1 + 15
                _l_elev.append(elev)
                _l_score.append(wrap2(elev))
        print(f"{_l_elev=}")
        print(f"{_l_score=}")
        best_elev_deg1=sorted(zip(_l_score,_l_elev),key=lambda x:x[0],reverse=True)[0][1]
    del noise
    if confs.enable_Lr:#Lr
        if confs.cache_4_ldm_get_input.enable:
            confs.cache_4_ldm_get_input.clear()
        cache4Lr=Cache4Lr(Cache4Lr.get_file_name_A(best_elev_deg1,best_shiftAzim)  )
        best_elev_deg1,best_shiftAzim,inter_Lr= calculate_match_score.refine(
            img1_warp, (best_elev_deg1,best_shiftAzim),
            estimator0,
            cache4Lr,
            image1_path_4_get_input_cache=img1_warp_path,
            noise=None,
            num_iter=confs.Lr.num_iter,
            decay_interval=confs.Lr.decay_interval,
        )
        cache4Lr.dump()
    pose0 = pose_w0Tor
    def elev1_az_2_poses(elev_deg,az_deg):
        rad = np.deg2rad(az_deg)
        pose_w0Tow1 = np.array(# z轴朝纸外（指向你的眼睛），w0的x轴 逆时针 旋转rad得到w1的x轴
            [
                [math.cos(rad), math.sin(rad), 0, 0],
                [-math.sin(rad), math.cos(rad), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        pose_rTow1 = pose_w0Tow1 @ np.linalg.inv(pose_w0Tor)
        pose_w1Toq,_,_=wq_elev_deg__2__pose_wTorq(elev_deg,image1_path,K1,bbox1,degree_counterClockwise1)
        pose_rToq = pose_w1Toq @ pose_rTow1
        relative_pose = pose_rToq
        R = relative_pose[:3, :3]
        t = relative_pose[:3, 3:]
        pose1 = relative_pose @ pose0
        return pose1,relative_pose,R,t
    if 1:
        pose1,relative_pose,R,t = elev1_az_2_poses(best_elev_deg1,best_shiftAzim,)
        if confs.enable_Lr:
            for i,(elev_deg,az_deg,_) in enumerate(  inter_Lr['pose_traj']):
                _,_,tmp_R,_ = elev1_az_2_poses(elev_deg,az_deg,)
                Rerr_interLr = compute_angular_error(R_GT_opencv, tmp_R )
                print(f"{Rerr_interLr=}")
                Global.pairs_result.set_cur_dic(f'Lr{i}', Rerr_interLr)
            err_diff=Global.pairs_result.get_cur_dic(f'Lr{len(inter_Lr["pose_traj"])-1}',  )-Global.pairs_result.get_cur_dic(f'Lr{0}',  )
            Global.pairs_result.set_cur_dic(f'err_diff', err_diff)
            err_diff_round=err_diff
            if abs(err_diff_round)<1.5:
                err_diff_round=0
            Global.pairs_result.set_cur_dic(f'err_diff_round', err_diff_round)
        # ------4 vis-------
        # Global.poseVisualizer0.append(pose0,color="grey")
        # Global.poseVisualizer0.append(pose1,color="blue")
        inter = dict(
            pose0=pose0,
            pose1=pose1,
        )
    if confs.cache_4_ldm_get_input.enable:
        confs.cache_4_ldm_get_input.clear()
    return R, t, relative_pose, inter
