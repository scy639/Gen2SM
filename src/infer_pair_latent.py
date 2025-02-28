import methodtools
import numpy as np

import elev_util
import image_util
import ldm_util
import misc_util
import pose_util
import root_config
from gen6d.Gen6D.pipeline import run4gen6d_main
from infer_pair import *
from miscellaneous import ModelGetters
from ldm_util import *
"""
def build_ldb(
        id_: str,
        K,
        image_path,
        bbox: list,  # x1,y1,x2,y2
        #
        elev_deg=None,
):
    img_warp_path = get_path_after_warp(image_path)
    img_warp, pose_rect, K_warp = look_at_wrapper_wrapper(image_path, bbox, K, save_path=img_warp_path)
    ldb = LatentDatabase(id_)
    ldb.build_A(img_warp_path, K_warp, np.deg2rad(elev_deg))
    return ldb,  img_warp, pose_rect, K_warp
ldb0,img_warp0, pose_rect0, K_warp0 = build_ldb(id0, K0, image0_path, bbox0, elev_deg=elev_deg0)
ldb1,img_warp1, pose_rect1, K_warp1 = build_ldb(id1, K1, image1_path, bbox1, elev_deg=elev_deg1)
"""

class Cache4lB:
    @staticmethod
    def get_file_name_C(elev_deg1:float):
        return f"{Global.curPairStr}-elev1={round(elev_deg1,2)}-{root_config.cache4lB_idSuffix}.json"
    def __init__(self,file_name):
        self.path=Path(root_config.path_data)/'Cache4lB'/file_name
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
class Cache4Lr:
    @staticmethod
    def get_file_name_A(elev_deg1_before_Lr:float,az_deg_before_Lr:float):
        return f"{Global.curPairStr}-{round(elev_deg1_before_Lr,2)},{round(az_deg_before_Lr,2)}-{root_config.cache4Lr_idSuffix}.json"
    def __init__(self,file_name):
        self.path=Path(root_config.path_data)/'Cache4Lr'/file_name
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
            assert Lr_cfg==root_config.Lr.get_fields_in_dic()
        else:
            i_iter_2_elev_az_r_deg_loss={}
        self.i_iter_2_elev_az_r_deg_loss=i_iter_2_elev_az_r_deg_loss
    def dump(self,):
        path=self.path
        path.parent.mkdir(exist_ok=True)
        dic = dict(
            Lr_cfg=root_config.Lr.get_fields_in_dic(),
            i_iter_2_elev_az_r_deg_loss=self.i_iter_2_elev_az_r_deg_loss,
        )
        with open(path, "w") as f:
            json.dump(dic, f)

class Cache4lB_B:#Cache4lB的B亚型. actually, it is for record, not for cache
    @staticmethod
    def get_file_name_C(elev_deg1:float):
        return f"{Global.curPairStr}-elev1={round(elev_deg1,2)}-{root_config.cache4lB_idSuffix}.xlsx"
    def __init__(self,file_name):
        self.path=Path(root_config.path_data)/'Cache4lB_B'/file_name
        # self.load()
        # self.dfIndex0=[]
        # self.dfIndex1=[]
        self.l_dic=[]
    def set(self,   **kw   ):
        # self.shiftAzim_2_l_score[shift_az]=l_score
        self.l_dic.append(  kw   )
    # def load(self,):
    #     path=self.path
    #     if path.exists():
    #         with open(path, "r") as f:
    #             dic = json.load(f)
    #         dic=misc_util.dic_key_str_2_int(dic)
    #     else:
    #         dic={}
    #     self.shiftAzim_2_l_score=dic
    def dump(self,):
        path=self.path
        path.parent.mkdir(exist_ok=True)
        with open(path, "w") as f:
            # json.dump(self.shiftAzim_2_l_score, f)
            # df = pd.DataFrame(l_dic_withAvg, index=[dfIndex0, dfIndex1])
            df = pd.DataFrame(self.l_dic, 
                            #   index=[ dfIndex0,]
                              )
            df.to_excel(  path  )
            print("excel:", path,  )
        if 0:
            assert path.count('.xlsx')==1
            assert path.endswith('.xlsx')
            json_path=path.replace('.xlsx','.json')
            with open(path, "w") as f:
                json.dump(meta, f)

def update__shiftAzim_2_totalScore(
        l_shift_azim,
        shiftAzim_2_totalScore,#will be updated
        #
        cache4lB: Cache4lB,#will be updated
        estimator0,
        img1_warp_tensor,
        elev_deg1,
        noise,
        ldm_model,
        image1_path_4_get_input_cache,
        ldb0=None,
        ndb0=None,
        cache4lB_B:Cache4lB_B=None,
):
    print(f"{l_shift_azim=}")
    for shift_azim in l_shift_azim:
        shift_azim = shift_azim % 360
        assert isinstance(shift_azim, int)
        if shift_azim in shiftAzim_2_totalScore:
            continue
        totalScore, l_score = Calculate_Set_Match_Score.B(
            estimator0,
            img1_warp_tensor,
            elev_deg1,
            shift_azim,
            noise,
            ldm_model,

            cache4lB.shiftAzim_2_l_score.get(shift_azim, None),
            # None,

            image1_path_4_get_input_cache=image1_path_4_get_input_cache,
            ldb0=ldb0,
            ndb0=ndb0,
            cache4lB_B=cache4lB_B,
        )
        print(f"{shift_azim=} {totalScore=}")
        assert totalScore > -1e5, totalScore  # !=-inf
        cache4lB.set(shift_azim, l_score)
        shiftAzim_2_totalScore[shift_azim] = totalScore
    best_shiftAzim = sorted(shiftAzim_2_totalScore, key=lambda k: shiftAzim_2_totalScore[k], reverse=True)[0]
    return best_shiftAzim

@fc.print_durations()
def infer_in_latent_space_B(#简称lB
        estimator0:Estimator4Co3dEval,
        estimator1:Estimator4Co3dEval,# just for get ...
        id0: str,
        id1: str,
        ldbId0: str,
        ldbId1: str,
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
    if root_config.selector_cache_A.enable:
        assert root_config.selector_cache_A.cache == {}
    # -------------step 1 cal mat
    #
    img0_warp_path = get_path_after_warp(image0_path)
    img0_warp, pose0_rect, K0_warp = look_at_wrapper_wrapper(image0_path, bbox0, K0, save_path=img0_warp_path)
    if 'GT_elev0' in Global.anything:
        root_config.Cheat.force_elev = elev_deg0
    else:
        root_config.Cheat.force_elev = None
    pose_w0Tor = SetMatcher_NaiveA.rq____path__2__pose_wjTorq(image0_path, estimator0, bbox0, K0,)#(w0: after rectify ipr
    root_config.Cheat.force_elev = None  # recover
    #
    img1_warp_path = get_path_after_warp(image1_path)
    img1_warp, pose1_rect, K1_warp = look_at_wrapper_wrapper(image1_path, bbox1, K1, save_path=img1_warp_path)
    if root_config.lBB.enable:
        ldb0 = LatentDatabase(ldbId0)
        ldb0.build_A(img0_warp_path, K0,q_img_eleRadian=estimator0.get__zero123Input_info(K0,img0_warp_path)['elevRadian'] )
        ndb0=calculate_match_score.NoisePr_Database_4lBB(ldb0)
        noise = np.random.randn(root_config.lBB.latentBB_bsz, 4, 32, 32)
    else:
        ldb0=None
        ndb0=None
        noise = np.random.randn(root_config.latentB_bsz, 4, 32, 32)
    if 'infer_in_latent_space_B__m1' in Global.anything:
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
        if root_config.CONSIDER_IPR or root_config.CONSIDER_Q1_IPR:
            degree_counterClockwise1=Global.degree_counterClockwise1_channel.get()
            img1_warp = image_util.rotate_C(img1_warp, [degree_counterClockwise1], )[0]
            img1_warp_path = f"{img1_warp_path}.rot{round(degree_counterClockwise1)}.png"
            imsave(img1_warp_path,img1_warp)
            print(f"{img1_warp_path=}")
        else:
            degree_counterClockwise1=0
        elevDeg_m1 = pose_util.R_or_pose__2__elev_azim_inDeg_B(inter['pose1'])[0]
        Global.pairs_result.set_cur_dic('elevDeg_m1', elevDeg_m1)
        if root_config.DirectlyUseTipResult_if_selector_max_score_larger_than.enable:
            if Global.pairs_result.get_cur_dic( 'selector-max-score'  )>\
            root_config.DirectlyUseTipResult_if_selector_max_score_larger_than.thres:
                return R, t,relative_pose,inter
        del R, t,relative_pose,inter
        elev_deg1=elevDeg_m1
        root_config.Cheat.force_elev = elevDeg_m1
    """
    else:
        if 'GT_elev1' in Global.anything:
            root_config.Cheat.force_elev = elev_deg1
        else:
            root_config.Cheat.force_elev = None
    pose_w1Toq = SetMatcher_NaiveA.rq____path__2__pose_wjTorq(image1_path, estimator1, bbox1, K1, )
    root_config.Cheat.force_elev = None#recover
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
            cache4lB = Cache4lB(Cache4lB.get_file_name_C(elev_)  )
            cache4lB_B = Cache4lB_B(Cache4lB_B.get_file_name_C(elev_)  )
            shiftAzim_2_totalScore = {}
            best_shiftAzim= update__shiftAzim_2_totalScore(
                range(0, 360, 45),
                shiftAzim_2_totalScore,  # will be updated
                #
                cache4lB,  # will be updated
                estimator0,
                img1_warp_tensor,
                elev_,
                noise,
                ldm_model,
                img1_warp_path,
                ldb0,
                ndb0,
                cache4lB_B,
            )
            cache4lB.dump()
            cache4lB_B.dump()
            score=shiftAzim_2_totalScore[best_shiftAzim]
            i_elev_2_best_shiftAzim.append(best_shiftAzim)
            i_elev_2_socre.append(score)
        # best_shiftAzim=i_elev_2_best_shiftAzim[ i_elev_2_socre.index(  max(i_elev_2_socre)  )   ]
        elev_deg1 = i_elev_2_elev[i_elev_2_socre.index(max(i_elev_2_socre))]
        print(f"{i_elev_2_elev=}")
        print(f"{i_elev_2_best_shiftAzim=}")
        print(f"{i_elev_2_socre=}")
        print(f"{elev_deg1=}")
    if 0:
        if 0:
            fn_cache4lB=f"{id0}--{id1}.json"
        else:
            fn_cache4lB= f"{root_config.idSuffix}--{id0}--{id1}.json"
    else:
        fn_cache4lB =Cache4lB.get_file_name_C(elev_deg1)  
    cache4lB = Cache4lB(fn_cache4lB)
    cache4lB_B = Cache4lB_B(   Cache4lB_B.get_file_name_C(elev_deg1)     )
    if 1:
        shiftAzim_2_totalScore = {}
        def wrap(l_shift_azim):
            return update__shiftAzim_2_totalScore(
                l_shift_azim,
                shiftAzim_2_totalScore,  # will be updated
                #
                cache4lB,  # will be updated
                estimator0,
                img1_warp_tensor,
                elev_deg1,
                noise,
                ldm_model,
                img1_warp_path,
                ldb0,
                ndb0,
                cache4lB_B,
            )
        best_shiftAzim= wrap(range(0, 360, 45))
        best_shiftAzim= wrap(range(best_shiftAzim-45, best_shiftAzim+45, 15))
        best_shiftAzim= wrap(range(best_shiftAzim-15, best_shiftAzim+15, 5))


        # print(f"{shiftAzim_2_totalScore=}")
        print(f"{best_shiftAzim=}")
        cache4lB.dump( )
        cache4lB_B.dump( )
    _l_elev = [elev_deg1]
    _l_score = [shiftAzim_2_totalScore[best_shiftAzim]]
    del shiftAzim_2_totalScore
    del cache4lB
    del cache4lB_B
    if 1:
        def wrap2(elev_deg1_):
            shiftAzim_2_totalScore={}
            cache4lB = Cache4lB(Cache4lB.get_file_name_C(elev_deg1_)  )
            cache4lB_B= Cache4lB_B(Cache4lB_B.get_file_name_C(elev_deg1_)  )
            _=update__shiftAzim_2_totalScore(
                (best_shiftAzim,),
                shiftAzim_2_totalScore,  # will be updated
                #
                cache4lB,  # will be updated
                estimator0,
                img1_warp_tensor,
                elev_deg1_,
                noise,
                ldm_model,
                img1_warp_path,
                ldb0,
                ndb0,
                cache4lB_B,
            )
            cache4lB.dump()
            cache4lB_B.dump()
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
    if root_config.enable_Lr:#Lr
        if root_config.cache_4_ldm_get_input.enable:
            root_config.cache_4_ldm_get_input.clear()
        cache4Lr=Cache4Lr(Cache4Lr.get_file_name_A(best_elev_deg1,best_shiftAzim)  )
        best_elev_deg1,best_shiftAzim,inter_Lr= calculate_match_score.refine(
            img1_warp, (best_elev_deg1,best_shiftAzim),
            estimator0,
            cache4Lr,
            image1_path_4_get_input_cache=img1_warp_path,
            noise=None,
            num_iter=root_config.Lr.num_iter,
            decay_interval=root_config.Lr.decay_interval,
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
        if root_config.enable_Lr:
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
        if root_config.SetMatch.SearchA.apply_refiner:
            Rerr_before_refiner=compute_angular_error(R_GT_opencv,relative_pose[:3,:3])
            Global.pairs_result.set_cur_dic( 'RerrBeforeRefine',Rerr_before_refiner)
            # prepare config
            root_config.REFINE_ITER = 3
            root_config.SetMatch.SearchA.SpeedUp.enable = False
            # run
            pose1 = gen6d_imgPath2absolutePose_B(estimator0, K1, image1_path, bbox1,
                                                 pose_init=pose1.copy())
            relative_pose = pose1 @ np.linalg.inv(pose0)
            R = relative_pose[:3, :3]
            t = relative_pose[:3, 3:]
            # recover config
            root_config.SetMatch.SearchA.SpeedUp.enable = True
            root_config.REFINE_ITER = 0
        # ------4 vis-------
        # Global.poseVisualizer0.append(pose0,color="grey")
        # Global.poseVisualizer0.append(pose1,color="blue")
        inter = dict(
            pose0=pose0,
            pose1=pose1,
        )
    if root_config.selector_cache_A.enable:
        root_config.selector_cache_A.cache = {}
    if root_config.cache_4_ldm_get_input.enable:
        root_config.cache_4_ldm_get_input.clear()
    return R, t, relative_pose, inter
