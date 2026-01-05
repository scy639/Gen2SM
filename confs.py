"""
conf and some global var
"""
"""
term:
    Ls: latent search, ie the search stage
    Lr: latent refine, ie the refine stage
"""
import torch.cuda

#-------
# DATASET="gso"
# DATASET="navi"
DATASET=None
#-------
BATCH_multiple_iv :bool = True #multi intermediate viewpoint(denoted as bIV )as one batch。denoted as bIV in ID system
bIV:int = 16 # 暂时只支持 bIV被num of IV整除 的情况 (不然ddpm里那个cache会出问题)
assert BATCH_multiple_iv is False or ( isinstance(bIV,int) and bIV>=1 and BATCH_multiple_iv)
#-------
FORCE_zero123_render_even_img_exist=0
#-------
SAMPLE_BATCH_SIZE=32
SAMPLE_BATCH_B_SIZE=9
#-------
NUM_REF=64
#-------
SKIP_EVAL_SEQ_IF_EVAL_RESULT_EXIST=False
MAX_PAIRS=20
CONSIDER_IPR=False  # in-plane rotation of q0
CONSIDER_Q1_IPR=False  # in-plane rotation of q1
Q0Sipr:bool=False
Q0Sipr_range:int=45
Q1Sipr:bool=False
Q1Sipr_range:int=45
SEED=0    
Q0INDEX:int=None

idSuffix=None
refIdSuffix=None
idSuffix_=None
_refIdSuffix=None
#-------misc
tmp_image__SUFFIX='.png' 
SHARE_tmp_batch_images=False #False or folder name  
NO_CARVEKIT:bool=True #CARVEKIT is a bg remover. if input img is masked, then no need to remove bg
VIS=0 ##  # do not  visualize result to save time. when debugging, you can let it be True. to save time, you can let it be 0.     VIS:  bool/int(0 or 1)  /fp(0.0-1.0,vis ratio).  
#-------



















class RefIdWhenNormal:
    @staticmethod
    def get_id(cate,seq,_refIdSuffix):
        return f"{cate}--{seq}--{_refIdSuffix}"
    
    
import numpy as np
import math
#
ddim_steps__get_generation:int=50
Ls_bsz:int=2 # Number of repeated sampling
cache4Ls_idSuffix:str=''
CalScore_half:bool=0
CalScore__db0_iterate_interval:int=1
CalScore_ts_range:tuple=(0.4,0.43,)

cache4Lr_idSuffix:str=''
enable_Lr:bool=True
class Lr: # conf for latent refine( ie refine stage)
    ts_range:tuple=(0.4,0.43,)
    num_iter:int=6
    elev_az_learning_rate:float=1400 # learning rate when refine
    decay_interval:int=2
    Lr_latentB_bsz:int=4 # Number of repeated sampling
    class db0_iterate_strategy:
        roll:bool=True
        interval:int=2
    @classmethod
    def get_fields_in_dic(cls):#以dic形式存放几乎所有field，包括类中类的field
        fields = {attr: getattr(cls, attr) for attr in dir(cls) if
                  not callable(getattr(cls, attr)) and not attr.startswith("__")}
        fields.update({attr: getattr(cls.db0_iterate_strategy, attr) for attr in dir(cls.db0_iterate_strategy) if
                       not callable(getattr(cls.db0_iterate_strategy, attr)) and not attr.startswith("__")})
        del fields['num_iter']
        return fields
class DirectlyUseE2VGresult_if_gen6d_selector_max_score_larger_than:#not maintained now
    enable:bool=False
    thres:float=6.
#--------------------------------------------------------------------------------------
class __classes:
    class Cache_4_ldm_get_input:
        def __init(self):
            # cur
            self.cur_cond_id = None
            self.cur_target_id = None
            ##target
            self.target_2_z = {}
            ##cond
            self.cond_2_xc = {}
            self.cond_2_clip_emb = {}
            self.cond_2_c_concat = {}
            #
            self.I_am_sure_that_cache_has_built_over=False
        def __init__(self):
            self.enable: bool = True #this can not move to __init because this will be set by driver only once
            self.__init()
        def swap_cur(self):
            self.cur_cond_id,self.cur_target_id=self.cur_target_id,self.cur_cond_id
        def set_cur(self,cur_cond_id,cur_target_id):
            self.cur_cond_id=cur_cond_id
            self.cur_target_id=cur_target_id
        def clear(self):
            print(f"{len(self.target_2_z)=}")
            print(f"{len(self.cond_2_xc)=}")
            print(f"{len(self.cond_2_clip_emb)=}")
            print(f"{len(self.cond_2_c_concat)=}")
            assert self.enable
            self.__init()
cache_4_ldm_get_input=__classes.Cache_4_ldm_get_input()
class elev_bsz:
    enable=True
    BSZ:int=10 # Number of repeated sampling when use zero123 when estimate elev of ref img
class elev_bsz__4IPR:
    enable= False
    BSZ:int= 10
class LOOKAT:
    """
    look_at_type:
        WARP: 长期以来的做法
        autoA: if angle<thres_autoA, use cropA, else WARP;
        cropA:
    """
    look_at_type:str='WARP'
    thres_autoA:int=5 #valid only when look_at_type=='autoA'

#-------debug | ablation
class ForDebug:
    class forceIPRtoBe:
        enable:bool=False
        IPR:int=None 
force_elev=None
#-------GPU
GPU_INDEX=0
DEVICE=f"cuda:{GPU_INDEX}" if torch.cuda.is_available() else "cpu"
#-------zero123 l_xyz
ZERO123_MULTI_INPUT_IMAGE = 0
#-------zero123 NUM_SAMPLE
USE_ALL_SAMPLE=0
NUM_SAMPLE=1
#-------gen6d
rfdb_img_suffix:str='.png'
NUM_REF=128
USE_white_bg_Detector=1
Q0_MIN_ELEV=0 
ELEV_RANGE=None   # if not None: in degree. lower and upper bound of absolute elev. eg. (-0.1,40)
USE_CONFIDENCE=0
REFINE_ITER:int=3
#-------check,geometry
LOOK_AT_CROP_OUTSIDE_GEN6D=1
# 1:call gen6d_imgPaths2relativeRt_B(where perspective trans is performed); 0:give detection_outputs to gen6d so that perspective trans in refiner
IGNORE_EXCEPTION=    0     
ONLY_GEN_DO_NOT_MEASURE=0
LOG_WHEN_SAMPLING=0
one_SEQ_mul_Q0__one_Q0_mul_Q1=1
class CONF_one_SEQ_mul_Q0__one_Q0_mul_Q1:
    ONLY_CHECK_BASENAME=False
FOR_PAPER=False
LOAD_BY_IPC=False  
MARGIN_in_LOOK_AT=0.05
#-------4 ablation
MASK_ABLATION=None # None/'EROSION'/'DILATION'
ABLATE_REFINE_ITER:int=None# None or int (when int, it can be 0, so must use 'if confs.ABLATE_REFINE_ITER is (not) None:' instead of 'if (not) confs.ABLATE_REFINE_ITER:')
#------
SKIP_GEN_REF_IF_REF_FOLDER_EXIST=False

#-------path
import os
path_root=os.path.dirname(os.path.abspath(__file__)) #path of src
path_4debug=os.path.join(path_root,"4debug")
projPath_gen6d=os.path.join(path_root,"gen6d/Gen6D")
projPath_zero123=os.path.join(path_root,"zero123/zero1")
dataPath_zero123=os.path.join(path_root,"zero123/zero1/output_im")
dataPath_gen6d=os.path.join(path_root,projPath_gen6d,"data/zero123")
from path_configuration import *
weightPath_zero123=os.path.join(path_root,"weight/105000.ckpt")
weightPath_gen6d='weight/weight_gen6d'
weightPath_selector=os.path.join(  weightPath_gen6d ,"selector_pretrain/model_best.pth")
weightPath_refiner=os.path.join(  weightPath_gen6d ,"refiner_pretrain/model_best.pth")
weightPath_loftr=os.path.join(path_root  ,"weight/indoor_ds_new.ckpt")
evalResultPath_co3d=os.path.join(path_root,"result/eval_result")
evalVisPath=os.path.join(path_root,"result/visual")
logPath=os.path.join(path_root,"log") 
os.makedirs(path_4debug,exist_ok=True)
os.makedirs(logPath,exist_ok=True)
os.makedirs(evalResultPath_co3d,exist_ok=True)
os.makedirs(evalVisPath,exist_ok=True)


dataPath_tmp_batch_images=os.path.join(path_root,"tmp_batch_images")
# path_gen6d_data=  os.path.join(  path_data  ,"gen6d_data")
dataPath_rfdb=dataPath_gen6d
# dataPath_zero123=os.path.join(path_data,"output_im")



class RefIdUtil:#RefId:标识gen6d refdatabase,是SparseViewEstimator的里的概念；
    @staticmethod
    def id2idDir(id_):
        dir_ = os.path.join(dataPath_rfdb, id_)
        return dir_
class RefIdWhenNormal:
    @staticmethod
    def get_id(cate,seq,_refIdSuffix,Q1INDEX=None):
        ret= f"{cate}--{seq}--{_refIdSuffix}"
        if Q1INDEX is not None:
            assert isinstance(Q1INDEX,int)
            ret+=f'--{Q1INDEX}'
        return ret
"""    
class RefIdWhen4Elev:#计算elev时用的id
    @staticmethod
    def get_id(id_,input_image_path):#id_ is idWhenNormal
        id2 = f"4elev-{id_}-{os.path.basename(input_image_path)}"
        return id2
    @staticmethod
    def find_id_basedOn_refIdWhenNormal(refIdWhenNormal):
        refIdWhen4Elev_prefix = f"4elev-{refIdWhenNormal}-"
        dir_=os.path.join(projPath_gen6d, f'./data/zero123')
        files=os.listdir(dir_)
        match_files=[]
        for file in files:
            if(file.startswith(refIdWhen4Elev_prefix)):
                match_files.append(file)
        if(len(match_files)!=1):
            print(f"[IdWhen4Elev]match_files={match_files}\nidWhenNormal={refIdWhenNormal}\n如果len==0,那很可能是gen6d的这个elev线代报错，没有得到elev.json")
            assert  0
        return match_files[0]
    @staticmethod
    def find_id_basedOn_cate_seq_refIdSuffix(cate,seq,_refIdSuffix):
        refIdWhenNormal=RefIdWhenNormal.get_id(cate,seq,_refIdSuffix)
        return RefIdWhen4Elev.find_id_basedOn_refIdWhenNormal(refIdWhenNormal)
"""

ggg={} # store some Gloabl var


#----------ddim.py(注意只是gen ref database那些会走ddim as need to denoise to clean to get RGB; cal score,refine那些不是走ddim,是ddpm, as just SDS loss) 
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
class FixedNoise:
    enable=False
    __NOISE_A =None
    @classmethod
    def get_batch_noise_A(cls,BS):
        if cls.__NOISE_A is None:
            cls.__NOISE_A= torch.randn(1, 4, 32, 32, device='cuda:0')
        return cls.__NOISE_A.repeat(BS, 1, 1, 1)# Repeat the single batch for the number of batches

p_sample_ddim__dont_add_noise:bool=False
ddim_sampling__repeat_noise:bool=False
#-----------------------------------------------------------------------------------------------