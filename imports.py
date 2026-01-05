import confs,os,sys,shutil
import methodtools
from skimage.io import imread, imsave
from logging_util import *
import pose_util
from pose_util import *
from misc_util import *
from exception_util import *
from dataset_util import *
import  vis.cv2_util as cv2_util
from vis.InterVisualizer import InterVisualizer
from vis.vis_rel_pose import PoseVisualizer
import numpy as np
import os,sys,math,functools,inspect,PIL
# import matplotlib
# matplotlib.use('Agg')
from miscellaneous.MyDataFrame import *
class Global:
    anything={}
    curPairStr:str=None
    curPairOnlyQ0Str:str=None
    pairs_result=MyDataFrameA()
    class degree_counterClockwise1_channel:
        x=None
        @classmethod
        def get(cls):
            ret=cls.x
            cls.x=None
            assert ret is not  None
            return ret
        @classmethod
        def set(cls,x):
            assert cls.x is   None,cls.x
            assert x is not   None
            cls.x=x
    class ImagePair:
        def __init__(self,):
            self.l=[]
        def append(self,im):
            __N = 1 if confs.one_SEQ_mul_Q0__one_Q0_mul_Q1 else 2
            self.l.append(im)
            if(len(self.l)>__N):
                self.l=self.l[-__N:]
    intermediate={
        "E2VG":{
            "inter_img":ImagePair(),
        },
    }
    
    poseVisualizer1=PoseVisualizer()
    # interVisualizer=InterVisualizer()
    class RefinerInterPoses:
        """
        """
        __l=[]# list of refiner raw output pose 
        """
        @classmethod
        def from_dicValue(cls,l:list):
            assert cls.__l ==[]
            assert l
            cls.__l=l
        """
        @classmethod
        def set_evalResult(cls,_evalResult,  ):
            cls.__evalResult = _evalResult
        @classmethod
        def load_pair(cls,sequence_name, i, j):
            assert cls.__l ==[]
            l=cls.__evalResult .get_pair__in_dic(sequence_name, i, j) ['RefinerInterPoses']
            assert l
            cls.__l=l
        @classmethod
        def to_dicValue_and_clear(cls, )->list:
            assert cls.__l
            ret=cls.__l
            cls.__l=[]
            ret=to_list_to_primitive(ret)
            return ret
        """
        @classmethod
        def append(cls,i,raw ):
            assert len(cls.__l)==i
            cls.__l.append(raw)
        """
        @classmethod
        def set(cls,l ):#l: [before refiner,after 1st refine,2nd,...]
            assert len(l)==confs.REFINE_ITER+1
            if confs.ABLATE_REFINE_ITER is not None:
                assert cls.__l==[]
            cls.__l=l
        @classmethod
        def get(cls, ):
            ret=cls.__l[confs.ABLATE_REFINE_ITER]
            ret=np.array(ret)
            ret=Pose_R_t_Converter.pose34_2_pose44(ret)
            return ret
from debug_util import debug_imsave
