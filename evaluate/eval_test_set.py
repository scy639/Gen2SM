
        
import os,sys
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from imports import *
from infer_pair import *
import sys
import gen6d.Gen6D.pipeline as pipeline
from miscellaneous.EvalResult import EvalResult
from evaluate.eval_on_an_obj import eval_on_an_obj



def run(datasetName_,model_name ="E2VG"):
    l__datasetName_cate_seq_Q0INDEX=get__l__datasetName_cate_seq_Q0INDEX(datasetNames=[datasetName_],datasetName_2_s=MyTestset.datasetName_2_s)
    # l__datasetName_cate_seq_Q0INDEX = l__datasetName_cate_seq_Q0INDEX[:11]
    # l__datasetName_cate_seq_Q0INDEX = l__datasetName_cate_seq_Q0INDEX[-12:]
    # l__datasetName_cate_seq_Q0INDEX = l__datasetName_cate_seq_Q0INDEX[:13]
    # l__datasetName_cate_seq_Q0INDEX = l__datasetName_cate_seq_Q0INDEX[-14:]
    #
    SUFFIX=""
    for datasetName,cate,seq, q0 in l__datasetName_cate_seq_Q0INDEX:
        if datasetName=='navi':
            confs.tmp_image__SUFFIX='.jpg' #save disk space
        assert seq==""
        confs.DATASET = datasetName
        for Q0INDEX in [q0]:
            confs.Q0INDEX = Q0INDEX
            confs.idSuffix = f"{confs.SEED}+{Q0INDEX}{confs.idSuffix_}"
            confs.refIdSuffix = f"+{Q0INDEX}{confs.refIdSuffix_}"
            
            
            
            
            eval_on_an_obj(
                category=cate,
                model_name=model_name,
            )
    EvalResult.AllAcc.dump_average_acc(SUFFIX )
