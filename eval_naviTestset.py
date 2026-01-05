

import sys,os
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(cur_dir, "src"))

import confs
from evaluate.eval_test_set import run





if __name__=='__main__':
    
    #--------------------------------------------configs------------------------------------------------
    #-------Works on L20/A40(46GB); if cuda OOM, keep **halving** the values below until it fits.
    confs.SAMPLE_BATCH_SIZE = 32
    confs.bIV = 32 # 1. NUM_REF % bIV must == 0; 2.
    #------key conf for:---- elev estimate------------------------------------
    confs.elev_bsz.BSZ=5
    #------key conf for:---- gen ref set (so called ref database in code)-----
    confs.NUM_REF = 64
    #------key conf for:---- search stage-------------------------------------
    confs.Ls_bsz = 2
    #------key conf for:---- refine stage-------------------------------------
    confs.Lr.Lr_latentB_bsz = 2
    confs.Lr.num_iter = 4
    confs.Lr.db0_iterate_strategy.interval = 2
    
    
    
    #---------------for cache---------------
    # skip to eval an obj if its eval result exists
    confs.SKIP_EVAL_SEQ_IF_EVAL_RESULT_EXIST = 1
    # NOTE: if you changed something in the ref set generation process, you'd better modify refIdSuffix_ to avoid reusing old ref set db
    confs.refIdSuffix_=f'-releaseV1-R{confs.NUM_REF}elev{confs.elev_bsz.BSZ}'
    # NOTE: if you changed search stage algorithm, you'd better modify cache4Ls_idSuffix to avoid hitting old search stage cache 
    confs.cache4Ls_idSuffix = f'-releaseV1-R{confs.NUM_REF}-Ls{confs.Ls_bsz}_{confs.CalScore__db0_iterate_interval}'
    # NOTE: if you changed refine stage algorithm, you'd better modify Lr_STR to avoid hitting old refine stage cache
    Lr_STR=f'-roll{confs.Lr.db0_iterate_strategy.interval}-{confs.Lr.Lr_latentB_bsz}-{confs.Ls_bsz}{confs.Lr.ts_range}-{confs.Lr.elev_az_learning_rate}-{confs.Lr.decay_interval}-bIV{confs.bIV}'
    #
    lB_STR=confs.cache4Ls_idSuffix
    confs.idSuffix_ = f'{lB_STR}{Lr_STR}-{confs.Lr.num_iter}'
    confs.cache4Lr_idSuffix = f'{lB_STR}{Lr_STR}'
    
    
    
    run(  'navi' )