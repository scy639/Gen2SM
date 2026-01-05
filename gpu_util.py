"""
2024.9.3 overwrite from D:\Python39\Lib\site-packages\scy639\gpu_util.py
"""
import math
import os
import sys
import time
def check_mem( cuda_device):
    devices_info = os.popen('"/usr/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().strip().split("\n")
    total, used = devices_info[int(cuda_device)].split(',')
    return total,used
def gpu_info(gpu_index:int):
    info = os.popen('nvidia-smi|grep %').read().split('\n')[gpu_index].split('|')
    power = int(info[1].split()[-3][:-1])
    memory = int(info[2].split('/')[0].strip()[:-3])
    return power, memory
def get_min_gpu_info(gpu_index:tuple ):#返回显存最小的GPU index and info
    for i in gpu_index:
        power, memory = gpu_info(i)
        if i == gpu_index[0]:
            min_power = power
            min_memory = memory
            min_index = i
        else:
            if memory < min_memory:
                min_power = power
                min_memory = memory
                min_index = i
    return min_index, min_power, min_memory

def block_util_gpu_idle(THRES_gpu_memory=5000,THRES_gpu_power=250,
                        INTERVAL=10,
                        L_GPU_INDEX=(0,1,2,3,  )  ):
    """usage: 

import os
if    0    :
    os.environ["CUDA_VISIBLE_DEVICES"] = '4'
else:
    from scy639.gpu_util import block_util_gpu_idle
    block_util_gpu_idle(
        # THRES_gpu_memory=99999,THRES_gpu_power=99999,
        THRES_gpu_memory=5000,THRES_gpu_power=250,
        INTERVAL=10,
        L_GPU_INDEX=(0,1,2,3,4,5,6,7  )  ,
    )


    """
    assert 'torch' not in sys.modules.keys() #assert pkg torch is not loaded
    if 0  :#for test
        THRES_gpu_memory=99999
        THRES_gpu_power=99999
    # ----------- CONFIG over----------------------

    def narrow_setup(interval=INTERVAL,gpu_index=L_GPU_INDEX):
        min_index,gpu_power, gpu_memory = get_min_gpu_info(gpu_index)
        i = 0
        while gpu_memory > THRES_gpu_memory or gpu_power > THRES_gpu_power:  # set waiting condition
            min_index,gpu_power, gpu_memory = get_min_gpu_info(gpu_index)
            i = i % 5
            symbol = 'monitoring: ' + '>' * i + ' ' * (10 - i - 1) + '|'
            gpu_power_str = 'gpu power:%d W |' % gpu_power
            gpu_memory_str = 'gpu memory:%d MiB |' % gpu_memory
            # sys.stdout.write('\r' + gpu_memory_str + ' ' + gpu_power_str + ' ' + symbol)
            sys.stdout.write(  gpu_memory_str + ' ' + gpu_power_str + ' ' + symbol+'\n')
            sys.stdout.flush()
            time.sleep(interval)
            i += 1
        if 1:
            assert 'torch' not in sys.modules.keys() #assert pkg torch is not loaded
            os.environ["CUDA_VISIBLE_DEVICES"] = str(min_index  )
            print(f"gpu {min_index} is selected")
    narrow_setup()
    