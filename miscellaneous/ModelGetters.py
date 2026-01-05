import confs
import gpu_util
import torch
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
import diffusers  # 0.12.1
from transformers import AutoFeatureExtractor  # , CLIPImageProcessor
from omegaconf import OmegaConf
from ldm.util import create_carvekit_interface, load_and_preprocess, instantiate_from_config
import funcy as fc




import os
class ModelsGetter:
    models = None
    #param. B
    device=None
    ckpt=None
    config=None
    @classmethod
    def set_param(cls, device='cuda:0', ckpt=confs.weightPath_zero123, config=os.path.join(confs.projPath_zero123,'configs/sd-objaverse-finetune-c_concat-256.yaml')):
        cls.device=device
        cls.ckpt=ckpt
        cls.config=config
        return None
    @classmethod
    def get_B(cls,  NO_CARVEKIT=True):
        return cls.get( cls.device, cls.ckpt, cls.config,NO_CARVEKIT=NO_CARVEKIT)
    #
    @classmethod
    def to(cls,device):
        if  cls.models is None:
            return
        cls.models['turncam'].to(device)
        # cls.models['carvekit'].to(device)
    # @classmethod
    # def to_cpu(cls,):
    #     cls.to("cpu")
    # @classmethod
    # def to_gpu(cls,):
    #     cls.to(confs.DEVICE)
    @classmethod
    def release_gmem(cls):
        cls.to("cpu")
        torch.cuda.empty_cache()
    @classmethod
    def get(cls, device, ckpt, config,NO_CARVEKIT):
        @fc.print_durations()
        def load_model_from_config(config, ckpt, device, verbose=False):
            print(f'Loading model from {ckpt}')
            if confs.LOAD_BY_IPC:
                from miscellaneous.MemoryCache import MemoryCache
                import io
                buffer=MemoryCache.receive()
                buffer = io.BytesIO(buffer)
                pl_sd = torch.load(buffer, map_location=device  )# if map_location='cpu', AIcluster 有时报 'killed'
                del buffer
            else:
                pl_sd = torch.load(ckpt, map_location=device )
            if 'global_step' in pl_sd:
                print(f'Global Step: {pl_sd["global_step"]}')
            sd = pl_sd['state_dict']
            del pl_sd
            # import sys
            model = instantiate_from_config(config.model)
            m, u = model.load_state_dict(sd, strict=False)
            del sd
            if len(m) > 0 and verbose:
                print('missing keys:')
                print(m)
            if len(u) > 0 and verbose:
                print('unexpected keys:')
                print(u)

            model.to(device)
            model.eval()
            return model
        if (cls.models == None):
            print("[ModelGetter]cls.model==None, loading model...")
            config = OmegaConf.load(config)

            # Instantiate all models beforehand for efficiency.
            models = dict()
            print('Instantiating LatentDiffusion...')
            models['turncam'] = load_model_from_config(config, ckpt, device=device)#  see zero123/zero1/configs/sd-objaverse-finetune-c_concat-256.yaml
            if   NO_CARVEKIT:
                print("NO_CARVEKIT")
            else:
                print('Instantiating Carvekit HiInterface...')
                models['carvekit'] = create_carvekit_interface()
            if 0:#CHECK_NSFW
                print('Instantiating StableDiffusionSafetyChecker...')
                models['nsfw'] = StableDiffusionSafetyChecker.from_pretrained(
                    'CompVis/stable-diffusion-safety-checker').to(device)
                # Reduce NSFW false positives.
                # NOTE: At the time of writing, and for diffusers 0.12.1, the default parameters are:
                # models['nsfw'].concept_embeds_weights:
                # [0.1800, 0.1900, 0.2060, 0.2100, 0.1950, 0.1900, 0.1940, 0.1900, 0.1900, 0.2200, 0.1900,
                #  0.1900, 0.1950, 0.1984, 0.2100, 0.2140, 0.2000].
                # models['nsfw'].special_care_embeds_weights:
                # [0.1950, 0.2000, 0.2200].
                # We multiply all by some factor > 1 to make them less likely to be triggered.
                models['nsfw'].concept_embeds_weights *= 1.07
                models['nsfw'].special_care_embeds_weights *= 1.07
                print('Instantiating AutoFeatureExtractor...')
                models['clip_fe'] = AutoFeatureExtractor.from_pretrained(
                    'CompVis/stable-diffusion-safety-checker')
            else:
                print('Skipping NSFW check.')

            cls.models = models
        else:
            cls.to(device)
        return cls.models