import json

import torch
from imports import *
import abc
import glob
from pathlib import Path

import cv2
import numpy as np
import os
import ldm_util

from PIL import Image
from skimage.io import imread, imsave


from import_util import is_in_sysPath
if(is_in_sysPath(path=os.path.abspath(os.path.join(os.path.dirname(__file__),os.path.pardir)))):
    from utils.base_utils import read_pickle, save_pickle, pose_compose, load_point_cloud, pose_inverse, resize_img, \
    mask_depth_to_pts, transform_points_pose
    from utils.read_write_model import read_model
else:
    from ..utils.base_utils import read_pickle, save_pickle, pose_compose, load_point_cloud, pose_inverse, resize_img, \
        mask_depth_to_pts, transform_points_pose
    from ..utils.read_write_model import read_model



# from ..utils.base_utils import read_pickle, save_pickle, pose_compose, load_point_cloud, pose_inverse, resize_img, \
#     mask_depth_to_pts, transform_points_pose
# from ..utils.read_write_model import read_model



SUN_IMAGE_ROOT = 'data/SUN2012pascalformat/JPEGImages'
SUN_IMAGE_ROOT_128 = 'data/SUN2012pascalformat/JPEGImages_128'
SUN_IMAGE_ROOT_256 = 'data/SUN2012pascalformat/JPEGImages_256'
SUN_IMAGE_ROOT_512 = 'data/SUN2012pascalformat/JPEGImages_512'
SUN_IMAGE_ROOT_32 = 'data/SUN2012pascalformat/JPEGImages_64'


def get_SUN397_image_fn_list():
    if Path('data/SUN397_list.pkl').exists():
        return read_pickle('data/SUN397_list.pkl')
    img_list = os.listdir(SUN_IMAGE_ROOT)
    img_list = [img for img in img_list if img.endswith('.jpg')]
    save_pickle(img_list, 'data/SUN397_list.pkl')
    return img_list


class BaseDatabase(abc.ABC):
    def __init__(self, database_name):
        self.database_name = database_name

    @abc.abstractmethod
    def get_image(self, img_id):
        pass

    @abc.abstractmethod
    def get_K(self, img_id):
        pass

    @abc.abstractmethod
    def get_pose(self, img_id):
        pass

    @abc.abstractmethod
    def get_img_ids(self):
        pass

    def get_mask(self, img_id):
        # dummy mask
        img = self.get_image(img_id)
        h, w = img.shape[:2]
        return np.ones([h, w],
                       #    np.bool)
                       bool)  # 


LINEMOD_ROOT = 'data/LINEMOD'


class LINEMODDatabase(BaseDatabase):
    K = np.array([[572.4114, 0., 325.2611],
                  [0., 573.57043, 242.04899],
                  [0., 0., 1.]], dtype=np.float32)

    def __init__(self, database_name):
        super().__init__(database_name)
        _, self.model_name = database_name.split('/')
        self.img_ids = [str(k) for k in range(len(os.listdir(f'{LINEMOD_ROOT}/{self.model_name}/JPEGImages')))]
        self.model = self.get_ply_model().astype(np.float32)
        self.object_center = np.zeros(3, dtype=np.float32)
        self.object_vert = np.asarray([0, 0, 1], np.float32)
        self.img_id2depth_range = {}
        self.img_id2pose = {}

    def get_ply_model(self):
        fn = Path(f'{LINEMOD_ROOT}/{self.model_name}/{self.model_name}.pkl')
        if fn.exists(): return read_pickle(str(fn))
        ply = plyfile.PlyData.read(f'{LINEMOD_ROOT}/{self.model_name}/{self.model_name}.ply')
        data = ply.elements[0].data
        x = data['x']
        y = data['y']
        z = data['z']
        model = np.stack([x, y, z], axis=-1)
        if model.shape[0] > 4096:
            idxs = np.arange(model.shape[0])
            np.random.shuffle(idxs)
            model = model[idxs[:4096]]
        save_pickle(model, str(fn))
        return model

    def get_image(self, img_id):
        return imread(f'{LINEMOD_ROOT}/{self.model_name}/JPEGImages/{int(img_id):06}.jpg')

    def get_K(self, img_id):
        return np.copy(self.K)

    def get_pose(self, img_id):
        if img_id in self.img_id2pose:
            return self.img_id2pose[img_id]
        else:
            pose = np.load(f'{LINEMOD_ROOT}/{self.model_name}/pose/pose{int(img_id)}.npy')
            self.img_id2pose[img_id] = pose
            return pose

    def get_img_ids(self):
        return self.img_ids.copy()

    def get_mask(self, img_id):
        return np.sum(imread(f'{LINEMOD_ROOT}/{self.model_name}/mask/{int(img_id):04}.png'), -1) > 0


GenMOP_ROOT = 'data/GenMOP'

genmop_meta_info = {
    'cup': {'gravity': np.asarray([-0.0893124, -0.399691, -0.912288]),
            'forward': np.asarray([-0.009871, 0.693020, -0.308549], np.float32)},
    'tformer': {'gravity': np.asarray([-0.0734401, -0.633415, -0.77032]),
                'forward': np.asarray([-0.121561, -0.249061, 0.211048], np.float32)},
    'chair': {'gravity': np.asarray((0.111445, -0.373825, -0.920779), np.float32),
              'forward': np.asarray([0.788313, -0.139603, 0.156288], np.float32)},
    'knife': {'gravity': np.asarray((-0.0768299, -0.257446, -0.963234), np.float32),
              'forward': np.asarray([0.954157, 0.401808, -0.285027], np.float32)},
    'love': {'gravity': np.asarray((0.131457, -0.328559, -0.93529), np.float32),
             'forward': np.asarray([-0.045739, -1.437427, 0.497225], np.float32)},
    'plug_cn': {'gravity': np.asarray((-0.0267497, -0.406514, -0.913253), np.float32),
                'forward': np.asarray([-0.172773, -0.441210, 0.216283], np.float32)},
    'plug_en': {'gravity': np.asarray((0.0668682, -0.296538, -0.952677), np.float32),
                'forward': np.asarray([0.229183, -0.923874, 0.296636], np.float32)},
    'miffy': {'gravity': np.asarray((-0.153506, -0.35346, -0.922769), np.float32),
              'forward': np.asarray([-0.584448, -1.111544, 0.490026], np.float32)},
    'scissors': {'gravity': np.asarray((-0.129767, -0.433414, -0.891803), np.float32),
                 'forward': np.asarray([1.899760, 0.418542, -0.473156], np.float32)},
    'piggy': {'gravity': np.asarray((-0.122392, -0.344009, -0.930955), np.float32),
              'forward': np.asarray([0.079012, 1.441836, -0.524981], np.float32)},
}



class CustomDatabase_(BaseDatabase):
    def __init__(self, database_name):
        super().__init__(database_name)
        self.root = Path(os.path.join('data', database_name))
        self.img_dir = self.root / 'images'
        if (self.root / 'img_fns.pkl').exists():
            self.img_fns = read_pickle(str(self.root / 'img_fns.pkl'))
        else:
            self.img_fns = [Path(fn).name for fn in glob.glob(str(self.img_dir) + '/*.jpg')]
            save_pickle(self.img_fns, str(self.root / 'img_fns.pkl'))

        self.colmap_root = self.root / 'colmap'
        if (self.colmap_root / 'sparse' / '0').exists():
            cameras, images, points3d = read_model(str(self.colmap_root / 'sparse' / '0'))
            self.poses, self.Ks, self.img_ids = parse_colmap_project(cameras, images, self.img_fns)
        else:
            self.img_ids = [str(k) for k in range(len(self.img_fns))]
            self.poses, self.Ks = {}, {}

        if len(self.poses.keys()) > 0:
            # read meta information to scale and rotate
            directions = np.loadtxt(str(self.root / 'meta_info.txt'))
            x = directions[0]
            z = directions[1]
            self.object_point_cloud = load_point_cloud(f'{self.root}/object_point_cloud.ply')
            # rotate
            self.rotation = GenMOPMetaInfoWrapper.compute_rotation(z, x)
            self.object_point_cloud = (self.object_point_cloud @ self.rotation.T)

            # scale
            self.scale_ratio = GenMOPMetaInfoWrapper.compute_normalized_ratio(self.object_point_cloud)
            self.object_point_cloud = self.object_point_cloud * self.scale_ratio
            # self.object_point_cloud = np.array([[0,0,0],[0,0,0]], dtype=np.float32)

            min_pt = np.min(self.object_point_cloud, 0)
            max_pt = np.max(self.object_point_cloud, 0)
            self.center = (max_pt + min_pt) / 2





            def scy_debug_check_pose():
                # l_key=["0","1","2","3","4","5","6","7","8","9","10","11","12","13"]
                l_key=[
                    # "5","4","17","2","10",
                    #    "21",
                       "20",
                          # "35","67","68",
                       "131",
                       ]
                l_w2c=[np.concatenate([ self.poses[k], np.array([[0, 0, 0, 1]])], axis=0)   for k in l_key]
                l_w2c=[np.linalg.inv(w2c)   for w2c in l_w2c]
                from vis.vis_rel_pose import    vis_w2cPoses,vis_w2cPoses_B
                vis_img = vis_w2cPoses(l_w2c, y_is_vertical=0)
                # vis_img = vis_w2cPoses_B(l_w2c, )





            # scy_debug_check_pose()

            # modify poses
            for k, pose in self.poses.items():
                R = pose[:3, :3]
                t = pose[:3, 3:]
                R = R @ self.rotation.T
                t = self.scale_ratio * t
                self.poses[k] = np.concatenate([R, t], 1).astype(np.float32)
            # scy_debug_check_pose()
            print(1)
    def get_image(self, img_id):
        return imread(str(self.img_dir / self.img_fns[int(img_id)]))

    def get_K(self, img_id):
        return self.Ks[img_id].copy()

    def get_pose(self, img_id):
        return self.poses[img_id].copy()

    def get_img_ids(self):
        return self.img_ids


class Zero123CustomDatabase(CustomDatabase_):
    """
    isinstance() 与 type() 区别：
        type() 不会认为子类是一种父类类型，不考虑继承关系。
        isinstance() 会认为子类是一种父类类型，考虑继承关系。
    """
    ID_zero123InputImage='zero123InputImage'
    def __eq__(self, other):
        if self.img_dir!=other.img_dir:
            return False
        if self.img_ids!=other.img_ids:
            return False
        return True
    def __init__(self, database_name):
        # super().__init__(database_name)
        # BaseDatabase.__init__(database_name)
        self.database_name = database_name


        self.root = (Path(  confs.dataPath_rfdb   )/(database_name.split('/')[-1]  )).absolute()
        # self.img_dir = self.root / 'images'
        self.img_dir = self.root / 'ref'
        def get_end_idInteger(img_dir)->int:#eg. if there are 0.jpg,...,127.jpg in img_dir, return 128
            img_fns = [Path(fn).name for fn in glob.glob(str(img_dir) + f'/*{confs.rfdb_img_suffix}')]
            print(f"{str(img_dir) + f'/*{confs.rfdb_img_suffix}'=}")
            img_idIntegers = [k[:-4] for k in img_fns]
            #sort asc
            img_idIntegers.sort(key=lambda x:int(x))
            if(confs.USE_ALL_SAMPLE):
                img_idIntegers=img_idIntegers[:len(img_idIntegers)//confs.NUM_SAMPLE]
            # print("img_idIntegers",img_idIntegers)
            assert len(img_idIntegers)>0,img_dir
            return int(img_idIntegers[-1])+1
        # self.img_fns = [Path(fn).name for fn in glob.glob(str(self.img_dir) + '/*.jpg')]
        # self.img_ids = [str(k) for k in range(len(self.img_fns))]
        # self.img_ids = [k[:-4] for k in self.img_fns]
        end_idInteger=get_end_idInteger(self.img_dir)
        if not  end_idInteger==confs.NUM_REF:
            if confs.ELEV_RANGE :
                assert end_idInteger<=confs.NUM_REF
                print(f"[warning] {end_idInteger=}")
            else:
                raise Exception(f"{end_idInteger=}")
        self.img_ids=[str(k) for k in range(end_idInteger)]
        self.img_fns=[str(k)+confs.rfdb_img_suffix for k in range(end_idInteger)]
        ddd(f"[Zero123CustomDatabase]self.img_ids='{self.img_ids}'(img name 即img_id+.png)")

        # from scy.IntermediateResult import IntermediateResult
        from gen6d.Gen6D.scy.IntermediateResult import IntermediateResult
        intermediate_result = IntermediateResult()
        intermediate_result.load(self.root / 'ref'/'intermediateResult.json')
        self.poses={}
        self.Ks={}
        for i_str,dic in intermediate_result.data.items():
            self.poses[i_str]=dic['pose']
            self.Ks[i_str]=dic['K']

        self.object_point_cloud = np.array(
            # [[-0.5, -0.5, -0.5, ], [0.5, 0.5, 0.5, ]],
            [[-1, -1, -1, ], [1, 1, 1, ]],
              dtype=np.float32)

        self.center = np.array([0, 0, 0], dtype=np.float32)

        def scy_debug_check_pose____hydrant__106_12648_23157__4():
            # l_key=58,55,60,68,73,57,75,80
            l_key =["58","55","60","68","73","57","75","80"]
            l_w2c = [np.concatenate([self.poses[k], np.array([[0, 0, 0, 1]])], axis=0) for k in l_key]
            # l_w2c = [np.linalg.inv(w2c) for w2c in l_w2c]
            from vis.vis_rel_pose import vis_w2cPoses, vis_w2cPoses_B
            vis_img = vis_w2cPoses(l_w2c, y_is_vertical=0)
            # vis_img = vis_w2cPoses_B(l_w2c, )
            np.save(confs.path_4download+"/Zero123CustomDatabase-leftMulPose.npy",l_w2c)

        # scy_debug_check_pose____hydrant__106_12648_23157__4()
        # print(1)
    def get_image(self, img_id):
        # return imread(str(self.img_dir / self.img_fns[int(img_id)]))
        return imread(  self.get_image_path(img_id)  )
    @methodtools.lru_cache()
    def get_image_B(self,img_id):
        """
        1. ret is after load
        2. will be cache
        """
        I0giRGB=self.get_image(img_id)
        I0giRGB_tensor=ldm_util.load_image_(I0giRGB)
        debug_imsave(f'load_image_/{os.path.basename(self.get_image_path(img_id))}',I0giRGB_tensor[0])
        return I0giRGB_tensor
    def get_image_path(self,img_id):# only used by imgPaths2relativeRt_naiveSetMatch
        if img_id==self.ID_zero123InputImage:
            return self.get_zero123Input_info()['img_path__in_ref']
        return str(self.img_dir / f"{img_id}{confs.rfdb_img_suffix}")
    def get_K(self, img_id):
        if img_id == self.ID_zero123InputImage:
            return np.array(  self.get_zero123Input_info()['K']  )
        return self.Ks[img_id].copy()

    def get_pose(self, img_id):
        if img_id == self.ID_zero123InputImage:
            return np.array(self.get_zero123Input_info()['pose'])[:3,:]
        return self.poses[img_id].copy()

    def get_img_ids(self):
        return self.img_ids

    # @functools.cache
    # @methodtools.lru_cache(maxsize=16)
    def get_zero123Input_info(self):
        path_zero123Input_info= self.img_dir /'zero123Input'/'info.json'
        with open(str(path_zero123Input_info),"r")as f:
            ret=json.load(f)
        return ret
    def get_ref_relaToBase_info(self)->np.ndarray:
        """
        ret will 一路原封不动地传递进selector（trainer里会被裹进dict,estimator里直接传
        """
        if confs.USE_CONFIDENCE:
            path  = self.img_dir / 'ref_relaToBase_info.json'
            with open(str(path),"r")as f:
                ret=json.load(f)
            ret=np.array(ret,dtype=np.float32)
            assert ret.shape==(len(self.get_img_ids()),3)
            #norm
            ret[:,0]=ret[:,0]/180.0
            ret[:,1]=ret[:,1]/(90-confs.Q0_MIN_ELEV)
            ret[:,2]=ret[:,2]/180.0
            # ret=torch.from_numpy(ret) this is done in selector
        else:
            ret=None
        return ret
    def get_elev_az_many_relaToBase(self,img_id):
        #get elev,az diff and sqrt (center_elev-elev)**2+ az_diff**2 in deg
        base_pose=self.get_zero123Input_info()['pose']
        base_pose=np.array(base_pose)
        base_elev, base_azim = pose_util.R_or_pose__2__elev_azim_inDeg_B(base_pose)
        pose_=self.get_pose(img_id)
        elev,azim=pose_util.R_or_pose__2__elev_azim_inDeg_B(pose_)
        az_diff = (base_azim - azim) % 360
        if az_diff > 180:
            az_diff = 360 - az_diff
        elev_diff=abs(base_elev-elev)
        dis = (base_elev - elev) ** 2 + az_diff ** 2
        dis= math.sqrt(dis)
        return elev,azim,elev_diff,az_diff,dis
    def get_elev_az_many_relaToBase_B(self,img_id):#compared to A (the above method), the diff is relative diff, from base to index
        base_pose=self.get_zero123Input_info()['pose']
        base_pose=np.array(base_pose)
        base_elev, base_azim = pose_util.R_or_pose__2__elev_azim_inDeg_B(base_pose)
        pose_=self.get_pose(img_id)
        elev,azim=pose_util.R_or_pose__2__elev_azim_inDeg_B(pose_)
        az_diff = (azim- base_azim ) % 360
        if az_diff > 180:
            az_diff = az_diff - 360#调整到(-180,0)
        elev_diff=elev- base_elev
        dis = (base_elev - elev) ** 2 + az_diff ** 2
        dis= math.sqrt(dis)
        return elev,azim,elev_diff,az_diff,dis
def parse_database_name(database_name: str) -> BaseDatabase:
    name2database = {
        'linemod': LINEMODDatabase,
        'custom': CustomDatabase_,

        'zero123': Zero123CustomDatabase,
    }
    database_type = database_name.split('/')[0]
    if database_type in name2database:
        return name2database[database_type](database_name)
    else:
        raise NotImplementedError


def get_database_split(database, split_name):
    if split_name.startswith('linemod'):  # linemod_test or linemod_val
        assert (database.database_name.startswith('linemod'))
        model_name = database.database_name.split('/')[1]
        lines = np.loadtxt(f"{LINEMOD_ROOT}/{model_name}/test.txt", dtype=str).tolist()
        que_ids, ref_ids = [], []
        for line in lines: que_ids.append(str(int(line.split('/')[-1].split('.')[0])))
        if split_name == 'linemod_val': que_ids = que_ids[::10]
        lines = np.loadtxt(f"{LINEMOD_ROOT}/{model_name}/train.txt", dtype=str).tolist()
        for line in lines: ref_ids.append(str(int(line.split('/')[-1].split('.')[0])))
    elif split_name == 'all':
        ref_ids = que_ids = database.get_img_ids()
    else:
        raise NotImplementedError
    return ref_ids, que_ids


def get_ref_point_cloud(database):
    if isinstance(database, LINEMODDatabase):
        ref_point_cloud = database.model
    elif isinstance(database, CustomDatabase_):
        ref_point_cloud = database.object_point_cloud
    elif isinstance(database, NormalizedDatabase):
        pc = get_ref_point_cloud(database.database)
        pc = pc * database.scale + database.offset
        return pc
    else:
        raise NotImplementedError
    return ref_point_cloud


def get_diameter(database):
    if isinstance(database, LINEMODDatabase):
        model_name = database.database_name.split('/')[-1]
        return np.loadtxt(f"{LINEMOD_ROOT}/{model_name}/distance.txt") / 100
    elif isinstance(database, NormalizedDatabase):
        return 2.0
    elif isinstance(database, CustomDatabase_):
        return 2.0
    else:
        raise NotImplementedError


def get_object_center(database):
    if isinstance(database, LINEMODDatabase):
        return database.object_center
    elif isinstance(database, CustomDatabase_):
        return database.center
    elif isinstance(database, NormalizedDatabase):
        return np.zeros(3, dtype=np.float32)
    else:
        raise NotImplementedError


def get_object_vert(database):
    if isinstance(database, LINEMODDatabase):
        return database.object_vert
    elif isinstance(database, CustomDatabase_):
        return np.asarray([0, 0, 1], np.float32)
    else:
        raise NotImplementedError


def normalize_pose(pose, scale, offset):
    # x_obj_new = x_obj * scale + offset
    R = pose[:3, :3]
    t = pose[:3, 3]
    t_ = R @ -offset + scale * t
    return np.concatenate([R, t_[:, None]], -1).astype(np.float32)


def denormalize_pose(pose, scale, offset):
    R = pose[:3, :3]
    t = pose[:3, 3]
    t = R @ offset / scale + t / scale
    return np.concatenate([R, t[:, None]], -1).astype(np.float32)




def mask2bbox(mask):
    if np.sum(mask) == 0:
        return np.asarray([0, 0, 0, 0], np.float32)
    ys, xs = np.nonzero(mask)
    x_min = np.min(xs)
    y_min = np.min(ys)
    x_max = np.max(xs)
    y_max = np.max(ys)
    return np.asarray([x_min, y_min, x_max - x_min, y_max - y_min], np.int32)


class NormalizedDatabase(BaseDatabase):
    def get_image(self, img_id):
        return self.database.get_image(img_id)

    def get_K(self, img_id):
        return self.database.get_K(img_id)

    def get_pose(self, img_id):
        pose = self.database.get_pose(img_id)
        return normalize_pose(pose, self.scale, self.offset)

    def get_img_ids(self):
        return self.database.get_img_ids()

    def get_mask(self, img_id):
        return self.database.get_mask(img_id)

    def __init__(self, database: BaseDatabase):
        super().__init__("norm/" + database.database_name)
        self.database = database
        center = get_object_center(self.database)
        diameter = get_diameter(self.database)

        self.scale = 2 / diameter
        self.offset = - self.scale * center

        # self.diameter = 2.0
        # self.center = np.zeros([3],np.float32)
        # self.vert = get_object_vert(self.database)
