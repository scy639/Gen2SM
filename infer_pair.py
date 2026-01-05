import os.path
import math
import cv2
import numpy as np
from PIL import Image
import elev_util
import image_util
import confs
from imports import Global
from imports import *
from skimage.io import imsave, imread
# from gen6d.Gen6D.pipeline import    Estimator4Co3dEval
from gen6d.Gen6D.utils.database_utils  import look_at_crop
from gen6d.Gen6D.utils.pose_utils  import let_me_look_at_2d
from gen6d.Gen6D.utils.base_utils  import pose_compose,pose_inverse
import funcy as fc
zero1_PATH = confs.projPath_zero123
import sys
sys.path.append(zero1_PATH)
def gen6d_imgPath2absolutePose(
        estimator,#Estimator4Co3dEval,
        K,
        image_path,
        detection_outputs=None,
        **kw,
):
    SHOW_GT_BBOX = 1
    SCALE_DOWN = 2
    q_img_path = image_path
    if 'pose_init' in kw:
        pose_init=kw['pose_init']
        del kw['pose_init']
    else:
        pose_init=None
    pose = estimator.estimate(
        K=K,
        q_img_path=q_img_path,
        pose_init=pose_init,
        #       SCALE_DOWN=SCALE_DOWN,
        #         SHOW_GT_BBOX=SHOW_GT_BBOX
        detection_outputs=detection_outputs,
        **kw,
    )
    return pose

#------------------------------- B -------------------------------
def look_at_wrapper_wrapper(image_path_or_arr,bbox,K,save_path=None):#TODO K will be change
    def look_at(que_img, que_K, in_pose, bbox, size=None):
        margin = confs.MARGIN_in_LOOK_AT
        h__obj_in_img = bbox[3] - bbox[1]
        w__obj_in_img = bbox[2] - bbox[0]
        size__obj_in_img = max(h__obj_in_img, w__obj_in_img) * (1 + margin * 2)
        if size is None:
            size = int(size__obj_in_img)
        bbox = np.array(bbox)
        assert bbox.shape[0] == 4
        assert len(bbox.shape) == 1
        image_center=np.array([ (bbox[0]+bbox[2])/2,(bbox[1]+bbox[3])/2])
        _, new_f = let_me_look_at_2d(image_center, que_K)
        in_f = new_f * size / size__obj_in_img
        scale = in_f / new_f
        position = image_center
        que_img_warp, que_K_warp, in_pose_warp, que_pose_rect, H = look_at_crop(
            que_img, que_K, in_pose, position, 0, scale, size, size)
        #
        cropA=False
        if confs.LOOKAT.look_at_type=='cropA':
            cropA=True
        elif confs.LOOKAT.look_at_type=='autoA':
            angle_pose_rect=compute_angular_error(que_pose_rect[:3,:3],np.eye(3))
            print(f"{angle_pose_rect=}")
            if angle_pose_rect<=confs.LOOKAT.thres_autoA:
                cropA=True
                print('cropA=1')
            else:
                print('>thres_autoA => cropA=0')
        if cropA:
            #min distance from obj center to img border
            min_dis = min((
                image_center[0],
                image_center[1],
                que_img.shape[1] - image_center[0],
                que_img.shape[0] - image_center[1],
            ))*2
            size__obj_in_img=min(size__obj_in_img,min_dis)
            #bbox -> bbox_square
            bbox=[
                image_center[0]-size__obj_in_img/2,
                image_center[1]-size__obj_in_img/2,
                image_center[0]+size__obj_in_img/2,
                image_center[1]+size__obj_in_img/2,
            ]
            #to int
            bbox=[int(round(i)) for i in bbox]
            que_img_warp= que_img[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
            #size
            ZERO123_REQUIRED_SIZE=256
            if size__obj_in_img>ZERO123_REQUIRED_SIZE:
                que_img_warp = cv2.resize(
                    que_img_warp, (ZERO123_REQUIRED_SIZE, ZERO123_REQUIRED_SIZE),
                    interpolation=cv2.INTER_AREA,
                )
            else:
                if size__obj_in_img/ZERO123_REQUIRED_SIZE  > 0.8 : 
                    print('if < REQUIRED => pad')
                    que_img_warp=image_util.pad_around_center(que_img_warp,(ZERO123_REQUIRED_SIZE,ZERO123_REQUIRED_SIZE))
                else:
                    print('if << REQUIRED => resize(upsample)')
                    que_img_warp = cv2.resize(
                        que_img_warp, (ZERO123_REQUIRED_SIZE, ZERO123_REQUIRED_SIZE),
                        interpolation=cv2.INTER_LINEAR,
                        # interpolation=cv2.INTER_AREA,
                    )
        return que_img_warp, que_pose_rect,que_K_warp

    def look_at_wrapper(image_path_or_arr,  bbox):
        if isinstance(image_path_or_arr,str):
            img = imread(image_path_or_arr)
            print(f"[look_at_wrapper_wrapper]img= {image_path_or_arr}")
        else:
            assert isinstance(image_path_or_arr,np.ndarray)
            img=image_path_or_arr
        # h, w = img.shape[:2]
        # f = np.sqrt(h ** 2 + w ** 2)
        # K = np.asarray([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float32)
        arbitrary_pose = np.concatenate([np.eye(3), np.zeros([3, 1])], 1).astype(np.float32)  
        img_warp, pose_rect ,que_K_warp= look_at(que_img=img, que_K=K, in_pose=arbitrary_pose, bbox=bbox)
        return img_warp, pose_rect,que_K_warp

    img_warp, pose_rect,que_K_warp = look_at_wrapper(image_path_or_arr, bbox)
    if save_path:
        if confs.SHARE_tmp_batch_images and os.path.exists(save_path):
            pDEBUG(f"confs.SHARE_tmp_batch_images and os.path.exists(full_path) => do not save again")
        else:
            imsave(save_path, img_warp)
        pDEBUG(f"[look_at_wrapper_wrapper]img_warp= {save_path}")
    return img_warp, pose_rect,que_K_warp
def de_look_at(pose,pose_rect):
    assert pose.shape==(4,4)
    assert pose_rect.shape==(3,4)
    pose = pose_compose(pose[:3,:], pose_inverse(pose_rect))
    pose=np.concatenate([pose,np.array([[0,0,0,1]])],axis=0)
    return pose
def get_path_after_warp(path:str):
    IMG_SUFFIX=confs.tmp_image__SUFFIX
    assert path.endswith(IMG_SUFFIX)
    img_warp_path = path.replace(IMG_SUFFIX, f"_warp{IMG_SUFFIX}")
    return img_warp_path
def gen6d_imgPath2absolutePose_B(estimator,#Estimator4Co3dEval,
                                 K,image_path,bbox,input_image_eleRadian=None,
                                 **kw,
     ):
    """
    given original img
    return img after warp and ...
    """
    img_warp_path=get_path_after_warp(image_path)
    img_warp,  pose_rect,K_warp=look_at_wrapper_wrapper(image_path, bbox,K,save_path=img_warp_path)
    if confs.ABLATE_REFINE_ITER is   None:
        pose_pr = gen6d_imgPath2absolutePose(estimator=estimator,K=K_warp, image_path=img_warp_path,
                detection_outputs=None,**kw,)  # TODO 先用none,后面给个gt detection_outputs
        #---4 vis----
        def tt46(target_note):
            for i, (pose, note) in enumerate(zip(Global.poseVisualizer1.l_pose, Global.poseVisualizer1.l_note)):
                if (note == target_note):
                    assert "(de_look_at over)" not in note
                    Global.poseVisualizer1.l_pose[i] = opencv_2_pytorch3d__leftMulW2cpose(de_look_at(pose, pose_rect))
                    Global.poseVisualizer1.l_note[i]+="(de_look_at over)"
                    Global.poseVisualizer1.append(
                        pose,
                        color=Global.poseVisualizer1.l_color[i],
                        # opacity=Global.poseVisualizer1.l_opacity[i]/2,
                        opacity=0.1,
                        note="no de_look_at",
                        size=Global.poseVisualizer1.l_size[i]
                    )
        if("Ro2" in Global.poseVisualizer1.l_note):
            tt46("Ro2")
        elif("Ro1" in Global.poseVisualizer1.l_note):
            tt46("Ro1")
        else:
            assert 1
    else:
        pose_pr=Global.RefinerInterPoses.get() 
    pose_pr=de_look_at(pose_pr,pose_rect)
    return pose_pr
def gen6d_imgPaths2relativeRt_B(
        estimator,#Estimator4Co3dEval,
        K0,
        K1,
        image0_path,
        image1_path,
        bbox0:list,#x1,y1,x2,y2
        bbox1:list,
        input_image_eleRadian=None,#image0 的
):

    if (confs.ZERO123_MULTI_INPUT_IMAGE):
        raise NotImplementedError

    if(confs.one_SEQ_mul_Q0__one_Q0_mul_Q1):
        img0_warp_path=get_path_after_warp(image0_path)
        img0_warp,   pose0_rect ,K0_warp= look_at_wrapper_wrapper(image0_path, bbox0, K0, save_path=img0_warp_path)
        zero123Input_info=estimator.get__zero123Input_info(K0,img0_warp_path)
        pose0=np.array(zero123Input_info["pose"])
        assert pose0[2, 3] == 0
        if confs.CONSIDER_IPR:
            assert pose0.shape == (4, 4)
            degree_counterClockwise=zero123Input_info["IPR_degree_counterClockwise"]
            rad_counterClockwise = np.deg2rad(degree_counterClockwise)
            P_IPR = np.asarray([  # same expression as R_z in Gen6D database_utils.py
                [np.cos(rad_counterClockwise), -np.sin(rad_counterClockwise), 0, 0],
                [np.sin(rad_counterClockwise), np.cos(rad_counterClockwise), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ], np.float32)
            pose0 = P_IPR @ pose0
        # z
        # obj_width=bbox0[2]-bbox0[0]
        # obj_height=bbox0[3]-bbox0[1]
        # img = Image.open(image0_path)
        img = Image.open(img0_warp_path)#fix bug: use img0_warp_path instead of image0_path
        img_w = img.width
        img_h = img.height
        obj_width=img_w#fix bug2
        obj_height=img_h
        z4normObj = get_z_4_normObj(fx=K0[0][0], fy=K0[1][1],
                                    obj_w_pixel=obj_width, obj_h_pixel=obj_height,
                                    img_w=img_w, img_h=img_h)
        pose0[2, 3]=z4normObj
        #
        pose0 = de_look_at(pose0, pose0_rect)# !!
    else:
        assert 0,'PR of translation of Q0 is not implemented'
        pose0=gen6d_imgPath2absolutePose_B(estimator,K0,image0_path,bbox0,input_image_eleRadian)

    pose1=gen6d_imgPath2absolutePose_B(estimator,K1,image1_path,bbox1,input_image_eleRadian)
    relative_pose =pose1 @ np.linalg.inv(pose0)
    R = relative_pose[:3, :3]
    t = relative_pose[:3, 3:]
    #------4 vis-------
    # Global.poseVisualizer0.append(pose0,color="grey")
    # Global.poseVisualizer0.append(pose1,color="blue")
    inter=dict(
        pose0=pose0,
        pose1=pose1,
    )
    return R, t,relative_pose,inter
class LookAt:
    def __init__(self,K,image_path,bbox):
        img_warp_path = get_path_after_warp(image_path)
        img_warp, pose_rect, K_warp = look_at_wrapper_wrapper(image_path, bbox, K, save_path=img_warp_path)
        self.img_warp=img_warp
        self.img_warp_path=img_warp_path
        self.pose_rect=pose_rect
        self.K_warp=K_warp
    def get(self,pose_pr=None):
        if pose_pr is not None:
            pose_pr = de_look_at(pose_pr, self.pose_rect)
        return self.K_warp,self.img_warp_path,self.img_warp,pose_pr

# class Perspective():
#     @staticmethod
#     def look_at_crop(img,R,t,):
#         """
#         ret new_img,R,t
#         """











def wq_elev_deg__2__pose_wTorq(
        elev_deg,
        path_before_warp,
        K,
        bbox,
        degree_counterClockwise=0,
):  # elev_deg1 -> pose_w1Toq,...
    image0_path = path_before_warp
    K0 = K
    bbox0 = bbox
    # ---------copy from gen6d_imgPaths2relativeRt_B
    img0_warp_path = get_path_after_warp(image0_path)
    with HiddenPrints():
        img0_warp, pose0_rect, K0_warp = look_at_wrapper_wrapper(image0_path, bbox0, K0, save_path=img0_warp_path)
    if confs.CONSIDER_IPR or confs.CONSIDER_Q1_IPR:
        rad_counterClockwise = np.deg2rad(-degree_counterClockwise)#不知道为什么，反正没负号结果就很差（虽然目前带负号结果也很差
        P_IPR = np.asarray([  # same expression as R_z in Gen6D database_utils.py
            [np.cos(rad_counterClockwise), -np.sin(rad_counterClockwise), 0, 0],
            [np.sin(rad_counterClockwise), np.cos(rad_counterClockwise), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], np.float32)
        assert pose0_rect.shape==(3,4,)
        pose0_rect = np.concatenate([pose0_rect, np.array([[0, 0, 0, 1]])], axis=0)
        pose0_rect = P_IPR@pose0_rect
        pose0_rect=pose0_rect[:3,:]
    else:
        assert degree_counterClockwise==0
    pose_w1Toqwarp = elev_util.eleRadian_2_base_w2c(np.deg2rad(elev_deg))  # warp后的pose;z随意给的（0）,not z4normObj
    assert pose_w1Toqwarp[2, 3] == 0
    img = Image.open(img0_warp_path)
    img_w = img.width
    img_h = img.height
    obj_width = img_w
    obj_height = img_h
    z4normObj = get_z_4_normObj(fx=K0[0][0], fy=K0[1][1],
                                obj_w_pixel=obj_width, obj_h_pixel=obj_height,
                                img_w=img_w, img_h=img_h)
    pose_w1Toqwarp[2, 3] = z4normObj
    pose_wTorq = de_look_at(pose_w1Toqwarp, pose0_rect)  # !!
    return pose_wTorq, pose_w1Toqwarp, pose0_rect

def rq____path__2__pose_wjTorq(path,estimator,bbox,K,f1=None):
    image0_path=path
    estimator0=estimator
    K0=K
    bbox0=bbox
    # ---------copy from gen6d_imgPaths2relativeRt_B
    img0_warp_path = get_path_after_warp(image0_path)
    img0_warp, pose0_rect, K0_warp = look_at_wrapper_wrapper(image0_path, bbox0, K0, save_path=img0_warp_path)
    if f1:
        zero123Input_info = f1(estimator0,  K0, img0_warp_path,  )
    else:
        zero123Input_info = estimator0.get__zero123Input_info(K0, img0_warp_path)
    pose0 = np.array(zero123Input_info["pose"])  # warp后的pose;z随意给的（0）,not z4normObj
    assert pose0[2, 3] == 0
    if confs.CONSIDER_IPR:
        assert pose0.shape == (4, 4)
        degree_counterClockwise=zero123Input_info["IPR_degree_counterClockwise"]
        rad_counterClockwise = np.deg2rad(degree_counterClockwise)
        P_IPR = np.asarray([  # same expression as R_z in Gen6D database_utils.py
            [np.cos(rad_counterClockwise), -np.sin(rad_counterClockwise), 0, 0],
            [np.sin(rad_counterClockwise), np.cos(rad_counterClockwise), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], np.float32)
        pose0 = P_IPR @ pose0
    # z
    # obj_width=bbox0[2]-bbox0[0]
    # obj_height=bbox0[3]-bbox0[1]
    # img = Image.open(image0_path)
    img = Image.open(img0_warp_path)  # fix bug: use img0_warp_path instead of image0_path
    img_w = img.width
    img_h = img.height
    obj_width = img_w  # fix bug2
    obj_height = img_h
    z4normObj = get_z_4_normObj(fx=K0[0][0], fy=K0[1][1],
                                obj_w_pixel=obj_width, obj_h_pixel=obj_height,
                                img_w=img_w, img_h=img_h)
    pose0[2, 3] = z4normObj
    pose0 = de_look_at(pose0, pose0_rect)  # !!
    return pose0

