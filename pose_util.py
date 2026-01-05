import numpy as np
import os,sys,math
# import funcy as fc
from PIL import Image
from zero123.zero1.util_4_e2vg import CameraMatrixUtil
from zero123.zero1.util_4_e2vg.CameraMatrixUtil import xyz2pose,get_z_4_normObj
def R_t_2_pose(R,t):
    if(isinstance(R,list)):
        R=np.array(R)
    if(isinstance(t,list)):
        t=np.array(t)
    if(t.shape==(3,)):
        t=t.reshape((3,1))
    t=t.reshape((3,))
    assert(t.shape==(3,))
    assert(R.shape==(3,3))
    pose=np.zeros((4,4))
    pose[:3,:3]=R
    pose[:3,3]=t
    pose[3,3]=1
    return pose
class Pose_R_t_Converter: 
    @staticmethod
    def pose_2_Rt(pose44_or_pose34):
        assert pose44_or_pose34.shape==(4,4) or pose44_or_pose34.shape==(3,4)
        R = pose44_or_pose34[:3, :3]
        t = pose44_or_pose34[:3, 3]
        return R,t
    @staticmethod
    # def Rt_2_pose44(R,t):
    def R_t3np__2__pose44(R,t):
        assert R.shape==(3,3)
        assert t.shape==(3,)
        pose=np.eye(4)
        pose[:3,:3]=R
        pose[:3,3]=t
        return pose
    @staticmethod
    def R_t3np__2__pose34(R,t):
        assert R.shape==(3,3)
        assert t.shape==(3,)
        pose44=Pose_R_t_Converter.R_t3np__2__pose44(R,t)
        pose34=pose44[:3,:]
        return pose34
    @staticmethod
    def pose34_2_pose44(pose34):
        assert pose34.shape==(3,4)
        pose44=np.concatenate([pose34,np.array([[0,0,0,1]])],axis=0)
        return pose44
    @staticmethod
    def R__2__arbitrary_t_pose44(R):
        assert R.shape==(3,3)
        pose44=Pose_R_t_Converter.R_t3np__2__pose44(R,np.zeros((3,)))
        return pose44

def opencv_2_pytorch3d__leftMulW2cR(R):#w2opencv to w2pytorch3d
    assert R.shape==(3,3)
    Rop = np.array([
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, 1],
    ], dtype=np.float64)#o means OpenCV, p means pytorch3d
    R = Rop @ R
    return R
def opencv_2_pytorch3d__leftMulW2cpose(pose):#TODO check correctness
    assert pose.shape==(4,4)
    Poseop = np.array([
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=np.float64)#o means OpenCV, p means pytorch3d
    pose = Poseop @ pose
    return pose
def opencv_2_pytorch3d__leftMulRelR(R):
    assert R.shape==(3,3)
    Rop = np.array([
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, 1],
    ], dtype=np.float64)
    R = Rop @ R @ (Rop.T)
    return R
# def pytorch3d_2_opencv__leftMulRelR(R):
#     assert R.shape==(3,3)
#     Rop = np.array([
#         [-1, 0, 0],
#         [0, -1, 0],
#         [0, 0, 1],
#     ], dtype=np.float64)
#     R = Rop @ R @ (Rop.T)
#     return R
def opencv_2_pytorch3d__leftMulRelPose(pose):
    assert pose.shape==(4,4)
    Pop = np.array([
    [-1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    ], dtype=np.float64)
    pose = Pop @ pose @ np.linalg.inv(Pop)
    return pose
def pytorch3d_2_opencv__leftMulRelPose(pose):
    return opencv_2_pytorch3d__leftMulRelPose(pose)
def pytorch3d_2_opencv__leftMulW2cpose(pose):
    assert pose.shape==(4,4)
    pose=opencv_2_pytorch3d__leftMulW2cpose(pose)
    return pose
def opengl_2_opencv__leftMulW2cpose(pose):#TODO check correctness
    assert pose.shape==(4,4)
    Posego = np.array([#GL to OpenCV
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1],
    ], dtype=np.float64)
    pose = Posego @ pose
    return pose


def compute_angular_error(rotation1, rotation2):
    # R_rel = rotation1.T @ rotation2
    R_rel =   rotation2 @ rotation1.T
    tr = (np.trace(R_rel) - 1) / 2 
    theta = np.arccos(tr.clip(-1, 1))
    return theta * 180 / np.pi
def compute_translation_error(t31_1, t31_2):
    assert t31_1.shape==(3,1)
    assert t31_2.shape==(3,1)
    # ret=np.linalg.norm(t31_1 - t31_2, axis=1)
    # angle between two vectors
    ret=np.arccos(np.dot(t31_1.T,t31_2)/(np.linalg.norm(t31_1)*np.linalg.norm(t31_2)))
    ret=ret.item()
    ret=ret * 180 / np.pi
    return ret


def R_or_pose__2__elev_azim_inDeg_A(R_or_pose,round_arg=None):
    """
    version A
    application:
        1. elevation,azimuth in refdb world coord system. the inverse operation of xyz2pose
        2. gso database get elev
    """
    vec=-R_or_pose[2][:3]#camera's coord in the refdb world coord system
    elev_degree=math.degrees(math.asin(vec[2]))
    azim_degree=math.degrees(math.atan2(vec[1],vec[0]))
    if round_arg is not None:
        # s+=f".elev={elev_degree:.3f},azim={azim_degree:.3f}"
        elev_degree=round(elev_degree,3)
        azim_degree=round(azim_degree,3)
    return elev_degree,azim_degree
# @fc.print_durations()
def R_or_pose__2__elev_azim_inDeg_B(R_or_pose,round_arg=None):# _B means version B
    ret=R_or_pose__2__elev_azim_inDeg_A(R_or_pose,round_arg)
    ret=(ret[0],   ret[1]%360)
    return ret
def is_angle_in_range_type360(angle, range_):
    """
    angle: [0,360)
    range_: (any number, any number (larger than range_[0]))
    check if an angle is in range。 note that range_[i] can be out of (0,360), eg.355 in (-10,10), 5 in (350,380)
    """
    assert 0<=angle<360,(angle,range_)
    assert len(range_)==2
    assert range_[0]<range_[1]

    # Normalize the range to 0-360
    range_ = [r % 360 for r in range_]

    # If the range crosses the 0/360 boundary, we split it into two ranges
    if range_[0] > range_[1]:
        ret= (range_[0] <= angle <= 360) or (0 <= angle <= range_[1])
    else:
        ret= range_[0] <= angle <= range_[1]
    # print("[is_angle_in_range_type360] ",angle,range_,ret)
    return ret