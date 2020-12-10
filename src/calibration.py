from cp_hw6 import computeIntrinsic, computeExtrinsic, pixel2ray
import numpy as np

class Calib:
    def __init__(self, ipath, epath):
        idata = np.load(ipath)
        edata = np.load(epath)

        self.mtx, self.dist = idata['mtx'], idata['dist'] # 5 x 1
        self.tvec_h, self.tvec_v = edata['tvec_h'], edata['tvec_v']
        self.rmat_h, self.rmat_v = edata['rmat_h'], edata['rmat_v']

    def plane_pts(self, p0, p1, isH):
        T = self.tvec_h if isH else self.tvec_v
        R = self.rmat_h if isH else self.rmat_v

        cam_pos = R.T @ (np.zeros((3,1)) - T)
        assert(cam_pos.shape == (3,1))
        rays = pixel2ray(np.array([p0, p1]), self.mtx, self.dist)
        # print("RAYS:\n")
        # print(rays)

        # intersects plane when z is 0
        # cam_pos.z + t * d.z = 0, t = -d.z / cam_pos.z

        ray0 = rays[0]
        ray1 = rays[1]

        t0 = -cam_pos[2] / (ray0)[0][2]
        t1 = -cam_pos[2] / (ray1)[0][2]

        P0 = cam_pos + t0*ray0
        P1 = cam_pos + t1*ray1

        P0_cam = (R @ P0) + T
        P1_cam = (R @ P1) + T

        return P0_cam, P1_cam




