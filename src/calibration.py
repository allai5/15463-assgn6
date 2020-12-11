from cp_hw6 import computeIntrinsic, computeExtrinsic, pixel2ray
import numpy as np

class Calib:
    def __init__(self, ipath, epath):
        idata = np.load(ipath)
        edata = np.load(epath)

        self.mtx    = idata['mtx']
        self.dist   = idata['dist']
        self.tvec_h = edata['tvec_h']
        self.tvec_v = edata['tvec_v']
        self.rmat_h = edata['rmat_h']
        self.rmat_v = edata['rmat_v']

    def plane_pts(self, p0, p1, isH):
        T = self.tvec_h if isH else self.tvec_v
        R = self.rmat_h if isH else self.rmat_v

        T = np.reshape(T, (3,))

        # print(np.zeros(3) - T)
        cam_pos = np.squeeze(R.T @ (np.zeros(3) - T))
        assert(cam_pos.shape == (3,))
        rays = pixel2ray(np.array([p0, p1]), self.mtx, self.dist)

        ray0 = R.T @ np.squeeze(rays[0]).T
        ray1 = R.T @ np.squeeze(rays[1]).T

        t0 = -cam_pos[2] / ray0[2]
        t1 = -cam_pos[2] / ray1[2]

        P0 = cam_pos + (t0*ray0)
        P1 = cam_pos + (t1*ray1)

        P0_cam = (np.linalg.inv(R.T) @ P0) + T
        P1_cam = (np.linalg.inv(R.T) @ P1) + T

        return P0_cam, P1_cam

