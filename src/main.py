from skimage.color import rgb2gray
from skimage import io
from shadow_edge import ShadowDetector
from calibration import Calib
from cp_hw6 import pixel2ray
import matplotlib.pyplot as plt
import numpy as np

dir_path = "../data/frog/v1/"
num_frames = 166
ipath = "../data/calib/intrinsic_calib.npz"
epath = "../data/frog/v1/extrinsic_calib.npz"

def view_planes():
    data = np.load("../data/plane.npz")
    for i in range(60, 141):
        P1_key = "P1_" + str(i)
        P2_key = "P2_" + str(i)
        P3_key = "P3_" + str(i)
        P4_key = "P4_" + str(i)

        P1 = data[P1_key]
        P2 = data[P2_key]
        P3 = data[P3_key]
        P4 = data[P4_key]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(P1[0], P1[1], P1[2])
        ax.scatter(P2[0], P2[1], P2[2])
        ax.scatter(P3[0], P3[1], P3[2])
        ax.scatter(P4[0], P4[1], P4[2])
        plt.show()

def main():

    s = ShadowDetector(dir_path, num_frames)
    c = Calib(ipath, epath)
    s.observe_video()
    ts_img = s.temporal_shadow()
    # s.save_shadow_planes()

    fig = plt.figure()

    planes = np.load("../data/plane.npz")
    intrinsics = np.load("../data/calib/intrinsic_calib.npz")
    extrinsics = np.load("../data/frog/v1/extrinsic_calib.npz")
    mtx = intrinsics['mtx']
    dist = intrinsics['dist']

    scol, srow = 350, 350
    col_span, row_span = 400, 245

    for i in range(60, 141):
        color_img = io.imread(dir_path + str(i).zfill(6) + ".jpg") # unnormalized
        img = rgb2gray(color_img)

        pts_x, pts_y, pts_z, colors = [], [], [], []

        for r in range(srow, srow+row_span+1):
            print(r)
            for c in range(scol, scol+col_span+1):
                frame_id = ts_img[r][c]
                if (frame_id < 60 or frame_id > 140):
                   continue

                assert(frame_id.dtype == np.double)
                frame_id1 = np.floor(frame_id)
                frame_id2 = np.ceil(frame_id)

                w1 = frame_id2 - frame_id
                w2 = frame_id - frame_id1

                nhat_1   = planes["n_hat_"  + str(int(frame_id1))]
                P1_1     = planes["P1_"     + str(int(frame_id1))]
                nhat_2   = planes["n_hat_"  + str(int(frame_id2))]
                P1_2     = planes["P1_"     + str(int(frame_id2))]

                n_hat = w1 * nhat_1 + w2 * nhat_2
                P1    = w1 * P1_1   + w2 * P1_2

                colors.append(color_img[r][c]/255.0)

                # Backproject pixel p into 3D ray r
                ray = np.reshape(pixel2ray((c, r), mtx, dist).T, (3,))
                t = np.dot(P1, n_hat) / np.dot(ray, n_hat)
                P = np.squeeze(t * ray)
                pts_x.append(P[0])
                pts_y.append(P[1])
                pts_z.append(P[2])

        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        ax.scatter(pts_x, pts_y, pts_z, c=colors)
        plt.show()

# view_planes()
main()

