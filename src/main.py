from skimage.color import rgb2gray
from skimage import io
from shadow_edge import ShadowDetector
from calibration import Calib
from cp_hw6 import pixel2ray, set_axes_equal
import matplotlib.pyplot as plt
import numpy as np

# dir_path = "../data/frog/v1/"
# ipath = "../data/calib/intrinsic_calib.npz"
# epath = "../data/frog/v1/extrinsic_calib.npz"
# ppath = "../data/plane.npz"
# num_frames = 166
# frame_shape = (768,1024)
# sframe = 60
# eframe = 140
# scol, srow = 350, 350
# col_span, row_span = 400, 245
# col_v = [225, 820]
# row_v = [0, 325]
# col_h = [190, 840]
# row_h = [655, 767]
# plane_y = 450.0
# thresh = 60.0/255.0

# dir_path = "../data/cow2/"
# ipath = "../data/mycalib2/intrinsic_calib.npz"
# epath = "../data/cow2/extrinsic_calib.npz"
# ppath = "../data/plane.npz"
# num_frames = 360
# frame_shape = (675,1200)

# sframe = 215
# eframe = 300

# col_v = [270, 775]
# row_v = [0, 475]
# col_h = [285, 805]
# row_h = [600, 674]

# scol, srow = 460, 475
# col_span, row_span = 135, 125
# plane_y = 440.0
# thresh = 0.2

dir_path = "../data/remote2/"
ipath   = "../data/mycalib2/intrinsic_calib.npz"
epath = "../data/remote2/extrinsic_calib.npz"
ppath = "../data/plane.npz"

num_frames = 290
frame_shape = (675,1200)
sframe = 190
eframe = 265

col_v = [270, 775]
row_v = [0, 450]
col_h = [285, 805]
row_h = [600, 674]

# crop image parameters
scol, srow = 450, 275
col_span, row_span = 150, 140
plane_y = 442.0
thresh = 0.15


def view_planes():
    data = np.load(ppath)
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
    # plt.imshow(io.imread(dir_path + str(1).zfill(6) + ".jpg"))
    # plt.show()

    c = Calib(ipath, epath)
    s = ShadowDetector(dir_path, num_frames, frame_shape, col_v, row_v, col_h,
                       row_h, c, thresh)
    s.observe_video()
    ts_img = s.temporal_shadow()
    s.save_shadow_planes(plane_y, sframe, eframe)

    fig = plt.figure()

    planes = np.load(ppath)
    intrinsics = np.load(ipath)
    extrinsics = np.load(epath)
    mtx = intrinsics['mtx']
    dist = intrinsics['dist']

    # for i in range(60, 141):
    for i in range(sframe, eframe+1):
        color_img = io.imread(dir_path + str(i).zfill(6) + ".jpg") # unnormalized
        img = rgb2gray(color_img)

        pts_x, pts_y, pts_z, colors = [], [], [], []

        for r in range(srow, srow+row_span+1):
            print(r)
            for c in range(scol, scol+col_span+1):
                frame_id = ts_img[r][c]
                if (frame_id < sframe or frame_id > eframe):
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

                # print(P)

                # print(P[2])

        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        # set_axes_equal(ax)
        # frog axes
        # ax.set_xlim3d(-300, 200)
        # ax.set_ylim3d(-200, 100)
        # ax.set_zlim3d(1400, 1800)

        # cow axes
        # ax.set_xlim3d(-30, 10)
        # ax.set_ylim3d(70, 140)
        # ax.set_zlim3d(1000, 1500)

        # remote axes

        ax.scatter(pts_x, pts_y, pts_z, c=colors, marker='.')
        plt.show()

# view_planes()
main()

