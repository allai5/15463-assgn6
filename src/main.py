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

    fig = plt.figure()


    savez_dict = dict()
    # for i in range(1, num_frames + 1):
    for i in range(60, 141):
        frame_path = dir_path + str(i).zfill(6) + ".jpg"
        hlv, hlm, hlb, vlv, vlm, vlb = s.spatial_shadow(frame_path, i)
        # 2 arbitrary points on
        x1, x2 = hlv[0], hlv[-1]
        y1, y2 = hlm*x1 + hlb, hlm*x2 + hlb
        p1 = (x1, y1)
        p2 = (x2, y2)

        if (y2 > y1):
            tmp = p1
            p1 = p2
            p2 = tmp

        x3, x4 = vlv[0], vlv[-1]
        y3, y4 = vlm*x3 + vlb, vlm*x4 + vlb
        p3 = (x3, y3)
        p4 = (x4, y4)

        if (y4 > y3):
            tmp = p3
            p3 = p4
            p4 = tmp

        P1, P2 = c.plane_pts(p1, p2, True)
        P3, P4 = c.plane_pts(p3, p4, False)

        P1, P2, P3, P4 = np.squeeze(P1), np.squeeze(P2), np.squeeze(P3), np.squeeze(P4)
        # # same plane lines
        # ax = fig.add_subplot(111, projection='3d')
        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        # ax.set_zlabel('Z Label')
        # # # print(P1.shape)
        # ax.plot([P1[0], P2[0]], [P1[1], P2[1]], c='g')
        # ax.plot([P3[0], P4[0]], [P3[1], P4[1]], c='r')

        # # across plane lines
        # ax.plot([P1[0], P3[0]], [P1[1], P3[1]], c='b')
        # ax.plot([P2[0], P4[0]], [P2[1], P4[1]], c='b')
        # plt.show()

        # np.savez("../data/plane_pts.npz", P1, P2, P3, P4)

        # # save 3D shadow plane parameters
        n = np.cross((P1 - P2), (P4 - P3))
        if (np.linalg.norm(n) == 0):
            print("BAD")
            print(n)
        n_hat = n / np.linalg.norm(n)

        n_key = "n_hat_" + str(i)
        P1_key = "P1_" + str(i)
        P2_key = "P2_" + str(i)
        P3_key = "P3_" + str(i)
        P4_key = "P4_" + str(i)
        savez_dict[n_key]  = n_hat
        savez_dict[P1_key] = P1
        savez_dict[P2_key] = P2
        savez_dict[P3_key] = P3
        savez_dict[P4_key] = P4


    np.savez("../data/plane.npz", **savez_dict)

    planes = np.load("../data/plane.npz")
    intrinsics = np.load("../data/calib/intrinsic_calib.npz")
    extrinsics = np.load("../data/frog/v1/extrinsic_calib.npz")
    mtx = intrinsics['mtx']
    dist = intrinsics['dist']

    rmat_h = extrinsics['rmat_h']
    tvec_h = extrinsics['tvec_h']


    scol, srow = 350, 350
    # scol, srow = 0, 0
    col_span, row_span = 400, 245
    # col_span, row_span = 100, 100

    for i in range(60, 141):
        color_img = io.imread(dir_path + str(i).zfill(6) + ".jpg") # unnormalized
        img = rgb2gray(color_img)

        pts_x, pts_y, pts_z, colors = [], [], [], []
        P2d = []
        P1s = []

        for r in range(srow, srow+row_span+1):
        # for r in range(768):
            print(r)
            for c in range(scol, scol+col_span+1):
            # for c in range(1024):
                # get the frame index
                frame_index = int(ts_img[r][c])
                if (frame_index < 60 or frame_index > 140):
                    # print(frame_index)
                    continue
                colors.append(color_img[r][c]/255.0)
                # colors.append(img[r][c])

                # get the shadow plane associated with t
                n_key   = "n_hat_" + str(frame_index)
                P1_key  = "P1_" + str(frame_index)
                n_hat   = planes[n_key]
                P1      = planes[P1_key]

                # Backproject pixel p into 3D ray r
                # TODO: play around
                ray = np.reshape(pixel2ray((c, r), mtx, dist).T, (3,))
                t = np.dot(P1, n_hat) / np.dot(ray, n_hat)
                P = t * ray
                P = np.squeeze(P)
                P1s.append(P1)
                pts_x.append(P[0])
                pts_y.append(P[1])
                pts_z.append(P[2])

        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        # ax.scatter(P1[0], P1[1], P1[2], c='r', marker='o')
        ax.scatter(pts_x, pts_y, pts_z, c=colors)
        # ax.scatter(P1s[0], P1s[1], P1s[2], c='r', marker='o')
        # visualize the point cloud!
        plt.show()





# view_planes()
main()

