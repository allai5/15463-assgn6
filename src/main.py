from shadow_edge import ShadowDetector
from calibration import Calib
import matplotlib.pyplot as plt
import numpy as np

dir_path = "../data/frog/v1/"
num_frames = 166
ipath = "../data/calib/intrinsic_calib.npz"
epath = "../data/frog/v1/extrinsic_calib.npz"

def main():
    s = ShadowDetector(dir_path, num_frames)
    c = Calib(ipath, epath)
    s.observe_video()
    # s.temporal_shadow()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # for i in range(1, num_frames + 1):
    for i in range(61, 131):
        frame_path = dir_path + str(i).zfill(6) + ".jpg"
        hlv, hlm, hlb, vlv, vlm, vlb = s.spatial_shadow(frame_path, i)
        # 2 arbitrary points on
        p1 = (hlv[0], hlm*hlv[0] + hlb)
        p2 = (hlv[len(hlv) - 1], hlm*hlv[len(hlv)-1] + hlb)
        p3 = (vlv[0], vlm*vlv[0] + vlb)
        p4 = (vlv[len(vlv) - 1], vlm*vlv[len(vlv)-1] + vlb)

        P1, P2 = c.plane_pts(p1, p2, True)
        P3, P4 = c.plane_pts(p3, p4, False)

        P1, P2, P3, P4 = np.squeeze(P1), np.squeeze(P2), np.squeeze(P3), np.squeeze(P4)

        # same plane lines
        ax.plot([P1[0], P2[0]], [P1[1], P2[1]], c='g')
        ax.plot([P3[0], P4[0]], [P3[1], P4[1]], c='r')

        # across plane lines
        ax.plot([P1[0], P4[0]], [P1[1], P4[1]], c='b')
        ax.plot([P2[0], P3[0]], [P2[1], P2[1]], c='b')

        # ax.scatter(P1[0], P1[1], P1[2], marker='o', c='r')
        # ax.scatter(P2[0], P2[1], P2[2], marker='o', c='r')
        # ax.scatter(P3[0], P3[1], P3[2], marker='o', c='b')
        # ax.scatter(P4[0], P4[1], P4[2], marker='o', c='b')
        # np.savez("../data/plane_pts.npz", P1, P2, P3, P4)

        # # save 3D shadow plane parameters
        # n = np.cross((P1 - P2), (P4 - P3))
        # n_hat = n / np.linalg.norm(n)
        # np.savez("../data/plane.npz", n_hat, P1)


    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

    # planes = np.load("../data/plane.npz")

main()

