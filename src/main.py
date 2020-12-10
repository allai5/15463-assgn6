from shadow_edge import ShadowDetector
from calibration import Calib
import matplotlib.pyplot as plt

dir_path = "../data/frog/v1/"
num_frames = 166
ipath = "../data/calib/intrinsic_calib.npz"
epath = "../data/frog/v1/extrinsic_calib.npz"

def main():
    s = ShadowDetector(dir_path, num_frames)
    c = Calib(ipath, epath)
    s.observe_video()
    # s.temporal_shadow()

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # for i in range(1, num_frames + 1):
    # for i in range(61, num_frames + 1):
    for i in range(61, 131):
        frame_path = dir_path + str(i).zfill(6) + ".jpg"
        hlv, hlm, hlb, vlv, vlm, vlb = s.spatial_shadow(frame_path, i)
        # 2 arbitrary points on
        # p1 = (hlv[0], hlm*hlv[0] + hlb)
        # p2 = (hlv[len(hlv) - 1], hlm*hlv[len(hlv)-1] + hlb)
        # p3 = (vlv[0], vlm*vlv[0] + vlb)
        # p4 = (vlv[len(vlv) - 1], vlm*vlv[len(vlv)-1] + vlb)

        # P1, P2 = c.plane_pts(p1, p2, True)
        # P3, P4 = c.plane_pts(p3, p4, False)

        # ax.scatter(P1[0], P1[1], P1[2], marker='o')
        # np.savez("../data/plane_pts.npz", P1, P2, P3, P4)

        # # save 3D shadow plane parameters
        # n = np.cross((P1 - P2), (P4 - P3))
        # n_hat = n / np.linalg.norm(n)
        # np.savez("../data/plane.npz", n_hat, P1)


    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # plt.show()

    # planes = np.load("../data/plane.npz")

main()

