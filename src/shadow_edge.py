import os
import numpy as np
from skimage import io
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

# frog_dir = "../data/frog/v1-lr/"
# frame_shape = (384,512)
frog_dir = "../data/frog/v1/"
frame_shape = (768,1024)
class ShadowDetector:
    def __init__(self, folder, num_frames):
        self.imax = np.zeros((frame_shape))
        self.imin = np.ones((frame_shape))
        self.ishadow = np.zeros((frame_shape))
        self.dir = folder
        self.num_frames = num_frames
        self.thresh = 60.0/255.0
        self.rows = frame_shape[0]
        self.cols = frame_shape[1]

    def observe_video(self):
        for i in range(1, self.num_frames + 1):
            img = (rgb2gray(io.imread(self.dir + str(i).zfill(6) + ".jpg"))) # unnormalized
            self.imax = np.maximum(self.imax, img)
            self.imin = np.minimum(self.imin, img)

        self.ishadow = 0.5 * (self.imax + self.imin)

    def low_contrast_mask(self):
        return np.where(self.imax - self.imin < self.thresh, 1.0, 0.0)

    def temporal_shadow(self):
        shadow_img = np.zeros((self.rows, self.cols))

        prev_img = rgb2gray(io.imread(self.dir + str(1).zfill(6) + ".jpg")) - self.ishadow
        for i in range(1, self.num_frames + 1):
            img = (rgb2gray(io.imread(self.dir + str(i).zfill(6) + ".jpg"))) # unnormalized
            diff_img = (img - self.ishadow)
            diff_img = np.clip(diff_img + self.low_contrast_mask(), None, 1)
            locs = np.where(np.sign(diff_img) == -1)

            us = locs[0]
            vs = locs[1]

            for j in range(len(us)):
                if (i == 1):
                    shadow_img[us[j]][vs[j]] = i
                    continue

                prev_val = prev_img[us[j]][vs[j]]
                curr_val = diff_img[us[j]][vs[j]]

                assert(curr_val <= 0)
                assert(prev_val >= 0)
                sum_val = np.abs(curr_val) + prev_val

                w1 = np.abs(curr_val)/sum_val
                w2 = prev_val/sum_val
                t_interp = (i-1)*w1 + i*w2
                shadow_img[us[j]][vs[j]] = t_interp
                # shadow_img[us[j]][vs[j]] = int(t_interp / 5.1875)
            prev_img = img

        # plt.imshow(shadow_img, cmap='jet')
        # plt.show()
        return shadow_img

    def spatial_shadow_line(self, img, srow, erow, scol, ecol):
        assert(srow < erow and scol < ecol)
        diff_img = (img - self.ishadow)
        diff_img = np.clip(diff_img + self.low_contrast_mask(), None, 1)

        lv, rv, u = [], [], []

        for r in range(srow, erow+1):
            row = np.reshape(diff_img[r,scol:ecol], (1, ecol - scol))
            signs = np.sign(row)
            signchange = ((np.roll(signs, 1) - signs) != 0).astype(int)

            locs = np.where(signchange == 1)
            # locs = np.where(row < 0)
            if(len(locs[0]) == 0): continue

            locs_v = locs[1] + scol
            u.append(r)
            lv.append(np.min(locs_v))
            rv.append(np.max(locs_v))

        lv = np.array(lv)
        rv = np.array(rv)
        return lv, rv, u

    def spatial_shadow(self, frame_path, i):
        img = rgb2gray(io.imread(frame_path))

        col_v = [225, 820]
        row_v = [0, 325]
        col_h = [190, 840]
        row_h = [655, 767]

        hlv, hrv, hu = self.spatial_shadow_line(img, row_h[0], row_h[1],
                                                      col_h[0], col_h[1])
        vlv, vrv, vu = self.spatial_shadow_line(img, row_v[0], row_v[1],
                                                      col_v[0], col_v[1])

        Ahr = np.vstack([hrv,np.ones(len(hrv))]).T
        hrm, hrb = np.linalg.lstsq(Ahr, hu, rcond=None)[0]

        Ahl = np.vstack([hlv,np.ones(len(hlv))]).T
        hlm, hlb = np.linalg.lstsq(Ahl, hu, rcond=None)[0]

        Avl = np.vstack([vlv,np.ones(len(vlv))]).T
        vlm, vlb = np.linalg.lstsq(Avl, vu, rcond=None)[0]

        Avr = np.vstack([vrv,np.ones(len(vrv))]).T
        vrm, vrb = np.linalg.lstsq(Avr, vu, rcond=None)[0]

        h_end = (450.0 - hlb)/hlm
        v_end = (450.0 - vlb)/vlm
        hlv = np.append(hlv, h_end)
        vlv = np.append(vlv, v_end)

        return hlv, hlm, hlb, vlv, vlm, vlb

    def save_shadow_planes(self):
        savez_dict = dict()
        for i in range(60, 141):
            frame_path = dir_path + str(i).zfill(6) + ".jpg"
            hlv, hlm, hlb, vlv, vlm, vlb = self.spatial_shadow(frame_path, i)
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
            np.savez("../data/plane_pts.npz", P1, P2, P3, P4)

            # save 3D shadow plane parameters
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
