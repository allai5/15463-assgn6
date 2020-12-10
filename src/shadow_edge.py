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
        # TODO: decide whether this should be normalized
        self.imin = np.ones((frame_shape))
        self.ishadow = np.zeros((frame_shape))
        self.dir = folder
        self.num_frames = num_frames
        # self.thresh = 80.0/255.0
        self.thresh = 60.0/255.0
        self.rows = frame_shape[0]
        self.cols = frame_shape[1]

    def observe_video(self):
        # TODO: get the image_path, format it with 6 zeros
        for i in range(1, self.num_frames + 1):
            img = (rgb2gray(io.imread(self.dir + str(i).zfill(6) + ".jpg"))) # unnormalized
            self.imax = np.maximum(self.imax, img)
            self.imin = np.minimum(self.imin, img)

        self.ishadow = 0.5 * (self.imax + self.imin)

    def low_contrast_mask(self):
        return np.where(self.imax - self.imin < self.thresh, 1.0, 0.0)

    def temporal_shadow(self):
        shadow_img = np.zeros((self.rows, self.cols))

        for i in range(1, self.num_frames + 1):
            img = (rgb2gray(io.imread(self.dir + str(i).zfill(6) + ".jpg"))) # unnormalized
            diff_img = (img - self.ishadow)
            diff_img = np.clip(diff_img + self.low_contrast_mask(), None, 1)
            locs = np.where(np.sign(diff_img) == -1)

            us = locs[0]
            vs = locs[1]

            for j in range(len(us)):
                shadow_img[us[j]][vs[j]] = int(i / 5.1875)

        plt.imshow(shadow_img, cmap='jet')
        plt.show()

    def spatial_shadow_plane(self, img, srow, erow, scol, ecol):
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

        hlv, hrv, hu = self.spatial_shadow_plane(img, row_h[0], row_h[1],
                                                      col_h[0], col_h[1])
        vlv, vrv, vu = self.spatial_shadow_plane(img, row_v[0], row_v[1],
                                                      col_v[0], col_v[1])

        Ahr = np.vstack([hrv,np.ones(len(hrv))]).T
        hrm, hrb = np.linalg.lstsq(Ahr, hu, rcond=None)[0]

        Ahl = np.vstack([hlv,np.ones(len(hlv))]).T
        hlm, hlb = np.linalg.lstsq(Ahl, hu, rcond=None)[0]

        Avl = np.vstack([vlv,np.ones(len(vlv))]).T
        vlm, vlb = np.linalg.lstsq(Avl, vu, rcond=None)[0]

        Avr = np.vstack([vrv,np.ones(len(vrv))]).T
        vrm, vrb = np.linalg.lstsq(Avr, vu, rcond=None)[0]

        # plt.plot(hlv, hu, 'bo')
        # plt.plot(vlv, vu, 'ro')
        # plt.plot(hlv, hlm*hlv + hlb, linewidth=2)
        # plt.plot(vlv, vlm*vlv + vlb, linewidth=2)
        # plt.title(frame_path)
        # plt.imshow(img)
        # plt.show()

        return hlv, hlm, hlb, vlv, vlm, vlb


