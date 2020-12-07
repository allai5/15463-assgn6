import os
import numpy as np
from skimage import io
import matplotlib.pyplot as plt

# frame_shape = (768,1024,3)
frame_shape = (384,512,3)
class ShadowDetector:
    def __init__(self, folder, num_frames):
        self.imax = np.zeros((frame_shape))
        # TODO: decide whether this should be normalized
        self.imin = np.ones((frame_shape)) * 255
        self.ishadow = np.zeros((frame_shape))
        self.dir = folder
        self.num_frames = num_frames
        self.thresh = 85.0

    def observe_video(self):
        # TODO: get the image_path, format it with 6 zeros
        for i in range(1, self.num_frames + 1):
            img = io.imread(self.dir + str(i).zfill(6) + ".jpg") # unnormalized
            self.imax = np.maximum(self.imax, img)
            self.imin = np.minimum(self.imin, img)

        self.ishadow = 0.5 * (self.imax + self.imin)

    def low_contrast_mask(self):
        return np.where(self.imax - self.imin < self.thresh, 1.0, 0.0)


    def spatial_shadow(self, frame_path):
        img = io.imread(frame_path)
        diff_img = (img - self.ishadow)/255.0
        diff_img = np.clip(diff_img + self.low_contrast_mask(), 0, 1)

        # diff_img = diff_img[0:230,:,:]
        # diff_img = diff_img[230:,:,:]

        locs = np.where(diff_img == 0)
        locs_u = locs[0]
        locs_v = locs[1]

        u_bar = np.mean(locs_u)
        v_bar = np.mean(locs_v)

        m = np.sum((locs_u - u_bar) * (locs_v - v_bar)) / np.sum((locs_v - v_bar)**2)

        b = u_bar - m * v_bar

        x = locs_v
        plt.plot(locs_v, locs_u, 'go')
        plt.plot(x, m*x + b)
        plt.imshow(img)
        plt.show()


frog_dir = "../data/frog/v1-lr/"
s = ShadowDetector(frog_dir, 166)
s.observe_video()
s.spatial_shadow("../data/frog/v1-lr/000150.jpg")
# s.temporal_shadow()
