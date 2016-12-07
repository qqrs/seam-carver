import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import imread


def calc_energy(im):
    slice_x1 = im[1:-1, 2:, :]
    slice_x2 = im[1:-1, :-2, :]
    slice_y1 = im[2:, 1:-1, :]
    slice_y2 = im[:-2, 1:-1, :]

    energy = np.zeros(im.shape[:2], dtype=np.float64)

    energy[1:-1, 1:-1] = np.sqrt(np.sum((slice_x1 - slice_x2)**2, axis=2)
                                 + np.sum((slice_y1 - slice_y2)**2, axis=2))

    ENERGY_BORDER = np.NaN
    energy[0, :] = ENERGY_BORDER
    energy[-1, :] = ENERGY_BORDER
    energy[:, 0] = ENERGY_BORDER
    energy[:, -1] = ENERGY_BORDER

    return energy


def get_seam_path(energy):
    seam_energy = np.zeros(energy.shape)

    seam_energy[0, 0] = energy[0, 0]
    seam_energy[0, -1] = energy[0, -1]

    for i in xrange(1, energy.shape[0]-1):
        seam_energy[i, 1:-1] = (np.fmin(np.fmin(seam_energy[i-1, :-2],
                                                seam_energy[i-1, 1:-1]),
                                        seam_energy[i-1, 2:])
                                + energy[i, 1:-1])
        seam_energy[i, 0] = seam_energy[i-1, 0] + energy[i, 0]
        seam_energy[i, -1] = seam_energy[i-1, 0] + energy[i, -1]

    (m, n) = seam_energy.shape
    path = np.zeros(m, np.int)
    col = np.nanargmin(seam_energy[-2])
    path[-1] = path[-2] = col

    for i in range(m-3, 0, -1):
        tgt_energy = seam_energy[i+1, col] - energy[i+1, col]
        tgt_idx = np.nanargmin(seam_energy[i, col-1:col+2] - tgt_energy)
        path[i] = col = col + tgt_idx - 1

    path[0] = path[1]

    return path


def remove_seam(img, seam, _):
    (m, n, _) = img.shape
    return np.array([np.delete(img[row], seam[row], axis=0)
                    for row in xrange(m)])


def run_step(im, orig_xmax):
    energy = calc_energy(im)
    path_x = get_seam_path(energy)
    path_y = np.arange(len(path_x))
    im = remove_seam(im, path_x, path_y)

    plt.clf()
    plt.imshow(im); plt.axis('off');
    plt.plot(path_x, path_y, color='red')
    plt.plot(np.arange(orig_xmax), np.zeros(orig_xmax), color='black')
    plt.show()
    return im


def main():
    plt.ion()

    #im = imread('Broadway_tower_edit.jpg')
    im = imread('HJoceanSmall.jpg')
    (m, n, _) = im.shape
    plt.imshow(im); plt.axis('off')
    plt.show()
    plt.pause(0.05)

    for i in xrange(n):
        im = run_step(im, n)
        #plt.savefig('output/%04d.png' % i)
        plt.pause(0.05)
        print '.',


if __name__ == '__main__':
    main()
