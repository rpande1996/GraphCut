import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from skimage import draw
from sklearn import mixture

def r(strin, val):
    if val in strin:
        new_str = strin.replace(val, ' - ')
    else:
        new_str = strin
    return new_str


def show_image(img, title='', isGray=False, xticks=[], yticks=[], scale=1.0, isCv2=True):
    img_name = r(title, '\n')
    img_name = r(img_name, ':')
    if len(img.shape) == 3:
        if isCv2:
            img = img[:, :, ::-1]
    plt.figure(figsize=scale * plt.figaspect(1))
    plt.imshow(img, interpolation='nearest')
    if isGray:
        plt.gray()
    plt.title(title)
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.axis('off')
    plt.savefig(f'../output/{img_name}.jpg')
    plt.close()



def poly2mask(vertex_row_coords, vertex_col_coords, shape):
    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
    mask = np.zeros(shape, dtype=bool)
    mask[fill_row_coords, fill_col_coords] = True
    return mask


def invMask(mask):
    if np.max(mask.astype(np.float64)) == 255.0:
        mask = mask / 255
    return (mask - np.ones(mask.shape)) * (-1)


def mask_for_fg(img, img_mask):
    mask_rgb = np.zeros(img.shape)
    for i in range(len(img.shape)):
        mask_rgb[:, :, i] = img_mask
    fg_image = img * mask_rgb
    return fg_image


def mask_for_bg(img, img_mask):
    im_mask = invMask(img_mask)
    bg_mask = np.zeros(img.shape)
    for i in range(len(img.shape)):
        bg_mask[:, :, i] = im_mask
    bg_image = img * bg_mask
    return bg_image


def gaussian_model(img):
    chnl = img.reshape(img.shape[0] * img.shape[1], -1)
    model = (mixture.GaussianMixture(n_components=8, covariance_type='full', tol=0.001, reg_covar=1e-06,
                                     max_iter=25, n_init=3)).fit(chnl)
    # gaussian_model.fit(chnl)
    gmm_model, gmm_labels = model, model.score_samples(chnl).reshape(img.shape[0], img.shape[1])

    return gmm_model, gmm_labels


def run_grabcut(img, bbox, itercount):
    mask = np.zeros(img.shape[:2], np.uint8)
    bgModel = np.zeros((1, 65), np.float64)
    fgModel = np.zeros((1, 65), np.float64)
    (mask, bgModel, fgModel) = cv2.grabCut(img, mask, bbox, bgModel, fgModel, itercount,
                                           cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    return (mask2, bgModel, fgModel)


def visualize_likelihood_map(img, img_mask, img_name):
    gmm_fg, fg_gmm_labels = gaussian_model(mask_for_fg(img, img_mask))
    gmm_bg, bg_gmm_labels = gaussian_model(mask_for_bg(img, img_mask))
    chnl = img.reshape(img.shape[0] * img.shape[1], -1)

    bg_chnl = gmm_bg.score_samples(chnl)
    bg_im = bg_chnl.reshape(img.shape[0], img.shape[1])
    show_image(bg_im, title=f"Background Likelihood Map for the {img_name} image", isGray=True)

    fg_chnl = gmm_fg.score_samples(chnl)
    fg_im = fg_chnl.reshape(img.shape[0], img.shape[1])
    show_image(fg_im, title=f"Foreground Likelihood Map for the {img_name} image", isGray=True)

    return None


def grabCut(bboxes, itercount):
    img_names = list(bboxes.keys())
    for i in range(len(itercount)):
        for j in range(len(img_names)):
            img = cv2.imread(f'../input/{img_names[j]}.jpg')
            bbox = bboxes[img_names[j]]
            (mask, bgModel, fgModel) = run_grabcut(img, bbox, itercount[i])
            show_image(mask, title=f'{img_names[j]} mask for {itercount[i]} iterations', isGray=True)
            masked_img = img * mask[:, :, np.newaxis]
            show_image(masked_img, title=f'GrabCut {img_names[j]} image for {itercount[i]} iterations')
    return None


img_name = 'cat'
img_path = '../input/cat.jpg'
img = cv2.imread(img_path)
poly = loadmat('../input/cat_poly.mat')['poly']

img_mask = poly2mask(poly[:, 1], poly[:, 0], (img.shape[0], img.shape[1]))

bboxes = {'cat': (40, 80, 285, 315), 'dog': (343, 50, 300, 600), 'plane': (44, 193, 830, 239),
          'messi': (78, 8, 579, 442), 'bird': (61, 97, 293, 375)}

itercount = [5, 50, 100, 500]


# choice = input("Please select the appropriate selection:\n"
#                "1. Visualization of the likehood map for given polygon mask\n"
#                "2. GrabCut segmentation\n"
#                "3. GrabCut segmentation with varying iterations\n")
#
# try:
#     choice = int(choice)
#     print("Incorrect selection. Please select a number between 1 and 3")
# except:
#     exit()
#
# if choice == 1:
#     visualize_likelihood_map(img, img_mask, img_name)
# elif choice == 2:
#     grabCut(bboxes, [itercount[-1]])
# elif choice == 3:
#     grabCut(bboxes, itercount[:-1])
# else:
#     print("Incorrect selection. Please select a number between 1 and 3")
#     exit()


visualize_likelihood_map(img, img_mask, img_name)

grabCut(bboxes, [itercount[-1]])

grabCut(bboxes, itercount[:-1])