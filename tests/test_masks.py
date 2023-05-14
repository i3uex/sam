from pathlib import Path
from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt

from tools.image_slice import ImageSlice

data_folder_path = Path('notebooks/data')


def load_slice(image_number, slice_number) -> Tuple[np.array, np.array]:
    image_slice_file_name = \
        f'coronacases_{image_number}_image_slice_{slice_number}.npy'
    masks_slice_file_name = \
        f'coronacases_{image_number}_masks_slice_{slice_number}.npy'
    image_slice_path = data_folder_path / Path(image_slice_file_name)
    masks_slice_path = data_folder_path / Path(masks_slice_file_name)

    image_slice = np.load(str(image_slice_path))
    print(f'image_slice.shape: {image_slice.shape}')

    masks_slice = np.load(str(masks_slice_path))
    print(f'masks_slice.shape: {masks_slice.shape}')

    return image_slice, masks_slice


# TODO: add documentation to this method, taken from SAM's notebooks.
# https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


# TODO: add documentation to this method, taken from SAM's notebooks.
# https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def main():
    # image_slice, masks_slice = load_slice(image_number='001', slice_number='98')
    image_slice, masks_slice = load_slice(image_number='001', slice_number='110')
    # image_slice, masks_slice = load_slice(image_number='001', slice_number='122')
    # image_slice, masks_slice = load_slice(image_number='008', slice_number='26')

    slice_masks = ImageSlice(points=image_slice, labeled_points=masks_slice)

    plt.figure(figsize=(10, 10), layout='constrained')
    plt.axis('off')
    plt.style.use('grayscale')
    plt.imshow(image_slice)

    for contour in slice_masks.__contours:
        plt.plot(contour[:, 0], contour[:, 1], color='green')
    for contour_mask in slice_masks.__contours_masks:
        show_mask(contour_mask, plt.gca())
    show_points(
        slice_masks.centers,
        slice_masks.centers_labels,
        plt.gca())
    for contours_bounding_box in slice_masks.contours_bounding_boxes:
        xs = [contours_bounding_box[0],
              contours_bounding_box[1],
              contours_bounding_box[1],
              contours_bounding_box[0],
              contours_bounding_box[0]]
        ys = [contours_bounding_box[3],
              contours_bounding_box[3],
              contours_bounding_box[2],
              contours_bounding_box[2],
              contours_bounding_box[3]]
        plt.plot(xs, ys, color='blue', linewidth=1.25)
    plt.show()


if __name__ == '__main__':
    main()
