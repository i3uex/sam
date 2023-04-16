import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple
from segment_anything import sam_model_registry, SamPredictor

import humanize
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from rich import print
from scipy.ndimage import center_of_mass
from skimage import measure

from tools.argparse_helper import ArgumentParserHelper

logger = logging.getLogger(__name__)

LoggingEnabled = True

DatasetPath = Path('datasets/zenodo')
DebugFolderPath = Path('debug')

# Windowing settings
WindowWidth = 1500
WindowLevel = -650


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:, :, i] = color_mask[i]
        ax.imshow(np.dstack((img, m * 0.35)))


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)

# TODO: use this method only when needed
def to_greyscale(image: np.ndarray) -> np.ndarray:
    greyscale_image = (image - image.min()) / (image.max() - image.min()) * 255
    return greyscale_image

def windowing(image: np.ndarray) -> np.ndarray:
    windowed_image = image[:, :].clip(
        WindowLevel - WindowWidth // 2,
        WindowLevel + WindowWidth // 2)
    return windowed_image


def process_image_slice(image_file_path: Path,
                        masks_file_path: Path,
                        slice_number: int,
                        debug: bool):
    """
    Process a slice of the image.

    :param image_file_path: path to the images file.
    :param masks_file_path: path to the masks file.
    :param slice_number: slice to work with.
    :param debug: if True, save debug data for later inspection.
    """

    logger.info('Process image slice')
    logger.debug(f'process_image_slice('
                 f'image_file_path="{image_file_path}", '
                 f'masks_file_path="{masks_file_path}", '
                 f'slice_number={slice_number}, '
                 f'debug={debug})')

    image = np.load(str(image_file_path))
    image_slice = image[:, :, slice_number]
    image_slice = windowing(image_slice)
    image_slice = to_greyscale(image_slice)
    image_slice = image_slice.astype(np.uint8)
    # image_slice = image_slice.reshape(image_slice.shape[0], image_slice.shape[1], 1)
    image_slice = np.stack((image_slice,) * 3, axis=-1)

    masks = np.load(str(masks_file_path))
    masks_slice = masks[:, :, slice_number]

    lungs_contours = []
    lungs_centers_of_mass = []
    masks_slice_max = int(masks_slice.max())
    for lung_mask_index in np.arange(start=1, stop=masks_slice_max + 1):
        lung_mask = masks_slice == lung_mask_index
        lung_mask_contour = measure.find_contours(lung_mask)[0]
        lungs_contours.append(lung_mask_contour)
        lung_center_of_mass = center_of_mass(lung_mask)
        lungs_centers_of_mass.append(lung_center_of_mass)
    lungs_centers_of_mass = np.array(lungs_centers_of_mass).astype(np.uint)

    # Use the center of mass as prompt for the segmentation
    # Include SAM contours in the debug information

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # sam = sam_model_registry['vit_h'](checkpoint='model_checkpoints/sam_vit_h_4b8939.pth')
    sam = sam_model_registry['vit_l'](checkpoint='model_checkpoints/sam_vit_l_0b3195.pth')
    # sam = sam_model_registry['vit_b'](checkpoint='model_checkpoints/sam_vit_b_01ec64.pth')
    sam.to(device)

    predictor = SamPredictor(sam)
    predictor.set_image(image_slice)

    masks, scores, logits = predictor.predict(
        point_coords=np.array([
            [lungs_centers_of_mass[0][1], lungs_centers_of_mass[0][0]],
            [lungs_centers_of_mass[1][1], lungs_centers_of_mass[1][0]]
        ]),
        point_labels=np.array([0, 1]),
        multimask_output=True)

    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image_slice)
        show_mask(mask, plt.gca())
        show_points(
            np.array([
                [lungs_centers_of_mass[0][1], lungs_centers_of_mass[0][0]],
                [lungs_centers_of_mass[1][1], lungs_centers_of_mass[1][0]]
            ]),
            np.array([0, 1]),
            plt.gca())
        plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()

    # Hipótesis: coge el más pequeño con el score más alto
    # https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb

        # mask_generator = SamAutomaticMaskGenerator(sam)
    # masks = mask_generator.generate(ct_image)
    #
    # plt.figure(figsize=(20, 20))
    # plt.imshow(ct_image)
    # show_anns(masks)
    # plt.axis('off')
    # plt.show()

    if debug:
        # Create the base file name for the debug data
        output_folder_path = DebugFolderPath / Path(image_file_path.stem)
        output_folder_path.mkdir(parents=True, exist_ok=True)
        output_base_file_name = f'slice_{slice_number}_prompt'
        output_base_file_path = output_folder_path / Path(output_base_file_name)

        # Save SAM's prompt to YML
        prompts = dict()
        for index, lung_center_of_mass in enumerate(lungs_centers_of_mass):
            prompts.update({
                index: {
                    'x': float(lung_center_of_mass[0]),
                    'y': float(lung_center_of_mass[1])
                }
            })

        data = dict(
            image=image_file_path.name,
            masks=masks_file_path.name,
            slice=slice_number,
            prompts=prompts
        )

        with open(output_base_file_path.with_suffix('.yml'), 'w') as file:
            yaml.dump(data, file, sort_keys=False)

        # Save a plot with SAM's prompt
        figure = plt.figure()
        plt.style.use('grayscale')
        plt.pcolormesh(image_slice.T)
        plt.colorbar()

        color = plt.colormaps['rainbow'](np.linspace(0, 1, len(lungs_contours)))
        for lung_mask_index in np.arange(start=0, stop=len(lungs_contours)):
            lung_contour = lungs_contours[lung_mask_index]
            plt.plot(lung_contour[:, 0], lung_contour[:, 1], linewidth=2, color=color[lung_mask_index])
            lung_center_of_mass = lungs_centers_of_mass[lung_mask_index]
            plt.scatter(lung_center_of_mass[0], lung_center_of_mass[1], color=color[lung_mask_index])

        figure.savefig(output_base_file_path.with_suffix('.png'), bbox_inches='tight')
        plt.close()


def parse_arguments() -> Tuple[Path, Path, int, bool, bool]:
    """
    Parse arguments passed via command line, returning them formatted. Adequate
    defaults are provided when possible.

    :return: path of the image file, path of the masks file, slice to work
    with, dry run option, debug option.
    :rtype: Tuple[Path, Path, int, bool, bool]
    """

    logger.info('Get script arguments')
    logger.debug('parse_arguments()')

    program_description = 'Process image'
    argument_parser = argparse.ArgumentParser(description=program_description)
    argument_parser.add_argument('-i', '--image_file_path',
                                 required=True,
                                 help='path to image file')
    argument_parser.add_argument('-m', '--masks_file_path',
                                 required=True,
                                 help='path to masks file')
    argument_parser.add_argument('-s', '--slice',
                                 required=True,
                                 help='slice to work with')
    argument_parser.add_argument('-n', '--dry_run',
                                 action='store_true',
                                 help='show what would be done, do not do it')
    argument_parser.add_argument('-d', '--debug',
                                 action='store_true',
                                 help='save debug data for later inspection')

    arguments = argument_parser.parse_args()
    image_file_path = ArgumentParserHelper.parse_file_path_segment(
        arguments.image_file_path)
    masks_file_path = ArgumentParserHelper.parse_file_path_segment(
        arguments.masks_file_path)
    slice_number = ArgumentParserHelper.parse_integer(arguments.slice)
    debug = arguments.debug
    dry_run = arguments.dry_run

    return Path(image_file_path), Path(masks_file_path), slice_number, debug, dry_run


def get_summary(
        image_file_path: Path,
        masks_file_path: Path,
        slice_number: int,
        dry_run: bool,
        debug: bool
) -> str:
    """
    Show a summary of the actions this script will perform.

    :param image_file_path: path to the images file.
    :param masks_file_path: path to the masks file.
    :param slice_number: slice to work with.
    :param debug: if True, save debug data for later inspection.
    :param dry_run: if True, the actions will not be performed.

    :return: summary of the actions this script will perform.
    :rtype: str
    """

    logger.info('Get summary')
    logger.debug(f'get_summary('
                 f'image_file_path="{image_file_path}", '
                 f'masks_file_path="{masks_file_path}", '
                 f'slice_number={slice_number}, '
                 f'debug={debug}, '
                 f'dry_run={dry_run})')

    image = np.load(str(image_file_path))
    image_slices = image.shape[-1]
    masks = np.load(str(masks_file_path))
    masks_slices = masks.shape[-1]

    summary = f'- Image file path: "{image_file_path}"\n' \
              f'- Masks file path: "{masks_file_path}"\n' \
              f'- Slice: {slice_number}\n' \
              f'- Debug: {debug}\n' \
              f'- Dry run: {dry_run}\n' \
              f'- Image slices: {image_slices}\n' \
              f'- Masks slices: {masks_slices}\n' \
              f'- Equal number of slices: {image_slices == masks_slices}\n' \
              f'- Requested slice in range: {slice_number < image_slices}'

    return summary


def main():
    """
    Set logging up, parse arguments, and process data.
    """

    if LoggingEnabled:
        logging.basicConfig(
            filename="debug.log",
            level=logging.DEBUG,
            format="%(asctime)-15s %(levelname)8s %(name)s %(message)s")

    logger.info("Start processing data")
    logger.debug("main()")

    start_timestamp = datetime.now()

    image_file_path, masks_file_path, slice_number, debug, dry_run = parse_arguments()

    summary = get_summary(
        image_file_path=image_file_path,
        masks_file_path=masks_file_path,
        slice_number=slice_number,
        debug=debug,
        dry_run=dry_run)

    if dry_run:
        print()
        print('[bold]Summary[/bold]')
        print(summary)
        return

    process_image_slice(image_file_path=image_file_path,
                        masks_file_path=masks_file_path,
                        slice_number=slice_number,
                        debug=debug)

    end_timestamp = datetime.now()
    elapsed_seconds = (end_timestamp - start_timestamp).seconds
    start_time = start_timestamp.strftime("%H:%M:%S")
    start_date = start_timestamp.strftime("%Y-%m-%d")
    elapsed_time = humanize.naturaldelta(timedelta(seconds=elapsed_seconds))
    notification_message = \
        f"The task started at {start_time} on {start_date} has just finished.\n" \
        f"It took {elapsed_time} to complete.\n" \
        f"Summary of operations performed:\n" \
        f"{summary}"
    print(notification_message)

    # images_paths = sorted(DatasetPath.glob('COVID-19-CT-Seg_20cases/*.nii.gz'))
    # images_masks_paths = sorted(DatasetPath.glob('Lung_Mask/*.nii.gz'))
    #
    # images_paths_len = len(images_paths)
    # logger.info(f'Number of images: {len(images_paths)}')
    #
    # image_path = images_paths[0]
    # image_mask_path = images_masks_paths[0]
    #
    # image_path_name = image_path.name
    # logger.info(f'Processing image "{image_path_name}"')
    #
    # image = nib.load(image_path)
    # image = image.get_fdata()
    # logger.info(f'Image shape: "{image.shape}"')
    #
    # image_mask = nib.load(image_mask_path)
    # image_mask = image_mask.get_fdata()
    # logger.info(f'Image mask shape: "{image_mask.shape}"')
    #
    # image_slice_index = 122
    #
    # image_slice = image[:, :, image_slice_index]
    # image_slice = windowing(image_slice)
    # image_slice = to_greyscale(image_slice)
    # image_slice_mask = image_mask[:, :, image_slice_index]


if __name__ == '__main__':
    main()
