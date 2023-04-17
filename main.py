import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple

import humanize
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from rich import print
from scipy.ndimage import center_of_mass
from segment_anything import sam_model_registry, SamPredictor
from skimage import measure
from tqdm import tqdm

from sam_model import SamModel
from tools.argparse_helper import ArgumentParserHelper

logger = logging.getLogger(__name__)

LoggingEnabled = True

DatasetPath = Path('datasets/zenodo')
DebugFolderPath = Path('debug')

# Windowing settings
WindowWidth = 1500
WindowLevel = -650


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


# TODO: use this method only when needed, this is, only if the image is not
#   greyscale already.
# TODO: improve naming.
# TODO: create a class for the images?
# TODO: create a class for the slices?
def to_greyscale(image: np.ndarray) -> np.ndarray:
    """
    Normalice slice values, so they are in the range 0 to 255, this is,
    greyscale.

    :param image: slice of the CT image to greyscale.

    :return: slice of the CT image in greyscale.
    :rtype: np.ndarray
    """

    logger.info('Apply windowing to CT image slice')
    logger.debug(f'windowing('
                 f'image={image.shape})')

    greyscale_image = (image - image.min()) / (image.max() - image.min()) * 255
    return greyscale_image


# TODO: improve naming.
# TODO: pass windowing parameters instead of using global values.
# TODO: create a class for the images?
# TODO: create a class for the slices?
def windowing(image: np.ndarray) -> np.ndarray:
    """
    Use windowing to improve image contrast, focusing only in the range of
    values of interest in the slice.

    :param image: slice of the CT image to window.

    :return: windowed slice of the CT image.
    :rtype: np.ndarray
    """

    logger.info('Apply windowing to CT image slice')
    logger.debug(f'windowing('
                 f'image={image.shape})')

    windowed_image = image[:, :].clip(
        WindowLevel - WindowWidth // 2,
        WindowLevel + WindowWidth // 2)

    return windowed_image


def get_sam_predictor(sam_model: SamModel) -> SamPredictor:
    """
    Get an instance of the SAM predictor, given the model details.

    :param sam_model: model name and checkpoint to use.

    :return: an instance of the SAM predictor, given the model details.
    :rtype: SamPredictor
    """

    logger.info('Get SAM predictor instance')
    logger.debug(f'get_sam_predictor('
                 f'sam_model={sam_model})')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device for SAM predictor: {device}')
    sam = sam_model_registry[sam_model.name](checkpoint=sam_model.checkpoint)
    sam.to(device)
    sam_predictor = SamPredictor(sam)

    return sam_predictor


def load_image(image_file_path: Path) -> np.array:
    """
    Load the CT image.

    :param image_file_path: path of the image file.

    :return: CT image.
    :rtype: np.array
    """

    logger.info('Load the CT image')
    logger.debug(f'load_image('
                 f'image_file_path="{image_file_path}")')

    image = np.load(str(image_file_path))
    return image


def load_masks(masks_file_path: Path) -> np.array:
    """
    Load the CT masks.

    :param masks_file_path: path of the masks file.

    :return: masks for the CT image.
    :rtype: np.array
    """

    logger.info('Load the CT image masks')
    logger.debug(f'load_masks('
                 f'masks_file_path="{masks_file_path}")')

    masks = np.load(str(masks_file_path))
    return masks


def load_image_slice(image: np.array, slice_number: int) -> np.array:
    """
    Return a slice from a CT image, given its position. The slice is windowed
    to improve its contrast, converted to greyscale, and expanded to RGB. It
    checks if the slice number exists.

    :param image: CT image from which to get the slice.
    :param slice_number: slice number to get from the image.

    :return: slice from a CT image.
    :rtype: np.array
    """

    logger.info('Load a slice from a CT image')
    logger.debug(f'load_image_slice('
                 f'image={image.shape}, '
                 f'slice_number={slice_number})')

    assert 0 <= slice_number < image.shape[-1]
    logger.info("Requested slice exists.")

    image_slice = image[:, :, slice_number]
    image_slice = windowing(image_slice)
    image_slice = to_greyscale(image_slice)
    image_slice = image_slice.astype(np.uint8)
    image_slice = np.stack((image_slice,) * 3, axis=-1)

    return image_slice


def load_masks_slice(masks: np.array, slice_number: int) -> np.array:
    """
    Return a slice masks from the list of masks, given its position. It checks
    if the slice number exists.

    :param masks: list of masks.
    :param slice_number: masks slice number to get from the list of masks.

    :return: masks slice from a list of masks.
    :rtype: np.array
    """

    logger.info('Load a masks slice from the list of masks')
    logger.debug(f'load_masks_slice('
                 f'masks={masks.shape}, '
                 f'slice_number={slice_number})')

    assert 0 <= slice_number < masks.shape[-1]
    logger.info("Requested masks slice exists.")

    masks_slice = masks[:, :, slice_number]

    return masks_slice


# TODO: pass a Debug object with file path properties, along with its status.
def process_image_slice(sam_predictor: SamPredictor,
                        image: np.array,
                        masks: np.array,
                        slice_number: int,
                        debug: bool,
                        image_file_path: Path,
                        masks_file_path: Path):
    """
    Process a slice of the image.

    :param sam_predictor: SAM predictor for image segmentation.
    :param image: array with the slices of the CT.
    :param masks: masks for each slice of the CT.
    :param slice_number: slice to work with.
    :param debug: if True, save debug data for later inspection.
    :param image_file_path: path to the images file, for debugging tasks.
    :param masks_file_path: path to the masks file, for debugging tasks.
    """

    logger.info('Process image slice')
    logger.debug(f'process_image_slice('
                 f'sam_predictor="{sam_predictor}", '
                 f'image={image.shape}, '
                 f'masks="{masks.shape}", '
                 f'slice_number={slice_number}, '
                 f'debug={debug}, '
                 f'image_file_path="{image_file_path}", '
                 f'masks_file_path="{masks_file_path}")')

    image_slice = load_image_slice(image=image, slice_number=slice_number)
    masks_slice = load_masks_slice(masks=masks, slice_number=slice_number)

    lungs_contours = []
    lungs_centers_of_mass = []
    masks = []
    scores = []
    masks_slice_max = int(masks_slice.max())

    if masks_slice_max > 0:
        for lung_mask_index in np.arange(start=1, stop=masks_slice_max + 1):
            lung_mask = masks_slice == lung_mask_index
            lung_mask_contours = measure.find_contours(lung_mask)
            if len(lung_mask_contours) > 0:
                lung_mask_contour = lung_mask_contours[0]
                lungs_contours.append(lung_mask_contour)
                lung_center_of_mass = center_of_mass(lung_mask)
                lungs_centers_of_mass.append(lung_center_of_mass)
        lungs_centers_of_mass = np.array(lungs_centers_of_mass).astype(np.uint)

        # Use the center of mass as prompt for the segmentation
        # Include SAM contours in the debug information

        sam_predictor.set_image(image_slice)

        # TODO: Use first center of mass for foreground, the rest for background.
        lungs_centers_of_mass_labels = np.zeros(len(lungs_centers_of_mass))
        lungs_centers_of_mass_labels[0] = 1
        masks, scores, logits = sam_predictor.predict(
            point_coords=np.array(lungs_centers_of_mass),
            point_labels=lungs_centers_of_mass_labels,
            multimask_output=True)
    else:
        logger.info("There is no masks for the current slice")

    # Hypothesis: get the smallest area with the highest score
    # https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb

    if debug:
        # Create the base file name for the debug data
        output_folder_path = DebugFolderPath / Path(image_file_path.stem)
        output_folder_path.mkdir(parents=True, exist_ok=True)
        output_base_file_name = f'slice_{slice_number}'
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

        file_stem = f'{output_base_file_path.stem}_prompt'
        file_path = output_base_file_path \
            .with_stem(file_stem) \
            .with_suffix('.yml')
        with open(file_path, 'w') as file:
            yaml.dump(data, file, sort_keys=False)

        # Save a plot with SAM's prompt
        figure = plt.figure()
        plt.gca().invert_yaxis()
        plt.style.use('grayscale')
        plt.pcolormesh(image_slice)
        plt.colorbar()

        color = plt.colormaps['rainbow'](np.linspace(0, 1, len(lungs_contours)))
        for lung_mask_index in np.arange(start=0, stop=len(lungs_contours)):
            lung_contour = lungs_contours[lung_mask_index]
            plt.plot(lung_contour[:, 0], lung_contour[:, 1], linewidth=2, color=color[lung_mask_index])
            lung_center_of_mass = lungs_centers_of_mass[lung_mask_index]
            plt.scatter(lung_center_of_mass[0], lung_center_of_mass[1], color=color[lung_mask_index])

        file_path = output_base_file_path \
            .with_stem(file_stem) \
            .with_suffix('.png')
        figure.savefig(file_path, bbox_inches='tight')
        plt.close()

        # Save SAM segmentation
        for i, (mask, score) in enumerate(zip(masks, scores)):
            figure = plt.figure(figsize=(10, 10))
            plt.imshow(image_slice)
            show_mask(mask, plt.gca())
            lungs_centers_of_mass_labels = np.zeros(len(lungs_centers_of_mass))
            lungs_centers_of_mass_labels[0] = 1
            show_points(
                np.array(lungs_centers_of_mass),
                np.array(lungs_centers_of_mass_labels),
                plt.gca())
            plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
            plt.axis('off')

            file_stem = f'{output_base_file_path.stem}_prediction_{i}'
            file_path = output_base_file_path \
                .with_stem(file_stem) \
                .with_suffix('.png')
            figure.savefig(file_path, bbox_inches='tight')
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
                                 required=False,
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
    if arguments.slice is not None:
        slice_number = ArgumentParserHelper.parse_integer(arguments.slice)
    else:
        slice_number = None
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
    if slice_number is not None:
        requested_slice_in_range = slice_number < image_slices
    else:
        requested_slice_in_range = None

    summary = f'- Image file path: "{image_file_path}"\n' \
              f'- Masks file path: "{masks_file_path}"\n' \
              f'- Slice: {slice_number}\n' \
              f'- Debug: {debug}\n' \
              f'- Dry run: {dry_run}\n' \
              f'- Image slices: {image_slices}\n' \
              f'- Masks slices: {masks_slices}\n' \
              f'- Equal number of slices: {image_slices == masks_slices}\n' \
              f'- Requested slice in range: {requested_slice_in_range}'

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
        logging.getLogger("matplotlib.font_manager").disabled = True

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

    sam_predictor = get_sam_predictor(SamModel.ViT_L)
    image = load_image(image_file_path=image_file_path)
    masks = load_masks(masks_file_path=masks_file_path)

    if slice_number is not None:
        process_image_slice(sam_predictor=sam_predictor,
                            image=image,
                            masks=masks,
                            slice_number=slice_number,
                            debug=debug,
                            image_file_path=image_file_path,
                            masks_file_path=masks_file_path)
    else:
        items = image.shape[-1]
        progress_bar = tqdm(desc='Processing image slices', total=items)
        for slice_number in range(items):
            progress_bar.desc = f'Processing image slice {slice_number}'
            process_image_slice(sam_predictor=sam_predictor,
                                image=image,
                                masks=masks,
                                slice_number=slice_number,
                                debug=debug,
                                image_file_path=image_file_path,
                                masks_file_path=masks_file_path)
            progress_bar.update()
        progress_bar.close()

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


if __name__ == '__main__':
    main()
