"""
Process a CT image or a slice of it. It performs the following steps:

1. If there are masks for the lungs:
    - find the centers of mass of each contour.
    - find the bounding boxes of each contour.
2. Use them as positive prompts:
    - "I'm looking for what this points mark."
    - "I'm looking for what's inside this box."
3. Use the center of the image as negative prompt ("This is the background").
4. Use SAM to segment the image using the provided prompts.
"""

import argparse
import logging
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from rich import print
from segment_anything import sam_model_registry, SamPredictor
from tqdm import tqdm

from csv_keys import *
from sam_model import SamModel
from tools.argparse_helper import ArgumentParserHelper
from tools.debug import Debug
from tools.slice_masks import SliceMasks
from tools.summarizer import Summarizer
from tools.timestamp import Timestamp

logger = logging.getLogger(__name__)

LoggingEnabled = True

DatasetPath = Path('datasets/zenodo')
DebugFolderPath = Path('debug')

# Windowing settings
# TODO: this could be redundant when working with Radiopaedia images, they
#   appear to have the corresponding windowing already applied.
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
    """

    logger.info('Load a masks slice from the list of masks')
    logger.debug(f'load_masks_slice('
                 f'masks={masks.shape}, '
                 f'slice_number={slice_number})')

    assert 0 <= slice_number < masks.shape[-1]
    logger.info("Requested masks slice exists.")

    masks_slice = masks[:, :, slice_number]

    return masks_slice


def compare_original_and_predicted_masks(
        original_mask: np.array, predicted_mask: np.array
) -> Tuple[float, float]:
    """
    Compares the original segmentation mask with the one predicted. Returns a
    tuple with the Jaccard index (IoU) and the Dice coefficient.

    :param original_mask: original segmentation mask.
    :param predicted_mask: predicted segmentation mask.

    :return: Jaccard index (IoU) and the Dice coefficient of the masks
    provided.
    """

    logger.info('Compare original and predicted masks')
    logger.debug(f'compare_original_and_predicted_masks('
                 f'original_mask={original_mask.shape}, '
                 f'predicted_mask={predicted_mask.shape})')

    original_mask_transformed = original_mask != 0

    original_mask_transformed = np.fliplr(np.rot90(original_mask_transformed, k=3))
    predicted_mask_transformed = np.squeeze(predicted_mask)

    intersection = original_mask_transformed * predicted_mask_transformed
    union = (original_mask_transformed + predicted_mask_transformed) > 0

    iou = intersection.sum() / float(union.sum())
    dice = intersection.sum() * 2 / (original_mask.sum() + predicted_mask.sum())

    return iou, dice


def save_results(output_path: Path, list_of_dictionaries: list) -> Tuple[Path, Path]:
    """
    Save the result to a CSV file.

    :param output_path: where the results must be saved.
    :param list_of_dictionaries: results to save.

    :return: Paths to the resulting CSV files.
    """

    logger.info('Save results')
    logger.debug(f'save_result('
                 f'output_path={output_path}, '
                 f'list_of_dictionaries={list_of_dictionaries})')

    timestamp = Timestamp.file()

    output_path.mkdir(parents=True, exist_ok=True)

    df_raw_data = pd.DataFrame(list_of_dictionaries)

    # Save results
    iou_column = df_raw_data[IoUKey]
    iou_results = {
        MetricKey: IoUKey,
        MinKey: iou_column.min(),
        MaxKey: iou_column.max(),
        AverageKey: iou_column.mean(),
        StandardDeviationKey: iou_column.std()
    }
    dice_column = df_raw_data[DiceKey]
    dice_results = {
        MetricKey: DiceKey,
        MinKey: dice_column.min(),
        MaxKey: dice_column.max(),
        AverageKey: dice_column.mean(),
        StandardDeviationKey: dice_column.std()
    }

    results = [iou_results, dice_results]

    results_csv_output_path = output_path / Path(f'results_{timestamp}.csv')
    df_results = pd.DataFrame(results)
    df_results.to_csv(results_csv_output_path, index=False)

    # Save raw data
    raw_data_csv_output_path = output_path / Path(f'raw_data_{timestamp}.csv')
    df_raw_data.to_csv(raw_data_csv_output_path, index=False)

    return results_csv_output_path, raw_data_csv_output_path


def process_image_slice(sam_predictor: SamPredictor,
                        image: np.array,
                        masks: np.array,
                        slice_number: int,
                        debug: Debug) -> dict:
    """
    Process a slice of the image. Returns the result of the analysis.

    :param sam_predictor: SAM predictor for image segmentation.
    :param image: array with the slices of the CT.
    :param masks: masks for each slice of the CT.
    :param slice_number: slice to work with.
    :param debug: instance of Debug class.

    :return: a dictionary with the number of the slice been processed and the
    IoU between the ground truth and the prediction masks.
    """

    logger.info('Process image slice')
    logger.debug(f'process_image_slice('
                 f'sam_predictor={sam_predictor.device.type}, '
                 f'image={image.shape}, '
                 f'masks={masks.shape}, '
                 f'slice_number={slice_number}, '
                 f'debug={debug.enabled})')

    image_slice = load_image_slice(image=image, slice_number=slice_number)
    masks_slice = load_masks_slice(masks=masks, slice_number=slice_number)

    slice_masks = SliceMasks(labeled_image=masks_slice)

    mask = []
    score = []
    iou = None
    dice = None

    if slice_masks.contours is not None:
        sam_predictor.set_image(image_slice)
        mask, score, logits = sam_predictor.predict(
            point_coords=slice_masks.contours_centers,
            point_labels=slice_masks.contours_centers_labels,
            box=slice_masks.contours_bounding_boxes[0],
            multimask_output=False)

        # Compare original and predicted lung masks
        iou, dice = compare_original_and_predicted_masks(
            original_mask=masks_slice, predicted_mask=mask)
    else:
        logger.info("There are no masks for the current slice")

    if debug.enabled:
        if slice_masks.contours is not None:
            debug.set_slice_number(slice_number=slice_number)

            # Save SAM's prompt to YML
            prompts = dict()
            for index, contour_center in enumerate(slice_masks.contours_centers):
                x = int(contour_center[0])
                y = int(contour_center[1])
                label = int(slice_masks.contours_centers_labels[index])
                prompts.update({
                    index: {
                        'x': x,
                        'y': y,
                        'label': label
                    }
                })

            data = dict(
                image=debug.image_file_path.name,
                masks=debug.masks_file_path.name,
                slice=slice_number,
                prompts=prompts
            )

            debug_file_path = debug.get_file_path('prompt', '.yml')
            with open(debug_file_path, 'w') as file:
                yaml.dump(data, file, sort_keys=False)

            # Save SAM segmentation
            figure = plt.figure(figsize=(10, 10))
            plt.imshow(image_slice)
            show_mask(mask, plt.gca())
            for mask_contour in slice_masks.contours:
                plt.plot(mask_contour[:, 0], mask_contour[:, 1], color='green')
            show_points(
                coords=slice_masks.contours_centers,
                labels=slice_masks.contours_centers_labels,
                ax=plt.gca())
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
                break
            plt.title(f"Score: {score[0]:.3f}", fontsize=18)
            plt.axis('off')

            debug_file_path = debug.get_file_path('prediction', '.png')
            figure.savefig(debug_file_path, bbox_inches='tight')
            plt.close()

    result = {
        SliceNumberKey: slice_number,
        IoUKey: iou,
        DiceKey: dice
    }

    return result


def process_image(sam_predictor: SamPredictor,
                  image: np.array,
                  masks: np.array,
                  debug: Debug) -> Tuple[Path, Path]:
    """
    Process all the slices of a given image. Saves the result as two CSV files,
    one with each slice's result, another with a statistical summary. Returns
    the paths where the resulting CSV files will be stored.

    :param sam_predictor: SAM predictor for image segmentation.
    :param image: array with the slices of the CT.
    :param masks: masks for each slice of the CT.
    :param debug: instance of Debug class.

    :return: paths where the resulting CSV files are stored.
    """

    logger.info('Process image')
    logger.debug(f'process_image('
                 f'sam_predictor={sam_predictor.device.type}, '
                 f'image={image.shape}, '
                 f'masks={masks.shape}, '
                 f'debug={debug.enabled})')

    items = image.shape[-1]
    progress_bar = tqdm(desc='Processing CT image slices', total=items)

    results = []
    for slice_number in range(items):
        result = process_image_slice(sam_predictor=sam_predictor,
                                     image=image,
                                     masks=masks,
                                     slice_number=slice_number,
                                     debug=debug)
        results.append(result)
        progress_bar.update()
    progress_bar.close()

    output_path = debug.image_file_path.parent / Path('results') / Path(debug.image_file_path.stem)
    results_path, raw_data_path = save_results(output_path, results)

    return results_path, raw_data_path


def parse_arguments() -> Tuple[Path, Path, int, bool, bool]:
    """
    Parse arguments passed via command line, returning them formatted. Adequate
    defaults are provided when possible.

    :return: path of the image file, path of the masks file, slice to work
    with, dry run option, debug option.
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
        debug: Debug
) -> str:
    """
    Show a summary of the actions this script will perform.

    :param image_file_path: path to the images file.
    :param masks_file_path: path to the masks file.
    :param slice_number: slice to work with.
    :param debug: instance of Debug class.
    :param dry_run: if True, the actions will not be performed.

    :return: summary of the actions this script will perform.
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
              f'- Debug: {debug.enabled}\n' \
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
        logging.getLogger("matplotlib.colorbar").disabled = True
        logging.getLogger("matplotlib.pyplot").disabled = True

    logger.info("Start processing data")
    logger.debug("main()")

    summarizer = Summarizer()

    image_file_path, masks_file_path, slice_number, debug_enabled, dry_run = parse_arguments()

    debug = Debug(
        enabled=debug_enabled,
        image_file_path=image_file_path,
        masks_file_path=masks_file_path)

    summarizer.summary = get_summary(
        image_file_path=image_file_path,
        masks_file_path=masks_file_path,
        slice_number=slice_number,
        debug=debug,
        dry_run=dry_run)

    if dry_run:
        print()
        print('[bold]Summary:[/bold]')
        print(summarizer.summary)
        return

    sam_predictor = get_sam_predictor(SamModel.ViT_L)
    image = load_image(image_file_path=image_file_path)
    masks = load_masks(masks_file_path=masks_file_path)

    if slice_number is None:
        result = process_image(sam_predictor=sam_predictor,
                               image=image,
                               masks=masks,
                               debug=debug)
        print(f'Results saved to: "{str(result[0])}"')
        print(f'Raw data saved to: "{str(result[1])}"')
    else:
        result = process_image_slice(sam_predictor=sam_predictor,
                                     image=image,
                                     masks=masks,
                                     slice_number=slice_number,
                                     debug=debug)
        print(f'IoU: {result[IoUKey]:.3f}')
        print(f'Dice: {result[DiceKey]:.3f}')

    print(summarizer.notification_message)


if __name__ == '__main__':
    main()
