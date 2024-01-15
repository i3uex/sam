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
import os
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
from tools.image_slice import ImageSlice
from tools.summarizer import Summarizer
from tools.timestamp import Timestamp

logger = logging.getLogger(__name__)

LoggingEnabled = True

DatasetPath = Path('datasets/covid')
DebugFolderPath = Path('debug')

DEBUG_DRAW_SAM_PREDICTION = bool(os.environ.get('DEBUG_DRAW_SAM_PREDICTION', 'True') == str(True))
DEBUG_DRAW_MASKS_CONTOURS = bool(os.environ.get('DEBUG_DRAW_MASKS_CONTOURS', 'True') == str(True))
DEBUG_DRAW_BOUNDING_BOX = bool(os.environ.get('DEBUG_DRAW_BOUNDING_BOX', 'True') == str(True))
DEBUG_DRAW_NEGATIVE_PROMPT = bool(os.environ.get('DEBUG_DRAW_NEGATIVE_PROMPT', 'True') == str(True))
USE_BOUNDING_BOX = bool(os.environ.get('USE_BOUNDING_BOX', 'True') == str(True))


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
    positive_points = coords[labels == 1]
    negative_points = coords[labels == 0]

    # scatter shows x and y, but we are using rows and columns (y and x)
    rows = positive_points[:, 0]
    columns = positive_points[:, 1]
    ax.scatter(columns, rows, color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)

    if DEBUG_DRAW_NEGATIVE_PROMPT:
        rows = negative_points[:, 0]
        columns = negative_points[:, 1]
        ax.scatter(columns, rows, color='red', marker='*', s=marker_size, edgecolor='white',
                   linewidth=1.25)


# TODO: add documentation to this method, taken from SAM's notebooks.
# https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='orange', facecolor=(0, 0, 0, 0), lw=3))


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

    image_npz = np.load(str(image_file_path))
    image_npz_keys = list(image_npz.keys())
    image = image_npz[image_npz_keys[0]]

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

    masks_npz = np.load(str(masks_file_path))
    masks_npz_keys = list(masks_npz.keys())
    masks = masks_npz[masks_npz_keys[0]]

    return masks


# TODO: these two methods could be the same.
def load_image_slice(image: np.array, slice_number: int) -> np.array:
    """
    Return a slice from a CT image, given its position. The slice is windowed
    to improve its contrast if needed, converted to greyscale, and expanded to
    RGB. It checks if the slice number exists.

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
    tuple with the Jaccard index and the Dice coefficient.

    :param original_mask: original segmentation mask.
    :param predicted_mask: predicted segmentation mask.

    :return: Jaccard index and the Dice coefficient of the masks provided.
    """

    logger.info('Compare original and predicted masks')
    logger.debug(f'compare_original_and_predicted_masks('
                 f'original_mask={original_mask.shape}, '
                 f'predicted_mask={predicted_mask.shape})')

    original_mask_as_bool = original_mask != 0
    predicted_mask_transformed = np.squeeze(predicted_mask)

    intersection = original_mask_as_bool * predicted_mask_transformed
    union = (original_mask_as_bool + predicted_mask_transformed) > 0

    jaccard = intersection.sum() / float(union.sum())
    dice = intersection.sum() * 2 / (original_mask_as_bool.sum() + predicted_mask.sum())

    return jaccard, dice


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
    jaccard_column = df_raw_data[JaccardKey]
    jaccard_results = {
        MetricKey: JaccardKey,
        MinKey: jaccard_column.min(),
        MaxKey: jaccard_column.max(),
        AverageKey: jaccard_column.mean(),
        StandardDeviationKey: jaccard_column.std()
    }
    dice_column = df_raw_data[DiceKey]
    dice_results = {
        MetricKey: DiceKey,
        MinKey: dice_column.min(),
        MaxKey: dice_column.max(),
        AverageKey: dice_column.mean(),
        StandardDeviationKey: dice_column.std()
    }

    results = [jaccard_results, dice_results]

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
                        apply_windowing: bool,
                        use_masks_contours: bool,
                        use_bounding_box: bool,
                        debug: Debug) -> dict:
    """
    Process a slice of the image. Returns the result of the analysis.

    :param sam_predictor: SAM predictor for image segmentation.
    :param image: array with the slices of the CT.
    :param masks: masks for each slice of the CT.
    :param slice_number: slice to work with.
    :param apply_windowing: if True, apply windowing to the image.
    :param use_masks_contours: if True, get positive prompts from contours.
    :param use_bounding_box: if True, include a bounding box in the prompts.
    :param debug: instance of Debug class.

    :return: a dictionary with the number of the slice been processed and the
    Jaccard index and Dice score between the ground truth and the prediction
    masks.
    """

    logger.info('Process image slice')
    logger.debug(f'process_image_slice('
                 f'sam_predictor={sam_predictor.device.type}, '
                 f'image={image.shape}, '
                 f'masks={masks.shape}, '
                 f'slice_number={slice_number}, '
                 f'apply_windowing={apply_windowing}, '
                 f'use_masks_contours={use_masks_contours}, '
                 f'use_bounding_box={use_bounding_box}, '
                 f'debug={debug.enabled})')

    points = load_image_slice(image=image, slice_number=slice_number)
    labeled_points = load_masks_slice(masks=masks, slice_number=slice_number)

    image_slice = ImageSlice(
        points=points,
        labeled_points=labeled_points,
        apply_windowing=apply_windowing,
        use_bounding_box=use_bounding_box,
        use_masks_contours=use_masks_contours)

    mask = []
    score = []
    jaccard = None
    dice = None

    if image_slice.labels.size > 1:
        point_coords = image_slice.get_point_coordinates()
        point_labels = image_slice.centers_labels
        if use_bounding_box:
            box = image_slice.get_box()
        else:
            box = None

        point_coords = np.append(point_coords, [[2970, 2096]], axis=0)
        point_coords = np.append(point_coords, [[2860, 2343]], axis=0)
        # point_coords = np.append(point_coords, [[2794, 2019]], axis=0)
        point_coords = np.append(point_coords, [[3196, 2701]], axis=0)
        point_coords = np.append(point_coords, [[2756, 2789]], axis=0)
        point_labels = np.append(point_labels, 0)
        point_labels = np.append(point_labels, 0)
        # point_labels = np.append(point_labels, 0)
        point_labels = np.append(point_labels, 0)
        point_labels = np.append(point_labels, 0)

        sam_predictor.set_image(image_slice.processed_points)
        if USE_BOUNDING_BOX:
            mask, score, logits = sam_predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                multimask_output=False)
        else:
            mask, score, logits = sam_predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=False)

        # Compare original and predicted lung masks
        jaccard, dice = compare_original_and_predicted_masks(
            original_mask=labeled_points, predicted_mask=mask)
    else:
        logger.info("There are no masks for the current slice")

    if debug.enabled:
        if image_slice.labels.size > 1:
            debug.set_slice_number(slice_number=slice_number)

            # Save SAM's prompt to YML
            prompts = dict()
            for index, contour_center in enumerate(point_coords):
                row = int(contour_center[1])
                column = int(contour_center[0])
                label = int(point_labels[index])
                prompts.update({
                    index: {
                        'row': row,
                        'column': column,
                        'label': label
                    }
                })

            if use_bounding_box:
                bounding_box = image_slice.get_box()
                prompts.update({
                    'bounding_box': {
                        'row_min': int(bounding_box[1]),
                        'colum_min': int(bounding_box[0]),
                        'row_max': int(bounding_box[3]),
                        'column_max': int(bounding_box[2])
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
            plt.imshow(image_slice.processed_points)
            if DEBUG_DRAW_SAM_PREDICTION:
                show_mask(mask, plt.gca())
            if DEBUG_DRAW_MASKS_CONTOURS:
                for mask_contour in image_slice.contours:
                    plt.plot(mask_contour[:, 1], mask_contour[:, 0], color='green', zorder=0)
            show_points(
                coords=np.flip(point_coords, axis=1),
                labels=point_labels,
                ax=plt.gca())
            if DEBUG_DRAW_BOUNDING_BOX:
                if use_bounding_box:
                    show_box(box=image_slice.get_box(), ax=plt.gca())
            plt.title(f"Score: {score[0]:.3f}", fontsize=18)
            plt.axis('off')

            debug_file_path = debug.get_file_path('prediction', '.png')
            figure.savefig(debug_file_path, bbox_inches='tight')
            plt.close()

    result = {
        SliceNumberKey: slice_number,
        JaccardKey: jaccard,
        DiceKey: dice
    }

    return result


def process_image(sam_predictor: SamPredictor,
                  image: np.array,
                  masks: np.array,
                  apply_windowing: bool,
                  use_bounding_box: bool,
                  use_masks_contours: bool,
                  debug: Debug) -> Tuple[Path, Path]:
    """
    Process all the slices of a given image. Saves the result as two CSV files,
    one with each slice's result, another with a statistical summary. Returns
    the paths where the resulting CSV files will be stored.

    :param sam_predictor: SAM predictor for image segmentation.
    :param image: array with the slices of the CT.
    :param masks: masks for each slice of the CT.
    :param apply_windowing: if True, apply windowing to the image.
    :param use_masks_contours: if True, get positive prompts from contours.
    :param use_bounding_box: if True, include a bounding box in the prompts.
    :param debug: instance of Debug class.

    :return: paths where the resulting CSV files are stored.
    """

    logger.info('Process image')
    logger.debug(f'process_image('
                 f'sam_predictor={sam_predictor.device.type}, '
                 f'image={image.shape}, '
                 f'masks={masks.shape}, '
                 f'apply_windowing={apply_windowing}, '
                 f'use_masks_contours={use_masks_contours}, '
                 f'use_bounding_box={use_bounding_box}, '
                 f'debug={debug.enabled})')

    items = image.shape[-1]
    progress_bar = tqdm(desc='Processing CT image slices', total=items)

    results = []
    for slice_number in range(items):
        result = process_image_slice(sam_predictor=sam_predictor,
                                     image=image,
                                     masks=masks,
                                     slice_number=slice_number,
                                     apply_windowing=apply_windowing,
                                     use_masks_contours=use_masks_contours,
                                     use_bounding_box=use_bounding_box,
                                     debug=debug)
        results.append(result)
        progress_bar.update()
    progress_bar.close()

    output_path = debug.image_file_path.parent / Path('results') / Path(debug.image_file_path.stem)
    results_path, raw_data_path = save_results(output_path, results)

    return results_path, raw_data_path


def parse_arguments() -> Tuple[Path, Path, int, bool, bool, bool, bool, bool]:
    """
    Parse arguments passed via command line, returning them formatted. Adequate
    defaults are provided when possible.

    :return: path of the image file, path of the masks file, slice to work
    with, perform windowing on the image slice, use mask contours to get the
    positive point prompts, use a bounding box as a prompt, dry run option,
    debug option.
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
    argument_parser.add_argument('-w', '--apply_windowing',
                                 action='store_true',
                                 help='apply windowing to the image')
    argument_parser.add_argument('-c', '--use_masks_contours',
                                 action='store_true',
                                 help='get positive prompts from contours')
    argument_parser.add_argument('-b', '--use_bounding_box',
                                 action='store_true',
                                 help='include a bounding box in the prompts')
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
    apply_windowing = arguments.apply_windowing
    use_masks_contours = arguments.use_masks_contours
    use_bounding_box = arguments.use_bounding_box
    dry_run = arguments.dry_run
    debug = arguments.debug

    return Path(image_file_path), Path(masks_file_path), \
        slice_number, apply_windowing, use_masks_contours, use_bounding_box, \
        dry_run, debug


def get_summary(
        image_file_path: Path,
        masks_file_path: Path,
        image: np.array,
        masks: np.array,
        slice_number: int,
        apply_windowing: bool,
        use_masks_contours: bool,
        use_bounding_box: bool,
        dry_run: bool,
        debug: Debug
) -> str:
    """
    Show a summary of the actions this script will perform.

    :param image_file_path: path of the image file.
    :param masks_file_path: path of the masks file.
    :param image: array with the CT volume slices.
    :param masks: array with the CT volume masks.
    :param slice_number: slice to work with.
    :param apply_windowing: if True, apply windowing to the image.
    :param use_masks_contours: if True, get positive prompts from contours.
    :param use_bounding_box: if True, include a bounding box in the prompts.
    :param dry_run: if True, the actions will not be performed.
    :param debug: instance of Debug class.

    :return: summary of the actions this script will perform.
    """

    logger.info('Get summary')
    logger.debug(f'get_summary('
                 f'image_file_path="{image_file_path}", '
                 f'masks_file_path="{masks_file_path}", '
                 f'image={image.shape}, '
                 f'masks={masks.shape}, '
                 f'slice_number={slice_number}, '
                 f'apply_windowing={apply_windowing}, '
                 f'use_masks_contours={use_masks_contours}, '
                 f'use_bounding_box={use_bounding_box}, '
                 f'debug={debug}, '
                 f'dry_run={dry_run})')

    image_slices = image.shape[-1]

    masks_slices = masks.shape[-1]

    if slice_number is not None:
        requested_slice_in_range = slice_number < image_slices
        slice_information = f'Slice: {slice_number}'
    else:
        requested_slice_in_range = None
        slice_information = 'Process all slices'

    summary = f'- Image file path: "{image_file_path}"\n' \
              f'- Masks file path: "{masks_file_path}"\n' \
              f'- {slice_information}\n' \
              f'- Apply windowing: {apply_windowing}\n' \
              f'- Use masks contours: {use_masks_contours}\n' \
              f'- Use bounding box: {use_bounding_box}\n' \
              f'- Debug: {debug.enabled}\n' \
              f'- Dry run: {dry_run}\n' \
              f'- Image slices: {image_slices}\n' \
              f'- Masks slices: {masks_slices}\n' \
              f'- Equal number of slices: {image_slices == masks_slices}'

    if requested_slice_in_range is not None:
        summary += f'\n' \
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

    image_file_path, masks_file_path, slice_number, \
        apply_windowing, use_masks_contours, use_bounding_box, \
        dry_run, debug_enabled = parse_arguments()

    debug = Debug(
        enabled=debug_enabled,
        image_file_path=image_file_path,
        masks_file_path=masks_file_path)

    image = load_image(image_file_path=image_file_path)
    masks = load_masks(masks_file_path=masks_file_path)

    summarizer.summary = get_summary(
        image_file_path=image_file_path,
        masks_file_path=masks_file_path,
        image=image,
        masks=masks,
        slice_number=slice_number,
        apply_windowing=apply_windowing,
        use_masks_contours=use_masks_contours,
        use_bounding_box=use_bounding_box,
        debug=debug,
        dry_run=dry_run)

    if dry_run:
        print()
        print('[bold]Summary:[/bold]')
        print(summarizer.summary)
        return

    sam_predictor = get_sam_predictor(SamModel.ViT_L)

    if slice_number is None:
        result = process_image(sam_predictor=sam_predictor,
                               image=image,
                               masks=masks,
                               apply_windowing=apply_windowing,
                               use_masks_contours=use_masks_contours,
                               use_bounding_box=use_bounding_box,
                               debug=debug)
        print(f'Results saved to: "{str(result[0])}"')
        print(f'Raw data saved to: "{str(result[1])}"')
    else:
        result = process_image_slice(sam_predictor=sam_predictor,
                                     image=image,
                                     masks=masks,
                                     slice_number=slice_number,
                                     apply_windowing=apply_windowing,
                                     use_masks_contours=use_masks_contours,
                                     use_bounding_box=use_bounding_box,
                                     debug=debug)
        print(f'Jaccard index: {result[JaccardKey]:.4f}')
        print(f'Dice score: {result[DiceKey]:.4f}')

    print(summarizer.notification_message)


if __name__ == '__main__':
    main()
