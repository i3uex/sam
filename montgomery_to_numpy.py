"""
Save Montgomery dataset X-rays and masks as NumPy arrays.
"""

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Tuple, List

import numpy as np
from PIL import Image
from rich import print
from tqdm import tqdm

from tools.argparse_helper import ArgumentParserHelper
from tools.summarizer import Summarizer

logger = logging.getLogger(__name__)

LoggingEnabled = True

DatasetPath = Path('datasets/montgomery')
XRaysPath = Path('CXR_png')
LeftLungMasks = Path('ManualMask/leftMask')
RightLungMasks = Path('ManualMask/rightMask')
ImagesPattern = '*.png'


def parse_arguments() -> Tuple[Path, Path, bool]:
    """
    Parse arguments passed via command line, returning them formatted. Adequate
    defaults are provided when possible.

    :return: path of the input folder, path of the output folder, dry run
    option.
    """

    logger.info('Get script arguments')
    logger.debug('parse_arguments()')

    program_description = 'Save Montgomery X-rays and masks to NumPy'
    argument_parser = argparse.ArgumentParser(description=program_description)
    argument_parser.add_argument('-i', '--input_folder_path',
                                 required=True,
                                 help='path to Montgomery input folder')
    argument_parser.add_argument('-o', '--output_folder_path',
                                 required=True,
                                 help='path to NumPy output folder')
    argument_parser.add_argument('-n', '--dry_run',
                                 action='store_true',
                                 help='show what would be done, do not do it')

    arguments = argument_parser.parse_args()
    input_folder_path_segment = ArgumentParserHelper.parse_dir_path_segment(
        arguments.input_folder_path)
    output_folder_path_segment = arguments.output_folder_path
    dry_run = arguments.dry_run
    return Path(input_folder_path_segment), Path(output_folder_path_segment), \
        dry_run


def get_summary(
        input_folder_path: Path,
        output_folder_path: Path,
        dry_run: bool
) -> str:
    """
    Show a summary of the actions this script will perform.

    :param input_folder_path: path to the input folder where the images to save
    as NumPy are.
    :param output_folder_path: path to the output directory.
    :param dry_run: if True, the actions will not be performed.

    :return: summary of the actions this script will perform.
    """

    logger.info('Get summary')
    logger.debug(f'get_summary('
                 f'input_folder_path="{input_folder_path}", '
                 f'output_folder_path="{output_folder_path}", '
                 f'dry_run={dry_run})')

    summary = f'- Input folder path: "{input_folder_path}"\n' \
              f'- Output folder path: "{output_folder_path}"\n' \
              f'- Dry run: {dry_run}'

    return summary


def process_png(
        output_folder_path: Path,
        x_ray_path: Path,
        left_lung_mask_path: Path,
        right_lung_mask_path: Path
):
    """
    Process a single PNG X-ray, with its masks. Store the result as a
    compressed NumPy array in two separate files, one for the X-ray, another
    for the masks.

    :param output_folder_path: path to the output directory.
    :param x_ray_path: paths for the X-ray image.
    :param left_lung_mask_path: path for the left lung mask.
    :param right_lung_mask_path:  path for the right lung mask.
    """

    logger.info('Process a single PNG X-ray with its masks')
    logger.debug(f'process_png('
                 f'output_folder_path="{output_folder_path}", '
                 f'x_ray_path="{x_ray_path}", '
                 f'left_lung_mask_path="{left_lung_mask_path}", '
                 f'right_lung_mask_path="{right_lung_mask_path}")')

    # Save X-ray as NumPy
    x_ray_image = Image.open(x_ray_path)
    x_ray_npy = np.asarray(x_ray_image, dtype=np.int16)
    x_ray_npy = np.expand_dims(x_ray_npy, axis=-1)
    x_ray_npy_name = f"image_{x_ray_path.stem}.npz"
    x_ray_npy_path = output_folder_path / Path(x_ray_npy_name)
    np.savez_compressed(str(x_ray_npy_path), x_ray_npy)

    # Combine left and right lung masks
    left_lung_masks_image = Image.open(left_lung_mask_path)
    left_lung_masks_npy = np.asarray(left_lung_masks_image, dtype=np.int16).copy()
    left_lung_masks_npy = np.expand_dims(left_lung_masks_npy, axis=-1)
    left_lung_masks_npy[left_lung_masks_npy == 1] = 1


    right_lung_masks_image = Image.open(right_lung_mask_path)
    right_lung_masks_npy = np.asarray(right_lung_masks_image, dtype=np.int16).copy()
    right_lung_masks_npy = np.expand_dims(right_lung_masks_npy, axis=-1)
    right_lung_masks_npy[right_lung_masks_npy == 1] = 2

    lung_masks_npy = np.add(left_lung_masks_npy, right_lung_masks_npy)
    lung_masks_npy_name = f"masks_{x_ray_path.stem}.npz"
    lung_masks_npy_path = output_folder_path / Path(lung_masks_npy_name)
    np.savez_compressed(str(lung_masks_npy_path), lung_masks_npy)


def png_to_numpy_synchronously(
        input_folder_path: Path,
        output_folder_path: Path,
        x_rays_paths: List[Path],
        left_lung_masks_paths: List[Path],
        right_lung_masks_paths: List[Path]
):
    """
    Save PNG X-rays and masks as NumPy, synchronously.

    :param input_folder_path: path to the input folder where the images to save
    as NumPy are.
    :param output_folder_path: path to the output directory.
    :param x_rays_paths: list of paths for each X-ray image.
    :param left_lung_masks_paths: list of paths for each left lung mask.
    :param right_lung_masks_paths: list of paths for each right lung mask.
    """

    logger.info('Save PNG files as NumPy, synchronously')
    logger.debug(f'png_to_numpy_synchronously('
                 f'input_folder_path="{input_folder_path}", '
                 f'output_folder_path="{output_folder_path}", '
                 f'x_rays_paths={len(x_rays_paths)} items, '
                 f'left_lung_masks_paths={len(left_lung_masks_paths)} items, '
                 f'right_lung_masks_paths={len(right_lung_masks_paths)} items)')

    len_x_rays_paths = len(x_rays_paths)

    progressbar = tqdm(desc="PNG to NumPy", total=len_x_rays_paths)
    for index, x_ray_path in enumerate(x_rays_paths):
        process_png(
            output_folder_path=output_folder_path,
            x_ray_path=x_ray_path,
            left_lung_mask_path=left_lung_masks_paths[index],
            right_lung_mask_path=right_lung_masks_paths[index])
        progressbar.update()

    progressbar.close()


def png_to_numpy_asynchronously(
        input_folder_path: Path,
        output_folder_path: Path,
        x_rays_paths: List[Path],
        left_lung_masks_paths: List[Path],
        right_lung_masks_paths: List[Path]
):
    """
    Save PNG X-rays and masks as NumPy, asynchronously.

    :param input_folder_path: path to the input folder where the images to save
    as NumPy are.
    :param output_folder_path: path to the output directory.
    :param x_rays_paths: list of paths for each X-ray image.
    :param left_lung_masks_paths: list of paths for each left lung mask.
    :param right_lung_masks_paths: list of paths for each right lung mask.
    """

    logger.info('Save PNG files as NumPy, synchronously')
    logger.debug(f'png_to_numpy_synchronously('
                 f'input_folder_path="{input_folder_path}", '
                 f'output_folder_path="{output_folder_path}", '
                 f'x_rays_paths={len(x_rays_paths)} items, '
                 f'left_lung_masks_paths={len(left_lung_masks_paths)} items, '
                 f'right_lung_masks_paths={len(right_lung_masks_paths)} items)')

    executor = ProcessPoolExecutor()

    futures = []
    for index, x_ray_path in enumerate(x_rays_paths):
        future = executor.submit(
            process_png,
            output_folder_path,
            x_ray_path,
            left_lung_masks_paths[index],
            right_lung_masks_paths[index])
        futures.append(future)

    len_x_rays_paths = len(x_rays_paths)
    progressbar = tqdm(desc="PNG to NumPy", total=len_x_rays_paths)

    for future in futures:
        future.result()
        progressbar.update()

    progressbar.close()


def png_to_numpy(
        input_folder_path: Path,
        output_folder_path: Path,
        asynchronous: bool = True
):
    """
    Save PNG X-rays and masks as NumPy.

    :param input_folder_path: path to the input folder where the images to save
    as NumPy are.
    :param output_folder_path: path to the output directory.
    :param asynchronous: if True, each conversion iteration will be launched
    asynchronously, taking advantage of the different cores available in the
    system. If false, the conversion iterations will be performed synchronously,
    taking more time, but it will be easier to debug.
    """

    logger.info('Save PNG files as NumPy')
    logger.debug(f'png_to_numpy('
                 f'input_folder_path="{input_folder_path}", '
                 f'output_folder_path="{output_folder_path}", '
                 f'asynchronous={asynchronous})')

    # Get the list of files in X-rays, left lung and right lung folders
    x_rays_path = DatasetPath / XRaysPath
    x_rays_paths = sorted(x_rays_path.glob(ImagesPattern))
    left_lung_masks_path = DatasetPath / LeftLungMasks
    left_lung_masks_paths = sorted(left_lung_masks_path.glob(ImagesPattern))
    right_lung_masks_path = DatasetPath / RightLungMasks
    right_lung_masks_paths = sorted(right_lung_masks_path.glob(ImagesPattern))

    # Check the number of X-rays, left lung masks and right lung masks are
    # the same
    len_x_rays_paths = len(x_rays_paths)
    len_left_lung_masks_paths = len(left_lung_masks_paths)
    len_right_lung_masks_paths = len(right_lung_masks_paths)

    assert len_x_rays_paths == len_left_lung_masks_paths, \
        f"The number of X-rays ({len_x_rays_paths}) is no the same than " \
        f"the number of left lung masks ({len_left_lung_masks_paths})"
    assert len_x_rays_paths == len_right_lung_masks_paths, \
        f"The number of X-rays ({len_x_rays_paths}) is no the same than " \
        f"the number of right lung masks ({len_right_lung_masks_paths})"

    if not asynchronous:
        png_to_numpy_synchronously(
            input_folder_path=input_folder_path,
            output_folder_path=output_folder_path,
            x_rays_paths=x_rays_paths,
            left_lung_masks_paths=left_lung_masks_paths,
            right_lung_masks_paths=right_lung_masks_paths)
    else:
        png_to_numpy_asynchronously(
            input_folder_path=input_folder_path,
            output_folder_path=output_folder_path,
            x_rays_paths=x_rays_paths,
            left_lung_masks_paths=left_lung_masks_paths,
            right_lung_masks_paths=right_lung_masks_paths)


def main():
    """
    Set logging up, parse arguments, and process data.
    """

    if LoggingEnabled:
        logging.basicConfig(
            filename='debug.log',
            level=logging.DEBUG,
            format='%(asctime)-15s %(levelname)8s %(name)s %(message)s')

    logger.info('Start conversion process')
    logger.debug('main()')

    summarizer = Summarizer()

    input_folder_path, output_folder_path, dry_run = parse_arguments()

    summarizer.summary = get_summary(
        input_folder_path=input_folder_path,
        output_folder_path=output_folder_path,
        dry_run=dry_run)

    if dry_run:
        print()
        print('[bold]Summary:[/bold]')
        print(summarizer.summary)
        return

    if not output_folder_path.exists():
        output_folder_path.mkdir(parents=True, exist_ok=True)

    png_to_numpy(
        input_folder_path=input_folder_path,
        output_folder_path=output_folder_path,
        asynchronous=True)

    print(summarizer.notification_message)


if __name__ == '__main__':
    main()
