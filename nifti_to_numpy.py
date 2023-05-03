"""
Save NIfTI images as NumPy arrays.
"""

import argparse
import logging
from pathlib import Path
from typing import Tuple

import nibabel as nib
import numpy as np
from rich import print

from tools.argparse_helper import ArgumentParserHelper
from tools.summarizer import Summarizer

logger = logging.getLogger(__name__)

LoggingEnabled = True

DatasetPath = Path('datasets/zenodo')


def parse_arguments() -> Tuple[Path, Path, bool, bool]:
    """
    Parse arguments passed via command line, returning them formatted. Adequate
    defaults are provided when possible.

    :return: path of the input file, path of the output file, should the image
    axes be swapped, dry run option.
    """

    logger.info('Get script arguments')
    logger.debug('parse_arguments()')

    program_description = 'Save NIfTI to NumPy'
    argument_parser = argparse.ArgumentParser(description=program_description)
    argument_parser.add_argument('-i', '--input_file_path',
                                 required=True,
                                 help='path to NIfTI input file')
    argument_parser.add_argument('-o', '--output_file_path',
                                 required=True,
                                 help='path to NumPy output file')
    argument_parser.add_argument('-s', '--swap_axes',
                                 action='store_true',
                                 help='should the image axes be swapped')
    argument_parser.add_argument('-n', '--dry_run',
                                 action='store_true',
                                 help='show what would be done, do not do it')

    arguments = argument_parser.parse_args()
    input_file_path_segment = ArgumentParserHelper.parse_file_path_segment(
        arguments.input_file_path)
    output_file_path_segment = arguments.output_file_path
    swap_axes = arguments.swap_axes
    dry_run = arguments.dry_run
    return Path(input_file_path_segment), Path(output_file_path_segment), \
        swap_axes, dry_run


def get_summary(
        input_file_path: Path,
        output_file_path: Path,
        swap_axes: bool,
        dry_run: bool
) -> str:
    """
    Show a summary of the actions this script will perform.

    :param input_file_path: path to the input NIfTI file to save as NumPy.
    :param output_file_path: path to the output NumPy file.
    :param swap_axes: if True, the image axes will be swapped.
    :param dry_run: if True, the actions will not be performed.

    :return: summary of the actions this script will perform.
    """

    logger.info('Get summary')
    logger.debug(f'get_summary('
                 f'input_file_path="{input_file_path}", '
                 f'output_file_path="{output_file_path}", '
                 f'swap_axes={swap_axes}, '
                 f'dry_run={dry_run})')

    summary = f'- Input file path: "{input_file_path}"\n' \
              f'- Output file path: "{output_file_path}"\n' \
              f'- Swap axes: {swap_axes}\n' \
              f'- Dry run: {dry_run}'

    return summary


def nifty_to_numpy(
        input_file_path: Path,
        output_file_path: Path,
        swap_axes: bool
):
    """
    Save NIfTI file as NumPy.

    :param input_file_path: path to the input NIfTI file to save as NumPy.
    :param output_file_path: path to the output NumPy file.
    :param swap_axes: if True, the image axes will be swapped.
    """

    logger.info('Save NIfTI file as NumPy')
    logger.debug(f'nifty_to_numpy('
                 f'input_file_path="{input_file_path}", '
                 f'output_file_path="{output_file_path}", '
                 f'swap_axes={swap_axes})')

    image = nib.load(input_file_path)
    logger.info(f'Voxel axes orientations: {nib.aff2axcodes(image.affine)}')
    image = image.get_fdata()
    logger.info(f'Image shape: {image.shape}')

    if swap_axes:
        logger.info('Swap image axes')
        image = image.swapaxes(0, 1)

    logger.info('Save the NIfTI image as NumPy')
    np.save(str(output_file_path), image)


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

    input_file_path, output_file_path, swap_axes, dry_run = parse_arguments()

    summarizer.summary = get_summary(
        input_file_path=input_file_path,
        output_file_path=output_file_path,
        swap_axes=swap_axes,
        dry_run=dry_run)

    if dry_run:
        print()
        print('[bold]Summary:[/bold]')
        print(summarizer.summary)
        return

    nifty_to_numpy(
        input_file_path=input_file_path,
        output_file_path=output_file_path,
        swap_axes=swap_axes)

    print(summarizer.notification_message)


if __name__ == '__main__':
    main()
