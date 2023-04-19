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


def parse_arguments() -> Tuple[Path, Path, bool, bool, bool]:
    """
    Parse arguments passed via command line, returning them formatted. Adequate
    defaults are provided when possible.

    :return: path of the input file, path of the output file, should the image
    be transposed, should the image be rotated 180º, dry run option.
    :rtype: Tuple[Path, Path, bool, bool, bool]
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
    argument_parser.add_argument('-t', '--transpose',
                                 action='store_true',
                                 help='should the image be transposed')
    argument_parser.add_argument('-r', '--rotate',
                                 action='store_true',
                                 help='should the image be rotated 180º')
    argument_parser.add_argument('-n', '--dry_run',
                                 action='store_true',
                                 help='show what would be done, do not do it')

    arguments = argument_parser.parse_args()
    input_file_path_segment = ArgumentParserHelper.parse_file_path_segment(
        arguments.input_file_path)
    output_file_path_segment = arguments.output_file_path
    transpose = arguments.transpose
    rotate = arguments.rotate
    dry_run = arguments.dry_run
    return Path(input_file_path_segment), Path(output_file_path_segment), \
        transpose, rotate, dry_run


def get_summary(
        input_file_path: Path,
        output_file_path: Path,
        transpose: bool,
        rotate: bool,
        dry_run: bool
) -> str:
    """
    Show a summary of the actions this script will perform.

    :param input_file_path: path to the input NIfTI file to save as NumPy.
    :param output_file_path: path to the output NumPy file.
    :param transpose: if True, the image will be transposed.
    :param rotate: if True, the image will be rotated.
    :param dry_run: if True, the actions will not be performed.

    :return: summary of the actions this script will perform.
    :rtype: str
    """

    logger.info('Get summary')
    logger.debug(f'get_summary('
                 f'input_file_path="{input_file_path}", '
                 f'output_file_path="{output_file_path}", '
                 f'transpose={transpose}, '
                 f'rotate={rotate}, '
                 f'dry_run={dry_run})')

    summary = f'- Input file path: "{input_file_path}"\n' \
              f'- Output file path: "{output_file_path}"\n' \
              f'- Transpose the image: {transpose}\n' \
              f'- Rotate the image: {rotate}\n' \
              f'- Dry run: {dry_run}'

    return summary


def nifty_to_numpy(
        input_file_path: Path,
        output_file_path: Path,
        transpose: bool,
        rotate: bool
):
    """
    Save NIfTI file as NumPy.

    :param input_file_path: path to the input NIfTI file to save as NumPy.
    :param output_file_path: path to the output NumPy file.
    :param transpose: if True, the image will be transposed.
    :param rotate: if True, the image will be rotated.

    """

    logger.info('Save NIfTI file as NumPy')
    logger.debug(f'nifty_to_numpy('
                 f'input_file_path="{input_file_path}", '
                 f'output_file_path="{output_file_path}", '
                 f'transpose={transpose}, '
                 f'rotate={rotate})')

    image = nib.load(input_file_path)
    logger.info(f'Voxel axes orientations: {nib.aff2axcodes(image.affine)}')
    image = image.get_fdata()
    logger.info(f'Image shape: {image.shape}')

    if transpose:
        logger.info('Transpose the image')
        image = image.swapaxes(0, 1)

    if rotate:
        logger.info('Rotate the image')
        image = np.rot90(image, k=2, axes=(0, 1))

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

    input_file_path, output_file_path, transpose, rotate, dry_run = parse_arguments()

    summarizer.summary = get_summary(
        input_file_path=input_file_path,
        output_file_path=output_file_path,
        transpose=transpose,
        rotate=rotate,
        dry_run=dry_run)

    if dry_run:
        print()
        print('[bold]Summary:[/bold]')
        print(summarizer.summary)
        return

    nifty_to_numpy(
        input_file_path=input_file_path,
        output_file_path=output_file_path,
        transpose=transpose,
        rotate=rotate)

    print(summarizer.notification_message)


if __name__ == '__main__':
    main()
