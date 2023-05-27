"""
Create a GIF from the images in a given folder.
"""

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import imageio
from PIL import Image
from tqdm import tqdm

from argparse_helper import ArgumentParserHelper

ResizeFactor = 0.50
TempPath = Path('temp')
SourceImagesPattern = '*_prediction.png'
TempImageSuffix = '_small.png'
TempImagesPattern = f'*{TempImageSuffix}'
ResultFileName = 'images.gif'

log = logging.getLogger(__name__)


def parse_arguments() -> Path:
    """
    Parse arguments passed via command line, returning them formatted. Adequate
    defaults are provided when possible.

    :return: relative path to the folder where the images are stored.
    """

    log.info('Get script arguments')
    log.debug('parse_arguments()')

    program_description = 'GIF Generation'
    argument_parser = argparse.ArgumentParser(description=program_description)
    argument_parser.add_argument('-i', '--images', required=True,
                                 help='path to the folder containing images')

    arguments = argument_parser.parse_args()
    images_path = ArgumentParserHelper.parse_dir_path_segment(arguments.images)

    return Path(images_path)


def create_gif(images_path: Path):
    """
    Create a GIF from a list of images.

    :param images_path: path to the samples to use for the GIF.
    """

    log.info('Create GIF')
    log.debug(f'create_gif('
              f'images_path={images_path})')

    executor = ProcessPoolExecutor()

    # Create small version of images
    images_paths = sorted(images_path.glob(SourceImagesPattern))
    items = len(images_paths)

    temp_path = images_path / TempPath
    temp_path.mkdir(parents=True, exist_ok=True)

    progressbar = tqdm(desc='Reducing images', total=items)

    futures = []
    for image_path in images_paths:
        future = executor.submit(
            reduce_image,
            image_path)
        futures.append(future)

    for future in futures:
        future.result()
        progressbar.update()
    progressbar.close()

    # Create GIF from small images
    progressbar = tqdm(desc='Creating GIF', total=items)

    images = []
    image_small_paths = sorted(temp_path.glob(TempImagesPattern))
    items = len(image_small_paths)

    for image_small_path in image_small_paths:
        images.append(imageio.v2.imread(image_small_path))
        progressbar.update()
    progressbar.close()

    output_path = images_path / ResultFileName
    imageio.mimsave(str(output_path), images)

    # Remove small version of images
    progressbar = tqdm(desc='Removing reduced images', total=items)

    futures = []
    for image_small_path in image_small_paths:
        future = executor.submit(
            image_small_path.unlink)
        futures.append(future)

    for future in futures:
        future.result()
        progressbar.update()
    progressbar.close()

    temp_path.rmdir()


def reduce_image(image_path: Path):
    """
    Reduce image by a given factor, save the result with the same name and a
    suffix.

    :param image_path: path of the image to be reduced.
    """

    log.info('Reduce image')
    log.debug(f'reduce_image('
              f'image_path="{image_path}")')

    image = Image.open(str(image_path))
    resize_factor = ResizeFactor
    new_image_size = int(image.size[0] * resize_factor), int(image.size[1] * resize_factor)
    image_resized = image.resize(new_image_size)
    output_filename = f'{image_path.stem}{TempImageSuffix}'
    output_path = image_path.parent / TempPath / output_filename
    image_resized.save(output_path)


def main():
    """
    Set logging up, parse arguments, process data, and save result.
    """

    logging.basicConfig(
        filename='debug.log',
        level=logging.DEBUG,
        format='%(asctime)-15s %(levelname)8s %(name)s %(message)s')
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.getLogger('PIL.PngImagePlugin').disabled = True

    log.info('GIF generation process')
    log.debug('main()')

    images_path = parse_arguments()
    create_gif(images_path)


if __name__ == '__main__':
    main()
