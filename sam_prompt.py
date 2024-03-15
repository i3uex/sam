import logging
from json import JSONEncoder
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class SAMPrompt:
    """
    Prompt data provided to SAM to guide the segmentation process. It contains
    all the details needed for reproducibility.

    The image, masks, and slice number describe the source slice where the
    segmentation is going to be applied.

    The prompt itself is composed of a list of points (positive and negative)
    and a bounding box.
    """

    ImageFilePathKey = 'image_file_path'
    MasksFilePathKey = 'masks_file_path'
    SliceNumberKey = 'slice_number'
    PromptKey = 'prompt'
    PointsCoordsKey = 'points_coords'
    PointCoordsRowKey = 'row'
    PointCoordsColumnKey = 'column'
    PointCoordsLabelKey = 'label'
    PointLabelsKey = 'points_labels'
    BoundingBoxKey = 'bounding_box'
    BoundingBoxUpperLeftCornerKey = 'upper_left_corner'
    BoundingBoxLowerRightCornerKey = 'bottom_right_corner'

    PointLabelPositive = 1
    PointLabelNegative = 0

    image_file_path: Path
    masks_file_path: Path
    slice_number: int
    points_coords: np.ndarray
    points_labels: np.ndarray
    boxes: np.ndarray

    def __init__(
            self,
            image_file_path: Path,
            masks_file_path: Path,
            slice_number: int = None,
            points_cords: np.ndarray = None,
            points_labels: np.ndarray = None,
            bounding_box: np.ndarray = None
    ):
        """
        Init Prompt class instance.

        :param image_file_path: path to the image file. It could be composed of
        multiple slices.
        :param masks_file_path: path to the masks file. It could be composed of
        multiple slices.
        :param slice_number: number of the slice this prompt correspond to. If
        None, the image and the masks are not divided into slices.
        :param points_cords: list of point coordinates for the prompt.
        :param points_labels: labels for the point coordinates (positive or
        negative).
        :param bounding_box: bounding box for the prompt.
        """

        logger.info('Init Prompt')
        logger.debug(f'Prompt.__init__('
                     f'image_file_path="{image_file_path}", '
                     f'masks_file_path="{masks_file_path}", '
                     f'slice_number={slice_number}, '
                     f'points_cords={points_cords}, '
                     f'points_labels={points_labels}, '
                     f'bounding_box={bounding_box})')

        self.image_file_path = image_file_path
        self.masks_file_path = masks_file_path
        self.slice_number = slice_number
        self.points_coords = points_cords
        self.points_labels = points_labels
        self.bounding_box = bounding_box

    def to_dictionary(self):
        """
        Store this class data into a dictionary.

        :return: a dictionary with the attributes of the class instance.
        """

        logger.info('Store class data into a dictionary')
        logger.debug('to_dictionary()')

        dictionary = {
            self.ImageFilePathKey: str(self.image_file_path),
            self.MasksFilePathKey: str(self.masks_file_path)
        }

        if self.slice_number is not None:
            dictionary[self.SliceNumberKey] = self.slice_number

        prompt_dictionary = {}
        points_coords_list = []
        if self.points_coords is not None and len(self.points_coords) > 0:
            for index, point_coords in enumerate(self.points_coords):
                points_coords_list.append({
                    self.PointCoordsRowKey: int(point_coords[0]),
                    self.PointCoordsColumnKey: int(point_coords[1]),
                    self.PointCoordsLabelKey: int(self.points_labels[index])
                })
            prompt_dictionary[self.PointsCoordsKey] = points_coords_list

        if self.bounding_box is not None:
            bounding_box = {
                self.BoundingBoxUpperLeftCornerKey: {
                    self.PointCoordsRowKey: int(self.bounding_box[0]),
                    self.PointCoordsColumnKey: int(self.bounding_box[1])
                },
                self.BoundingBoxLowerRightCornerKey: {
                    self.PointCoordsRowKey: int(self.bounding_box[2]),
                    self.PointCoordsColumnKey: int(self.bounding_box[3])
                }
            }
            prompt_dictionary[self.BoundingBoxKey] = bounding_box

        if len(prompt_dictionary.keys()) > 0:
            dictionary[self.PromptKey] = prompt_dictionary

        return dictionary


class SAMPromptJSONEncoder(JSONEncoder):
    """
    JSON encoder for the SAMPrompt class.
    """
    def default(self, sam_prompt: SAMPrompt):
        """
        Encode class data into a dictionary for JSON serialization.

        :param sam_prompt: SAMPrompt class instance.
        :return: class instance data as a dictionary.
        """

        logger.info('Encode class data into a dictionary for JSON serialization')
        logger.debug(f'default('
                     f'sam_prompt="{sam_prompt}")')

        return sam_prompt.to_dictionary()
