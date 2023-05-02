import logging
import random
from typing import Tuple

import numpy as np
import scipy.ndimage
from skimage import measure
from skimage.draw import polygon

from tools.point_location import PointLocation

logger = logging.getLogger(__name__)


class Masks:
    """
    Process a NumPy array containing a series of masks over a CT slice.

    Each mask marks a section of the slice where a given object is present. The
    value 0 means no object was present in that location. Every value greater
    than 0 marks a different object.

    This class processes those masks and calculates varios features of each of
    them.

    First, the contours of each mask. A given mask can have multiple contours.

    Second, the center of each contour. The center of mass is used as a seed.
    If it falls outside the contour, it is taken inside.

    Third, the bounding boxes for each contour.
    """

    __points: np.array = None
    __mask_indexes: np.array = None

    contours: list = None  # coordinates of the polygon surrounding an object
    contours_masks: list = None  # filled contours
    contours_bounding_boxes: list = None  # top left, bottom right coordinates
    contours_centers: list = None  # point inside each contour

    def __init__(self, points: np.array):
        """
        Init Masks class instance with an array of values. The value 0 means no
        object was present in that location. Every value greater than 0 marks a
        different object.

        :param points: array with the values of the masks.
        """

        logger.info('Init Masks')
        logger.debug(f'Masks.__init__('
                     f'points={points.shape})')

        self.__points = points
        self.__process_points()

    def __process_points(self):
        """
        If the points provided to the class contains other values than 0, find
        the contours. For each contour found, find their masks, their centers
        and their bounding boxes.
        """

        logger.info('Process points')
        logger.debug('__process_points()')

        self.__masks_indexes = np.unique(self.__points)
        if len(self.__masks_indexes) > 1:
            self.__find_contours()
            self.__find_contours_masks()
            self.__find_contours_centers()
            self.__find_contours_bounding_boxes()

    def __find_contours(self):
        """
        Find contours for every point value other than 0.
        """

        logger.info('Find contours')
        logger.debug('__find_contours()')

        mask_indexes_len = len(self.__masks_indexes)
        if mask_indexes_len > 1:
            logger.info(f'Finding contours for {mask_indexes_len} point values.')
            self.contours = []
            for mask_index in self.__masks_indexes:
                if mask_index != 0:
                    mask = self.__points == mask_index
                    mask_contours = measure.find_contours(mask)
                    self.contours += mask_contours
        else:
            logger.info('There are no contours to find.')

    def __find_contours_masks(self):
        """
        Find masks for every contour. A mask is an array with the same shape
        that the points provided, filled with False. The True values are inside
        the polygon defined by the contour.
        """

        logger.info('Find contours masks')
        logger.debug('__find_contours_masks()')

        contours_len = len(self.contours)
        if contours_len > 1:
            logger.info(f'Finding masks for {contours_len} contours.')
            self.contours_masks = []
            for contour in self.contours:
                contour_mask = np.zeros(
                    (self.__points.shape[0], self.__points.shape[1]), 'uint8')
                rows, columns = polygon(
                    contour[:, 1], contour[:, 0], contour_mask.shape)
                contour_mask[rows, columns] = 1
                contour_mask = contour_mask == 1
                self.contours_masks.append(contour_mask)
        else:
            logger.info('There are no contours to work with.')

    def __find_contours_centers(self):
        """
        Find the centers for every mask for every contour found.
        """

        logger.info('Find the centers of the contours masks')
        logger.debug('__find_contours_centers()')

        contours_masks_len = len(self.contours_masks)
        if contours_masks_len > 1:
            logger.info(f'Finding centers for {contours_masks_len} contours masks.')
            self.contours_centers = []
            for contour_mask in self.contours_masks:
                contour_center = self.__find_mask_center(contour_mask)
                self.contours_centers.append(contour_center)
        else:
            logger.info('There are no contours masks to work with.')

    def __find_contours_bounding_boxes(self):
        """
        Find the bounding boxes for every contour mask found.
        """

        logger.info('Find the bounding boxes of the contours masks')
        logger.debug('__find_bounding_boxes()')

        contours_masks_len = len(self.contours_masks)
        if contours_masks_len > 1:
            logger.info(f'Finding bounding boxes for {contours_masks_len} contours masks.')
            self.contours_bounding_boxes = []
            for contour in self.contours:
                contour_bounding_box = self.__find_contour_bounding_box(contour)
                self.contours_bounding_boxes.append(contour_bounding_box)
        else:
            logger.info('There are no contours masks to work with.')

    def __find_mask_center(self, mask: np.array) -> np.array:
        """
        Find the center of the mask where the values are True. If the resulting
        point is outside the mask, move it inside.

        :param mask: mask which center is needed.

        :return: an array with the X and Y coordinates of the center of the
        mask.
        """

        logger.info("Find mask's center")
        logger.debug(f'__find_mask_center('
                     f'mask={mask.shape})')

        center_of_mass_x, center_of_mass_y = scipy.ndimage.center_of_mass(mask)
        center_of_mass = int(center_of_mass_x), int(center_of_mass_y)
        if self.__is_point_inside_mask(mask, center_of_mass):
            logger.info("Point is inside mask")
            center_x, center_y = self.__center_point_in_mask(mask, center_of_mass)
        else:
            logger.info("Point is outside mask")
            center_x, center_y = self.__move_point_inside_mask(mask, center_of_mass)

        return np.array([center_x, center_y])

    @staticmethod
    def __is_point_inside_mask(mask: np.array, point: Tuple) -> bool:
        """
        Checks whether a given point is inside the mask.

        :param mask: mask where the point should be inside.
        :param point: X and Y coordinates of the point.

        :return: True if the point is inside the mask, False otherwise.
        """

        logger.info("Check if point is inside mask")
        logger.debug(f'__is_point_inside_mask('
                     f'mask={mask.shape}, '
                     f'point={point})')

        x, y = point

        assert 0 <= x < mask.shape[0]
        assert 0 <= y < mask.shape[1]

        result = mask[x, y]

        return result

    @staticmethod
    def __center_point_in_mask(mask: np.array, point: Tuple) -> Tuple:
        """
        Center a given point in the mask. The point must be inside the mask.

        :param mask: mask where the point should be centered.
        :param point: X and Y coordinates of the point.

        :return: new coordinates of the point, now centered in the mask.
        """

        logger.info("Center point inside mask")
        logger.debug(f'__center_point_in_mask('
                     f'mask={mask.shape}, '
                     f'point={point})')

        x, y = point

        # Check the point's coordinates are in the boundaries of the mask's
        # points.
        assert 0 <= x < mask.shape[0]
        assert 0 <= y < mask.shape[1]

        # Check the point is inside the mask.
        assert mask[x, y]

        # Get the mask's points in the X axis of the point: the last False
        # to the left, the first to the right. This is done to avoid multiple
        # cuts to the mask, we are only interested in the region adjacent to
        # the center.
        mask_points_left = mask[:x, y]
        mask_points_left_start = np.where(mask_points_left == False)[0][-1]
        mask_points_left = mask_points_left[mask_points_left_start + 1:]

        mask_points_right = mask[x + 1:, y]
        mask_points_right_end = np.where(mask_points_right == False)[0][0]
        mask_points_right = mask_points_right[:mask_points_right_end]

        # Get the mask's points in the Y axis of the point: the last False
        # above, the first below. This is done to avoid multiple cuts to the
        # mask, we are only interested in the region adjacent to the center.
        mask_points_above = mask[x, :y]
        mask_points_above_start = np.where(mask_points_above == False)[0][-1]
        mask_points_above = mask_points_above[mask_points_above_start + 1:]

        mask_points_below = mask[x, y + 1:]
        mask_points_below_end = np.where(mask_points_below == False)[0][0]
        mask_points_below = mask_points_below[:mask_points_below_end]

        center_x = ((x - len(mask_points_left)) + (x + len(mask_points_right)) + 1) // 2
        center_y = ((y - len(mask_points_above)) + (y + len(mask_points_below)) + 1) // 2

        return center_x, center_y

    @staticmethod
    def __move_point_inside_mask(mask: np.array, point: Tuple) -> Tuple:
        """
        Move and center a given point inside the mask.

        :param mask: mask where the point should be inside.
        :param point: X and Y coordinates of the point.

        :return: new coordinates of the point, now centered inside in the mask.
        """

        logger.info("Move point inside mask")
        logger.debug(f'__move_point_inside('
                     f'mask={mask.shape}, '
                     f'point={point})')

        x, y = point

        # Check the point's coordinates are in the boundaries of the mask's
        # points.
        assert 0 <= x < mask.shape[0]
        assert 0 <= y < mask.shape[1]

        # Check the point is outside the mask.
        assert not mask[x, y]

        # Get the mask points left, right, above and below the given point.
        mask_points_left = mask[:x, y]
        mask_points_right = mask[x + 1:, y]
        mask_points_above = mask[x, :y]
        mask_points_below = mask[x, y + 1:]

        # Get the segment of the mask in each list of points.
        mask_segment_left = np.where(mask_points_left == True)[0]
        mask_segment_right = np.where(mask_points_right == True)[0] + x + 1
        mask_segment_above = np.where(mask_points_above == True)[0]
        mask_segment_below = np.where(mask_points_below == True)[0] + y + 1

        # Put them in a list to get the max.
        mask_segments = [
            mask_segment_left,
            mask_segment_right,
            mask_segment_above,
            mask_segment_below]
        mask_segments_lengths = [len(mask_segment_left),
                                 len(mask_segment_right),
                                 len(mask_segment_above),
                                 len(mask_segment_below)]
        mask_segments_lengths_arg_max = np.argmax(mask_segments_lengths)

        if sum(mask_segments_lengths) > 0:
            # Get the max segment and its center
            mask_segment_max = mask_segments[mask_segments_lengths_arg_max]
            mask_segment_max_center = int(np.mean(mask_segment_max))

            new_point_location = PointLocation(mask_segments_lengths_arg_max)
            if new_point_location == PointLocation.Left:
                logger.info('We have found an intersection with the mask on the left')
                new_point = mask_segment_max_center, y
            elif new_point_location == PointLocation.Right:
                logger.info('We have found an intersection with the mask on the right')
                new_point = mask_segment_max_center, y
            elif new_point_location == PointLocation.Above:
                logger.info('We have found an intersection with the mask above')
                new_point = x, mask_segment_max_center
            elif new_point_location == PointLocation.Below:
                logger.info('We have found an intersection with the mask below')
                new_point = x, mask_segment_max_center
            else:
                raise NotImplementedError

            # Center the new point inside the mask's region.
            new_point_centered = Masks.__center_point_in_mask(mask, new_point)
        else:
            logger.info('We have not found an intersection with the mask')
            mask_points = np.where(mask == True)
            random_mask_point = random.randrange(len(mask_points[0]))
            new_point = mask_points[0][random_mask_point], mask_points[1][random_mask_point]
            new_point_centered = Masks.__center_point_in_mask(mask, new_point)

        return new_point_centered

    @staticmethod
    def __find_contour_bounding_box(contour: np.array) -> np.array:
        """
        Given a contour, return its bounding box.

        :param contour: contour which bounding box is needed.
        :return: contour's bounding box, x coordinates first, y coordinates
        last.
        """

        logger.info("Find contour mask's bounding box")
        logger.debug(f'__find_contour_bounding_box('
                     f'contour={contour.shape})')

        x_min = np.min(contour[:, 0])
        x_max = np.max(contour[:, 0])
        y_min = np.min(contour[:, 1])
        y_max = np.max(contour[:, 1])

        return [x_min, x_max, y_min, y_max]
