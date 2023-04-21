import logging
from typing import Tuple

import numpy as np
from scipy.ndimage import center_of_mass

from tools.point_location import PointLocation

logger = logging.getLogger(__name__)


# TODO: use a point class.
class Mask:
    """
    A mask is an array with False and True values. The True values represent a
    region of the array where the mask is activated, while the False values
    represent a region where the mask is deactivated.

    Usage:

    Example:
    """

    points: np.array

    def __init__(self, points: np.array):
        """
        Init Mask class instance with an array of True/False values.
        """

        logger.info('Init Mask')
        logger.debug(f'Mask.__init__('
                     f'points={points.shape})')

        self.points = points

    def get_center(self) -> np.array:
        """
        Get the center of the region of the mask where the values are True.

        :return: An array with the X and Y coordinates of the center of the
        mask.
        """

        logger.info("Get mask's center")
        logger.debug('get_center()')

        center_of_mass_x, center_of_mass_y = center_of_mass(self.points)
        mask_center_of_mass = int(center_of_mass_x), int(center_of_mass_y)
        if self.__is_point_inside(mask_center_of_mass):
            x, y = self.__center_point(mask_center_of_mass)
        else:
            x, y = self.__move_point_inside(mask_center_of_mass)

        return np.array([x, y])

    def __is_point_inside(self, point: Tuple) -> bool:
        """
        Checks whether a given point is inside the mask.

        :param point: X and Y coordinates of the point.

        :return: True if the point is inside the mask, False otherwise.
        """

        logger.info("Check if point is inside mask")
        logger.debug(f'__is_point_inside_mask('
                     f'point={point})')

        x, y = point

        assert 0 <= x < self.points.shape[0]
        assert 0 <= y < self.points.shape[1]

        result = self.points[x - 1, y] and \
                 self.points[x + 1, y] and \
                 self.points[x, y - 1] and \
                 self.points[x, y + 1]

        return result

    def __center_point(self, point: Tuple) -> Tuple:
        """
        Center a given point inside the mask. The point must be inside the
        mask.

        :param point: X and Y coordinates of the point.

        :return: new coordinates of the point, now centered in the mask.
        """

        logger.info("Center point inside mask")
        logger.debug(f'__center_point('
                     f'point={point})')

        x, y = point

        # Check the point's coordinates are in the boundaries of the mask's
        # points.
        assert 0 <= x < self.points.shape[0]
        assert 0 <= y < self.points.shape[1]

        # Check the point is inside the mask.
        assert \
            self.points[x - 1, y] and \
            self.points[x + 1, y] and \
            self.points[x, y - 1] and \
            self.points[x, y + 1]

        # Get the mask's points in the X axis of the point.
        mask_points_x = self.points[:, y]
        # Get the mask's region points in X axis.
        mask_segment_x = np.where(mask_points_x == True)[0]
        # Get the center of the segment.
        center_x = int(np.mean(mask_segment_x))

        # Get the mask's points in the Y axis of the point.
        mask_points_y = self.points[x, :]
        # Get the mask's region points in the Y axis.
        mask_segment_y = np.where(mask_points_y == True)[0]
        # Get the center of the segment.
        center_y = int(np.mean(mask_segment_y))

        return center_x, center_y

    def __move_point_inside(self, point: Tuple) -> Tuple:
        """
        Move a given point inside the mask.

        :param point: X and Y coordinates of the point.

        :return: new coordinates of the point, now centered inside in the mask.
        """

        logger.info("Move point inside mask")
        logger.debug(f'__move_point_inside('
                     f'point={point})')

        x, y = point

        # Check the point's coordinates are in the boundaries of the mask's
        # points.
        assert 0 <= x < self.points.shape[0]
        assert 0 <= y < self.points.shape[1]

        # Check the point is outside the mask.
        assert \
            not self.points[x - 1, y] or \
            not self.points[x + 1, y] or \
            not self.points[x, y - 1] or \
            not self.points[x, y + 1]

        # Get the mask points left, right, above and below the given point.
        mask_points_left = self.points[:x, y]
        mask_points_right = self.points[x + 1:, y]
        mask_points_above = self.points[x, :y]
        mask_points_below = self.points[x, y + 1:]

        # Get the segment of the mask in each list of points.
        mask_segment_left = np.where(mask_points_left == True)[0]
        mask_segment_right = np.where(mask_points_right == True)[0]
        mask_segment_above = np.where(mask_points_above == True)[0]
        mask_segment_below = np.where(mask_points_below == True)[0]

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

        # Get the max segment and its center
        mask_segment_max = mask_segments[mask_segments_lengths_arg_max]
        mask_segment_max_center = int(np.mean(mask_segment_max))

        new_point_location = PointLocation(mask_segments_lengths_arg_max)
        if new_point_location == PointLocation.Left:
            new_point = mask_segment_max_center, y
        elif new_point_location == PointLocation.Right:
            new_point = mask_segment_max_center, y
        elif new_point_location == PointLocation.Above:
            new_point = x, mask_segment_max_center
        elif new_point_location == PointLocation.Below:
            new_point = x, mask_segment_max_center
        else:
            raise NotImplementedError

        # Center the new point inside the mask's region.
        # new_point_centered = self.__center_point(new_point)
        new_point_centered = new_point

        return new_point_centered
