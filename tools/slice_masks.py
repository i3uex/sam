import logging
import random
from typing import Tuple

import numpy as np
import scipy.ndimage
from skimage import measure
from skimage.draw import polygon

from tools.point_location import PointLocation

logger = logging.getLogger(__name__)


class SliceMasks:
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

    labeled_image: np.array = None
    labels: np.array = None

    # Coordinates of the polygon surrounding each object in the masks
    contours: list = None
    # Original labels for the contours
    contours_labels: list = None
    # Masks for the contours
    contours_masks: list = None
    # Centered points inside each contour, and a last one marking the center of
    # the image
    contours_centers: np.array = None
    # SAM's labels of the points inside each contour
    contours_centers_labels: np.array = None
    # Top left, bottom right coordinates
    contours_bounding_boxes: np.array = None

    def __init__(self, labeled_image: np.array):
        """
        Init Slice Masks class instance with an array of values. The value 0
        means no object was present in that location. Every value greater than
        0 marks a different object.

        :param labeled_image: array with the values of the masks.
        """

        logger.info('Init Masks')
        logger.debug(f'Masks.__init__('
                     f'labeled_image={labeled_image.shape})')

        self.labeled_image = labeled_image
        self.process_labeled_image()

    def process_labeled_image(self):
        """
        If the labeled image provided to the class contains other values than
        0, find the contours. For each contour found, find their masks, their
        centers and their bounding boxes.
        """

        logger.info('Process labeled image')
        logger.debug('process_labeled_image()')

        self.labels = np.unique(self.labeled_image)
        if self.labels.size > 1:
            self.find_contours()
            self.find_contours_masks()
            self.find_contours_centers()
            self.find_contours_bounding_boxes()

    def find_contours(self):
        """
        Find contours for every point value other than 0.
        """

        logger.info('Find contours')
        logger.debug('find_contours()')

        if self.labels.size > 1:
            logger.info(f'Finding contours for {self.labels.size} labels.')
            self.contours = []
            self.contours_labels = []
            for label in range(1, self.labels.size):
                mask = self.labeled_image == label
                # Avoid one contour inside another
                mask_filled = scipy.ndimage.binary_fill_holes(mask)
                mask_contours = measure.find_contours(mask_filled)
                for mask_contour in mask_contours:
                    self.contours.append(mask_contour)
                    self.contours_labels.append(label)
        else:
            logger.info('There are no contours to find.')

    def find_contours_masks(self):
        """
        Find masks for every contour. A mask is an array with the same shape
        that the points provided, filled with False. The True values are inside
        the polygon defined by the contour.
        """

        logger.info('Find contours masks')
        logger.debug('find_contours_masks()')

        if self.contours is not None:
            logger.info(f'Finding masks for {len(self.contours)} contours.')
            self.contours_masks = []
            for contour in self.contours:
                contour_mask = np.zeros(
                    (self.labeled_image.shape[0], self.labeled_image.shape[1]), 'uint8')
                rows, columns = polygon(
                    contour[:, 0], contour[:, 1], contour_mask.shape)
                contour_mask[rows, columns] = 1
                contour_mask = contour_mask == 1
                self.contours_masks.append(contour_mask)
        else:
            logger.info('There are no contours to work with.')

    def find_contours_centers(self):
        """
        Find the centers for every contour found.
        """

        logger.info('Find the centers of the contours')
        logger.debug('find_contours_centers()')

        mask_indexes_len = len(self.labels)
        if mask_indexes_len > 1:
            logger.info(f'Finding centers for {mask_indexes_len - 1} masks.')
            contours_centers = []
            for index, contour_mask in enumerate(self.contours_masks):
                contour_label = self.contours_labels[index]
                contour_center = self.find_mask_center(contour_mask, contour_label)
                contours_centers.append(contour_center)
            # Add an extra center point, just in the center of the points
            contours_centers.append([
                self.labeled_image.shape[0] // 2,
                self.labeled_image.shape[1] // 2
            ])
            self.contours_centers = np.array(contours_centers).astype(np.int16)
            # Mark every center with 1 but the last one
            self.contours_centers_labels = np.ones(len(self.contours_centers))
            self.contours_centers_labels[-1] = 0
        else:
            logger.info('There are no contours masks to work with.')

    def find_contours_bounding_boxes(self):
        """
        Find the bounding boxes for every contour mask found.
        """

        logger.info('Find the bounding boxes of the contours masks')
        logger.debug('find_bounding_boxes()')

        if self.contours_masks is not None:
            logger.info(f'Finding bounding boxes for {len(self.contours_masks)} contours masks.')
            contours_bounding_boxes = []
            for contour in self.contours:
                contour_bounding_box = self.find_contour_bounding_box(contour)
                contours_bounding_boxes.append(contour_bounding_box)
            self.contours_bounding_boxes = np.array(contours_bounding_boxes).astype(np.int16)
        else:
            logger.info('There are no contours masks to work with.')

    def find_mask_center(self, mask: np.array, contour_label: int) -> np.array:
        """
        Find the center of the mask where the values are True. If the resulting
        point is outside the mask, move it inside.

        :param mask: mask which center is needed.
        :param contour_label: original label for the contour. Use it to check
        if the center is inside the original region.

        :return: an array with the row and column coordinates of the center of
        the mask.
        """

        logger.info("Find mask's center")
        logger.debug(f'find_mask_center('
                     f'mask={mask.shape}, '
                     f'contour_label={contour_label})')

        original_mask = self.labeled_image == contour_label
        center_of_mass_row, center_of_mass_column = scipy.ndimage.center_of_mass(mask)
        center_of_mass = int(center_of_mass_row), int(center_of_mass_column)
        if self.is_point_inside_mask(original_mask, center_of_mass):
            logger.info("Point is inside mask")
            center_row, center_column = self.center_point_in_mask(original_mask, center_of_mass)
        else:
            logger.info("Point is outside mask")
            center_row, center_column = self.move_point_inside_mask(original_mask, center_of_mass)

        return np.array([center_row, center_column])

    @staticmethod
    def is_point_inside_mask(mask: np.array, point: Tuple) -> bool:
        """
        Checks whether a given point is inside the mask.

        :param mask: mask where the point should be inside.
        :param point: row and column coordinates of the point.

        :return: True if the point is inside the mask, False otherwise.
        """

        logger.info("Check if point is inside mask")
        logger.debug(f'is_point_inside_mask('
                     f'mask={mask.shape}, '
                     f'point={point})')

        row, column = point

        assert 0 <= row < mask.shape[0]
        assert 0 <= column < mask.shape[1]

        result = mask[row, column]

        return result

    @staticmethod
    def center_point_in_mask(mask: np.array, point: Tuple) -> Tuple:
        """
        Center a given point in the mask. The point must be inside the mask.

        :param mask: mask where the point should be centered.
        :param point: row and column coordinates of the point.

        :return: new coordinates of the point, now centered in the mask.
        """

        logger.info("Center point inside mask")
        logger.debug(f'center_point_in_mask('
                     f'mask={mask.shape}, '
                     f'point={point})')

        row, column = point

        # Check the point's coordinates are in the boundaries of the mask's
        # points.
        assert 0 <= row < mask.shape[0]
        assert 0 <= column < mask.shape[1]

        # Check the point is inside the mask.
        assert mask[row, column]

        # Get the mask's points in the row of the point: the last False to the
        # left, the first to the right. This is done to avoid multiple cuts to
        # the mask, we are only interested in the region adjacent to the
        # center.
        mask_points_left = mask[row, :column]
        mask_points_left_start = np.where(mask_points_left == False)[0][-1]
        mask_points_left = mask_points_left[mask_points_left_start + 1:]

        mask_points_right = mask[row, column + 1:]
        mask_points_right_end = np.where(mask_points_right == False)[0][0]
        mask_points_right = mask_points_right[:mask_points_right_end]

        # Get the mask's points in the column axis of the point: the last False
        # above, the first below. This is done to avoid multiple cuts to the
        # mask, we are only interested in the region adjacent to the center.
        mask_points_above = mask[:row, column]
        mask_points_above_start = np.where(mask_points_above == False)[0][-1]
        mask_points_above = mask_points_above[mask_points_above_start + 1:]

        mask_points_below = mask[row + 1:, column]
        mask_points_below_end = np.where(mask_points_below == False)[0][0]
        mask_points_below = mask_points_below[:mask_points_below_end]

        center_row = ((row - len(mask_points_left)) + (row + len(mask_points_right)) + 1) // 2
        center_column = ((column - len(mask_points_above)) + (column + len(mask_points_below)) + 1) // 2

        return center_row, center_column

    @staticmethod
    def move_point_inside_mask(mask: np.array, point: Tuple) -> Tuple:
        """
        Move and center a given point inside the mask.

        :param mask: mask where the point should be inside.
        :param point: row and column coordinates of the point.

        :return: new coordinates of the point, now centered inside in the mask.
        """

        logger.info("Move point inside mask")
        logger.debug(f'move_point_inside_mask('
                     f'mask={mask.shape}, '
                     f'point={point})')

        row, column = point

        # Check the point's coordinates are in the boundaries of the mask's
        # points.
        assert 0 <= row < mask.shape[0]
        assert 0 <= column < mask.shape[1]

        # Check the point is outside the mask.
        assert not mask[row, column]

        # Get the mask points left, right, above and below the given point.
        mask_points_left = mask[row, :column]
        mask_points_right = mask[row, column + 1:]
        mask_points_above = mask[:row, column]
        mask_points_below = mask[row + 1:, column]

        # Get the segment of the mask in each list of points.
        mask_segment_left = np.where(mask_points_left == True)[0]
        mask_segment_right = np.where(mask_points_right == True)[0] + column + 1
        mask_segment_above = np.where(mask_points_above == True)[0]
        mask_segment_below = np.where(mask_points_below == True)[0] + row + 1

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
                new_point = row, mask_segment_max_center
            elif new_point_location == PointLocation.Right:
                logger.info('We have found an intersection with the mask on the right')
                new_point = row, mask_segment_max_center
            elif new_point_location == PointLocation.Above:
                logger.info('We have found an intersection with the mask above')
                new_point = mask_segment_max_center, column
            elif new_point_location == PointLocation.Below:
                logger.info('We have found an intersection with the mask below')
                new_point = mask_segment_max_center, column
            else:
                raise NotImplementedError

            # Center the new point inside the mask's region.
            new_point_centered = SliceMasks.center_point_in_mask(mask, new_point)
        else:
            logger.info('We have not found an intersection with the mask')
            mask_points = np.where(mask == True)
            random_mask_point = random.randrange(len(mask_points[0]))
            new_point = mask_points[0][random_mask_point], mask_points[1][random_mask_point]
            new_point_centered = SliceMasks.center_point_in_mask(mask, new_point)

        return new_point_centered

    @staticmethod
    def find_contour_bounding_box(contour: np.array) -> np.array:
        """
        Given a contour, return its bounding box.

        :param contour: contour which bounding box is needed.
        :return: contour's bounding box, row coordinates first, row coordinates
        last.
        """

        logger.info("Find contour mask's bounding box")
        logger.debug(f'find_contour_bounding_box('
                     f'contour={contour.shape})')

        row_min = np.min(contour[:, 0])
        column_min = np.min(contour[:, 1])
        row_max = np.max(contour[:, 0])
        column_max = np.max(contour[:, 1])

        return [row_min, column_min, row_max, column_max]
