import logging
import random
from typing import Tuple

import numpy as np
import scipy.ndimage
from skimage import measure
from skimage.draw import polygon
from skimage.measure import regionprops

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
    use_masks_contours: bool = False

    labels: np.array = None

    # Centered points inside each contour, and a last one marking the center of
    # the image
    centers: np.array = None
    # SAM's labels of the points inside each contour
    centers_labels: np.array = None
    # Top left, bottom right coordinates
    bounding_box: Tuple[int, int, int, int] = None
    # Coordinates of the polygon surrounding each object in the masks
    contours: list = None

    # Original labels for the contours
    __contours_labels: list = None
    # Masks for the contours
    __contours_masks: list = None

    def __init__(self, labeled_image: np.array, use_masks_contours: bool):
        """
        Init Slice Masks class instance with an array of values. The value 0
        means no object was present in that location. Every value greater than
        0 marks a different object.

        :param labeled_image: array with the values of the masks.
        :param use_masks_contours: if True, get positive prompts from contours.
        Else, get them from the mask. This means that if a given mask have more
        than one contour, the number of positive prompts will be greater than
        one.
        """

        logger.info('Init Masks')
        logger.debug(f'Masks.__init__('
                     f'labeled_image={labeled_image.shape}, '
                     f'use_masks_contours={use_masks_contours})')

        self.labeled_image = labeled_image
        self.use_masks_contours = use_masks_contours

        self.__process_labeled_image()

    def get_point_coordinates(self) -> np.array:
        """
        Change the coordinates of each center found to match what SAM expects.

        This project uses rows, columns (y, x) as coordinate convention, SAM
        expects point prompts to be x, y.

        :return: array of x, y point coordinates.
        """

        logger.info('Get point coordinates')
        logger.debug('get_point_coordinates()')

        return np.flip(self.centers, axis=1)

    def get_box(self) -> np.array:
        """
        Change the coordinates of the bounding box surrounding all the masks to
        match what SAM expects.

        This project uses rows, columns (y, x) as coordinate convention, SAM
        expects the bounding box to be (x, y, x, y).

        :return: array with the bounding box surrounding all the masks.
        """

        logger.info('Get box')
        logger.debug('get_box()')

        row_min = np.min(self.bounding_box[:, 0])
        column_min = np.min(self.bounding_box[:, 1])
        row_max = np.max(self.bounding_box[:, 2])
        column_max = np.max(self.bounding_box[:, 3])

        return np.array([column_min, row_min, column_max, row_max])

    def __process_labeled_image(self):
        """
        If the labeled image provided to the class contains other values than
        0, find the centers of the objects of interest.

        If the masks contours are not going to be used, find the center of each
        mask, and the corresponding bounding box.

        If the mask contours are going to be used, find the contours of each
        mask, their centers, and the corresponding bounding box.
        """

        logger.info('Process labeled image')
        logger.debug('process_labeled_image()')

        self.labels = np.unique(self.labeled_image)
        if self.labels.size > 1:
            self.__find_contours()
            if not self.use_masks_contours:
                self.__find_masks_centers()
            else:
                self.__find_contours_masks()
                self.__find_contours_centers()
            self.__add_centers_labels()
            self.__find_bounding_box()

    def __find_masks_centers(self):
        """
        Find the centers for every mask.
        """

        logger.info('Find the centers of the masks')
        logger.debug('__find_masks_centers()')

        if self.labels.size > 1:
            logger.info(f'Finding centers for {self.labels.size - 1} masks.')
            masks_centers = []
            for label in self.labels:
                if label != 0:
                    mask = self.labeled_image == label
                    mask_center = self.__find_mask_center(mask, label)
                    masks_centers.append(mask_center)
            self.__add_image_center(masks_centers)
            self.centers = np.array(masks_centers).astype(np.int16)
        else:
            logger.info('There are no masks to work with.')

    def __find_contours(self):
        """
        Find contours for every point value other than 0.
        """

        logger.info('Find contours')
        logger.debug('find_contours()')

        if self.labels.size > 1:
            logger.info(f'Finding contours for {self.labels.size - 1} labels.')
            self.contours = []
            self.__contours_labels = []
            for label in self.labels:
                if label != 0:
                    mask = self.labeled_image == label
                    # Avoid one contour inside another
                    mask_filled = scipy.ndimage.binary_fill_holes(mask)
                    mask_contours = measure.find_contours(mask_filled)
                    for mask_contour in mask_contours:
                        self.contours.append(mask_contour)
                        self.__contours_labels.append(label)
        else:
            logger.info('There are no contours to find.')

    def __find_contours_masks(self):
        """
        Find masks for every contour. A mask is an array with the same shape
        that the points provided, filled with False. The True values are inside
        the polygon defined by the contour.
        """

        logger.info('Find contours masks')
        logger.debug('find_contours_masks()')

        if self.contours is not None:
            logger.info(f'Finding masks for {len(self.contours)} contours.')
            self.__contours_masks = []
            for contour in self.contours:
                contour_mask = np.zeros(
                    (self.labeled_image.shape[0], self.labeled_image.shape[1]), 'uint8')
                rows, columns = polygon(
                    contour[:, 0], contour[:, 1], contour_mask.shape)
                contour_mask[rows, columns] = 1
                contour_mask = contour_mask == 1
                self.__contours_masks.append(contour_mask)
        else:
            logger.info('There are no contours to work with.')

    def __find_contours_centers(self):
        """
        Find the centers for every contour found.
        """

        logger.info('Find the centers of the contours')
        logger.debug('find_contours_centers()')

        mask_indexes_len = len(self.labels)
        if mask_indexes_len > 1:
            logger.info(f'Finding centers for {mask_indexes_len - 1} masks.')
            contours_centers = []
            for index, contour_mask in enumerate(self.__contours_masks):
                contour_label = self.__contours_labels[index]
                contour_center = self.__find_mask_center(contour_mask, contour_label)
                contours_centers.append(contour_center)
            self.__add_image_center(contours_centers)
            self.centers = np.array(contours_centers).astype(np.int16)
        else:
            logger.info('There are no contours masks to work with.')

    def __add_image_center(self, centers: list):
        """
        Add an extra point to a list of centers, with the coordinates of the
        center of the image.

        :param centers: list of centers to modify.
        """

        logger.info('Add image center to list')
        logger.debug(f'__add_image_center('
                     f'centers={len(centers)})')

        centers.append([
            self.labeled_image.shape[0] // 2,
            self.labeled_image.shape[1] // 2
        ])

    def __add_centers_labels(self):
        """
        Mark every center as a positive prompt (with 1) but the last
        one (with 0).
        """

        logger.info('Add labels to the centers')
        logger.debug('__add_centers_labels()')

        self.centers_labels = np.ones(len(self.centers))
        self.centers_labels[-1] = 0

    def __find_bounding_box(self):
        """
        Find the bounding box around every mask.
        """

        logger.info('Find the bounding boxes around every mask')
        logger.debug('find_bounding_box()')

        if self.labels.size > 1:
            logger.info(f'Finding the bounding box around every mask.')
            mask = np.where(self.labeled_image > 0, 1, self.labeled_image)
            mask = mask.astype(np.int16)
            regions_properties = regionprops(mask)
            region_properties = regions_properties[0]
            self.bounding_box = region_properties.bbox
        else:
            logger.info('There are no contours masks to work with.')

    def __find_mask_center(self, mask: np.array, contour_label: int) -> np.array:
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
        if self.__is_point_inside_mask(original_mask, center_of_mass):
            logger.info("Point is inside mask")
            center_row, center_column = self.__center_point_in_mask(original_mask, center_of_mass)
        else:
            logger.info("Point is outside mask")
            center_row, center_column = self.__move_point_inside_mask(original_mask, center_of_mass)

        return np.array([center_row, center_column])

    @staticmethod
    def __is_point_inside_mask(mask: np.array, point: Tuple) -> bool:
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
    def __center_point_in_mask(mask: np.array, point: Tuple) -> Tuple:
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

        mask_points_left_len = len(mask_points_left)
        mask_points_right_len = len(mask_points_right)
        mask_points_above_len = len(mask_points_above)
        mask_points_below_len = len(mask_points_below)
        mask_points_in_row = mask_points_left_len + mask_points_right_len
        mask_points_in_column = mask_points_above_len + mask_points_below_len

        if mask_points_in_row > mask_points_in_column:
            center_row = row
            center_column = column + (mask_points_right_len - mask_points_left_len + 1) // 2
        else:
            center_row = row + (mask_points_below_len - mask_points_above_len + 1) // 2
            center_column = column

        return center_row, center_column

    @staticmethod
    def __move_point_inside_mask(mask: np.array, point: Tuple) -> Tuple:
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
            new_point_centered = SliceMasks.__center_point_in_mask(mask, new_point)
        else:
            logger.info('We have not found an intersection with the mask')
            mask_points = np.where(mask == True)
            random_mask_point = random.randrange(len(mask_points[0]))
            new_point = mask_points[0][random_mask_point], mask_points[1][random_mask_point]
            new_point_centered = SliceMasks.__center_point_in_mask(mask, new_point)

        return new_point_centered
