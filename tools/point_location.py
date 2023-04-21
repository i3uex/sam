import logging
from enum import Enum

logger = logging.getLogger(__name__)


class PointLocation(Enum):
    """
    Locations around a point.
    """

    Left = 0
    Right = 1
    Above = 2
    Below = 3
