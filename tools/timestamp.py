import datetime
import logging

logger = logging.getLogger(__name__)


class Timestamp:
    @staticmethod
    def get(format_string: str) -> str:
        """
        Get a timestamp with the format specified.

        :param format_string: format the timestamp must have.
        :return: timestamp with the format specified.
        :rtype: str
        """

        logger.info('Get timestamp with format')
        logger.debug(f'Timestamp.get('
                     f'format_string={format_string}')

        timestamp = datetime.datetime.now().strftime(format_string)

        return timestamp

    @staticmethod
    def file() -> str:
        """
        Get a file timestamp.

        :return: timestamp for a file
        :rtype: str
        """

        logger.info('Get timestamp for a file')
        logger.debug('Timestamp.file()')

        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        return timestamp
