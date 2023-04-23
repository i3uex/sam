import logging
from datetime import datetime, timedelta

import humanize

logger = logging.getLogger(__name__)


class Summarizer:
    """
    Show summary of script execution.

    Usage:

        1. Create class instance. That saves the start timer.
        2. Optionally, assign a summary.
        3. Get the notification message. That saves the stop timestamp.

    Example:

        summarizer = Summarizer()
        summarizer.summary = get_summary()
        print(summarizer.notification_message)
    """

    summary: str
    __notification_message: str
    start_timestamp: datetime
    end_timestamp: datetime

    def __init__(self):
        """
        Init Summarizer class instance. It also saves a start timestamp.
        """

        logger.info('Init Summarizer')
        logger.debug('Summarizer.__init__()')

        self.__save_start_timestamp()

    def __save_start_timestamp(self):
        """
        Save start timestamp.
        """

        logger.info('Save start timestamp')
        logger.debug('__save_start_timestamp()')

        self.start_timestamp = datetime.now()

    def __save_end_timestamp(self):
        """
        Save end timestamp.
        """

        logger.info('Save end timestamp')
        logger.debug('__save_end_timestamp()')

        self.end_timestamp = datetime.now()

    def get_notification_message(self):
        """
        Get notification message with time details and the summary provided.

        :return: notification message with time details and the summary
        provided.
        """

        logger.info('Get notification message')
        logger.debug('get_notification_message()')

        self.__save_end_timestamp()

        elapsed_seconds = (self.end_timestamp - self.start_timestamp).seconds
        start_time = self.start_timestamp.strftime("%H:%M:%S")
        start_date = self.start_timestamp.strftime("%Y-%m-%d")
        elapsed_time = humanize.naturaldelta(timedelta(seconds=elapsed_seconds))

        self.__notification_message = \
            f'The task started at {start_time} on {start_date} has just finished.\n' \
            f'It took {elapsed_time} to complete.'

        if self.summary != '':
            self.__notification_message += \
                '\n' \
                f'Summary of operations performed:\n' \
                f'{self.summary}'

        return self.__notification_message

    # Getter method for self.__notification_message
    notification_message = property(get_notification_message)
