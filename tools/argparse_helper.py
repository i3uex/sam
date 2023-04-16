import logging
import os

log = logging.getLogger(__name__)


class ArgumentParserHelper:

    @staticmethod
    def parse_file_path_segment(
            file_path_segment: str,
            check_is_file: bool = True
    ):
        """
        Parse a file path segment and check the corresponding file exists if
        needed.

        :param file_path_segment: path segment for the file to check.
        :param check_is_file: if True, check if the file exists.
        """

        log.info('Parse file path segment')
        log.debug(f'ArgumentParserHelper.parse_file_path_segment('
                  f'file_path_segment={file_path_segment}, '
                  f'check_is_file={check_is_file})')

        if file_path_segment == "":
            print('file path segment not provided')
            exit(1)

        if check_is_file:
            if not os.path.isfile(file_path_segment):
                print(f'cannot open file {file_path_segment}')
                exit(1)

        return file_path_segment

    @staticmethod
    def parse_integer(integer_as_string: str) -> int:
        """
        Parse an aspiring integer as a string and check if it is right.

        :param integer_as_string: what should be an integer, as a string.
        """

        log.info('Parse integer')
        log.debug(f'ArgumentParserHelper.parse_integer('
                  f'integer_as_string={integer_as_string})')

        if integer_as_string == "":
            print('integer not provided')
            exit(1)

        try:
            integer_value = int(integer_as_string)
        except ValueError:
            print('integer provided is not a number')
            exit(1)

        return integer_value
