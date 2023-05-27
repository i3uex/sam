import logging
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)


def get_most_recent_timestamped_file(
        folder_path: Path,
        file_pattern: str
) -> Union[Path, None]:
    """
    Looks for all the files in a given folder that matches the pattern. Sorts
    them alphabetically, as a timestamp is part of the name. Returns the last
    one, it should be the most recent. If no file matching the pattern is found
    in the folder, returns None.

    :param folder_path: path to the folder where to look for the files.
    :param file_pattern: pattern the files must follow.

    :return: the most recent file in the given folder that matches the pattern,
    None if no file does.
    """

    logger.info('Get most recent timestamped file')
    logger.debug(f'get_most_recent_timestamped_file('
                 f'folder_path="{folder_path}", '
                 f'file_pattern="{file_pattern}")')

    files_paths_generator = folder_path.glob(file_pattern)
    files_paths = []
    for file_path in files_paths_generator:
        files_paths.append(file_path)

    if len(files_paths) > 0:
        logger.info(f'There are {len(files_paths)} file(s) matching the pattern')
        files_paths.sort()
        result = files_paths[-1]
        logger.info(f'There most recent one is "{result}".')
    else:
        logger.info('There are no files matching the pattern')
        result = None

    return result
