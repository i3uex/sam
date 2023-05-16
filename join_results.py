import argparse
import logging
from pathlib import Path
from typing import Tuple, Union

import pandas as pd
from rich import print

from csv_keys import *
from tools.argparse_helper import ArgumentParserHelper
from tools.summarizer import Summarizer
from tools.timestamp import Timestamp

logger = logging.getLogger(__name__)

LoggingEnabled = True
RawDataFilePattern = 'raw_data_*.csv'
ResultsFilePattern = 'results_*.csv'


def parse_arguments() -> Tuple[Path, bool]:
    """
    Parse arguments passed via command line, returning them formatted. Adequate
    defaults are provided when possible.

    :return: path of the results folder, dry run option.
    """

    logger.info('Get script arguments')
    logger.debug('parse_arguments()')

    program_description = 'Process image'
    argument_parser = argparse.ArgumentParser(description=program_description)
    argument_parser.add_argument('-r', '--results_folder_path',
                                 required=True,
                                 help='path to results folder')
    argument_parser.add_argument('-n', '--dry_run',
                                 action='store_true',
                                 help='show what would be done, do not do it')

    arguments = argument_parser.parse_args()
    results_folder_path = ArgumentParserHelper.parse_dir_path_segment(
        arguments.results_folder_path)
    dry_run = arguments.dry_run

    return Path(results_folder_path), dry_run


def get_summary(
        results_folder_path: Path,
        dry_run: bool
) -> str:
    """
    Show a summary of the actions this script will perform.

    :param results_folder_path: path to the results' folder.
    :param dry_run: if True, the actions will not be performed.

    :return: summary of the actions this script will perform.
    :rtype: str
    """

    logger.info('Get summary')
    logger.debug(f'get_summary('
                 f'results_folder_path="{results_folder_path}", '
                 f'dry_run={dry_run})')

    summary = f'- Results folder path: "{results_folder_path}"\n' \
              f'- Dry run: {dry_run}\n'

    return summary


def get_most_recent_timestamped_file(
        folder_path: Path,
        file_pattern: str
) -> Union[Path, None]:
    """
    Looks for all the files in a given folder that matches the pattern. Sorts
    them alphabetically, as a timestamp is part of the name. Return the last
    one, it should be the most recent. If no file matching the pattern is found
    in the folder, return None.

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


def join_results(results_folder_path: Path) -> Union[Path, None]:
    """
    Given a folder, gets every folder inside it with a name that follows a
    given pattern. Inside, get a CSV file with a name that follows a given
    pattern and with the latest timestamp. Join all the CSV files obtained
    this way in a new CSV file. Save it in the results folder, adding a
    timestamp at the end.

    :param results_folder_path: folder where the results are stored.

    :return: path to the join results file created, None if there were no
    results to join.
    """

    logger.info('Join results in a new file')
    logger.debug(f'join_results('
                 f'results_folder_path={results_folder_path})')

    timestamp = Timestamp.file()

    result_folder_paths = []
    for path in Path(results_folder_path).iterdir():
        if path.is_dir():
            result_folder_paths.append(path)

    if len(result_folder_paths) > 0:
        result_folder_paths.sort()
        logger.info(f'There are {len(result_folder_paths)} result folders:')
        for result_folder_path in result_folder_paths:
            logger.info(f'- {result_folder_path}')
    else:
        logger.info('There are no result folders to process')
        return None

    joint_results = []
    for result_folder_path in result_folder_paths:
        raw_data_file_path = get_most_recent_timestamped_file(
            result_folder_path, RawDataFilePattern)
        results_file_path = get_most_recent_timestamped_file(
            result_folder_path, ResultsFilePattern)

        df_raw_data = pd.read_csv(raw_data_file_path)
        df_results = pd.read_csv(results_file_path)
        result = {
            ImageKey: result_folder_path.name,
            SlicesKey: len(df_raw_data)
        }
        for index, row in df_results.iterrows():
            metric = row[MetricKey]
            result[f'{metric}_{MinKey}'] = row[MinKey]
            result[f'{metric}_{MaxKey}'] = row[MaxKey]
            result[f'{metric}_{AverageKey}'] = row[AverageKey]
            result[f'{metric}_{StandardDeviationKey}'] = row[StandardDeviationKey]

        joint_results.append(result)

    df_joint_results = pd.DataFrame(joint_results)
    if len(joint_results) > 0:
        # Include aggregate values in the last row
        averages = {
            ImageKey: 'average',
            SlicesKey: df_joint_results[SlicesKey].mean()
        }
        for metric_key in MetricKeys:
            key = f'{metric_key}_{MinKey}'
            averages[key] = df_joint_results[key].mean()
            key = f'{metric_key}_{MaxKey}'
            averages[key] = df_joint_results[key].mean()
            key = f'{metric_key}_{AverageKey}'
            averages[key] = df_joint_results[key].mean()
            key = f'{metric_key}_{StandardDeviationKey}'
            averages[key] = df_joint_results[key].mean()
        df_averages = pd.DataFrame([averages])
        df_joint_results = pd.concat([df_joint_results, df_averages])
    joint_results_path = results_folder_path / Path(f'results_{timestamp}.csv')
    df_joint_results.to_csv(joint_results_path, index=False)

    return joint_results_path


def main():
    """
    Set logging up, parse arguments, and process data.
    """

    if LoggingEnabled:
        logging.basicConfig(
            filename="scripts/debug.log",
            level=logging.DEBUG,
            format="%(asctime)-15s %(levelname)8s %(name)s %(message)s")
        logging.getLogger("matplotlib.font_manager").disabled = True
        logging.getLogger("matplotlib.colorbar").disabled = True
        logging.getLogger("matplotlib.pyplot").disabled = True

    logger.info("Start joining results")
    logger.debug("main()")

    summarizer = Summarizer()

    results_folder_path, dry_run = parse_arguments()

    summarizer.summary = get_summary(
        results_folder_path=results_folder_path,
        dry_run=dry_run)

    if dry_run:
        print()
        print('[bold]Summary:[/bold]')
        print(summarizer.summary)
        return

    result = join_results(results_folder_path=results_folder_path)
    print(f'Results joint in file: "{result}"')

    print(summarizer.notification_message)


if __name__ == '__main__':
    main()
