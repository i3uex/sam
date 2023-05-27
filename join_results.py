"""
Given a collection of results in a series of folders, this scripts collects
all, joins them, and compiles a series of global results.
"""

import argparse
import logging
from pathlib import Path
from typing import Tuple, Union

import pandas as pd
from rich import print

from csv_keys import *
from tools.argparse_helper import ArgumentParserHelper
from tools.files import get_most_recent_timestamped_file
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
    """

    logger.info('Get summary')
    logger.debug(f'get_summary('
                 f'results_folder_path="{results_folder_path}", '
                 f'dry_run={dry_run})')

    summary = f'- Results folder path: "{results_folder_path}"\n' \
              f'- Dry run: {dry_run}'

    return summary


def join_results(results_folder_path: Path) -> Union[Tuple, None]:
    """
    Given a folder, gets every folder inside it with a name that follows a
    given pattern. Inside, gets a CSV file with a name that follows a given
    pattern and with the latest timestamp. Joins all the CSV files obtained
    this way in a new CSV file. Includes some new data (i.e., aggregate
    statistical values). Saves it in the results folder, adding a timestamp at
    the end.

    :param results_folder_path: folder where the results are stored.

    :return: paths to the join results files created, None if there were no
    results to join.
    """

    logger.info('Join results in a new file')
    logger.debug(f'join_results('
                 f'results_folder_path="{results_folder_path}")')

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
    df_joint_raw_data = None
    for result_folder_path in result_folder_paths:
        raw_data_file_path = get_most_recent_timestamped_file(
            result_folder_path, RawDataFilePattern)
        results_file_path = get_most_recent_timestamped_file(
            result_folder_path, ResultsFilePattern)

        df_raw_data = pd.read_csv(raw_data_file_path)
        df_results = pd.read_csv(results_file_path)

        # Prepare joint raw data
        df_raw_data.insert(0, ImageKey, result_folder_path.stem)
        if df_joint_raw_data is None:
            df_joint_raw_data = df_raw_data
        else:
            df_joint_raw_data = pd.concat([df_joint_raw_data, df_raw_data])

        # Prepare joint results
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

    # Create new file with raw data's statistical values
    results = [
        {
            MetricKey: JaccardKey,
            MinKey: df_joint_raw_data[JaccardKey].min(),
            MaxKey: df_joint_raw_data[JaccardKey].max(),
            AverageKey: df_joint_raw_data[JaccardKey].mean(),
            StandardDeviationKey: df_joint_raw_data[JaccardKey].std()
        },
        {
            MetricKey: DiceKey,
            MinKey: df_joint_raw_data[DiceKey].min(),
            MaxKey: df_joint_raw_data[DiceKey].max(),
            AverageKey: df_joint_raw_data[DiceKey].mean(),
            StandardDeviationKey: df_joint_raw_data[DiceKey].std()
        }
    ]
    df_results = pd.DataFrame(results)

    joint_raw_data_path = results_folder_path / Path(f'joint_raw_data_{timestamp}.csv')
    joint_results_path = results_folder_path / Path(f'joint_results_{timestamp}.csv')
    results_path = results_folder_path / Path(f'results_{timestamp}.csv')

    df_joint_raw_data.to_csv(joint_raw_data_path, index=False)
    df_joint_results.to_csv(joint_results_path, index=False)
    df_results.to_csv(results_path, index=False)

    return joint_results_path, joint_results_path, results_path


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

    results = join_results(results_folder_path=results_folder_path)

    print(f'Raw data joint in file: "{results[0]}"')
    print(f'Results joint in file: "{results[1]}"')
    print(f'Results in file: "{results[2]}"')

    print(summarizer.notification_message)


if __name__ == '__main__':
    main()
