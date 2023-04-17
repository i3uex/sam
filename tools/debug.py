import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class Debug:
    enabled: bool
    image_file_path: Path
    masks_file_path: Path
    slice_number: int
    base_folder_path = Path('debug')
    folder_path: Path
    base_file_path: Path

    def __init__(
            self,
            enabled: bool,
            image_file_path: Path,
            masks_file_path: Path
    ):
        """
        Init Debug class instance.

        :param enabled: True if debug information should be created, False
        otherwise.
        :param image_file_path: path to the images file.
        :param masks_file_path: path to the masks file.
        """

        logger.info('Init Debug')
        logger.debug(f'Debug.__init__('
                     f'enabled={enabled}, '
                     f'image_file_path="{image_file_path}", '
                     f'mask_file_path="{masks_file_path}")')

        self.enabled = enabled
        self.image_file_path = image_file_path
        self.masks_file_path = masks_file_path

        if self.enabled:
            self.set_folder_path()

    def set_folder_path(self):
        self.folder_path = self.base_folder_path / Path(self.image_file_path.stem)
        self.folder_path.mkdir(parents=True, exist_ok=True)

    def set_slice_number(self, slice_number: int):
        self.slice_number = slice_number
        self.set_base_file_path(self.slice_number)

    def set_base_file_path(self, slice_number: int):
        base_file_name = f'slice_{slice_number}'
        self.base_file_path = self.folder_path / Path(base_file_name)

    def get_file_path(self, name_suffix: str, extension: str) -> Path:
        output_file_stem = f'{self.base_file_path.stem}_{name_suffix}'
        output_file_path = self.base_file_path \
            .with_stem(output_file_stem) \
            .with_suffix(extension)

        return output_file_path
