# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.4] - 2024-01-15

### Added

- Experiments folder, with not-so-refined scripts.
- Montgomery dataset boxplot creation script.
- Script to process all Montgomery dataset images.
- Notebook to test Montgomery X-rays.
- Python script to transform Montgomery dataset to a common format (NPZ).
- Shell script to download Montgomery dataset.

### Changed

- Update project URL.
- Replace **letter** with **paper**.
- README now reflects IEEE Access published the paper.
- COVID dataset boxplot creation now reflects it in its naming.
- Montgomery's transformation script now adds an extra dimension at the end of each image, so the processing script can work with them.
- COVID dataset is now compressed in NPZ format.
- COVID dataset is now processed inside a folder.
- Boxplot for IJIMAI.

### Fixed

- Montgomery's data was treated as boolean, not integer.
- When saving Montgomery images as compressed NumPy arrays, dimensions were lost.

## [1.1.3] - 2023-05-27

### Added

- Instructions to recreate the evidences included in the letter.

### Changed

- Zotero dataset is now COVID-19 dataset.
- Review **mask_center.ipynb** notebook.
- Review **coordinates_conventions.ipynb** notebook.
- Review **windowing.ipynb** notebook.
- Review **get_slice.ipynb** notebook.
- Improve summary.
- GIF creation improvements.
- **README.md** improvements.

### Removed

- Obsolete notebooks.
- Pending tasks already completed, or no longer needed.

### Fixed

- Statistical values calculation for joint results.

## [1.1.2] - 2023-05-23

### Added

- Notebook to show box plots.
- Notebook to show algorithm's evolution.

## [1.1.1] - 2023-05-17

### Changed

- Bounding box color is now orange and covers lungs' contours.

## [1.1.0] - 2023-05-16

### Added

- Include number of slices in summary.
- Parameter to choose if the centers of the masks or center of the contours will be used as positive prompts.
- Bounding box are now optional.
- Use bounding box to aid prediction.
- Notebook explaining coordinates conventions.
- Class to manage multiple masks.
- Summary generation process when multiple images are processed.
- Notebook describing prompting process.
- Dice coefficient in the results.
- Notebook describing the windowing process.
- Shell script to process multiple CT images.
- Shell script to transform NIfTI data into NumPy data.

### Changed

- When using bounding boxes, the negative prompt is now its center.
- Rename IoU as Jaccard.
- Refactor class for the slice masks. It now also contains the image slice.
- Script **process_images.sh** now allows for windowing operation to be selected.
- Script **process_images.sh** now includes all the images in Zenodo's dataset.
- Windowing is now optional.
- Scripts to save NIfTI images as NumPy arrays now swap axis, so we follow the rows, columns coordinates convention.
- Parametrize shell scripts.
- Use same calling strategy for shell scripts.
- Debug data is now saved in the same folder the image being processed.

### Removed

- Script **process_image.sh** is no longer necessary, it's been replaced by calling **process_image.py** with its parameters.

### Fixed

- Dice calculation error.
- Center of segment calculation when moving centroid inside mask.
- Include all lung's contours in the debug data.
- Avoid using hard-coded shape for arrays in code.
- Edge case when getting the center of a mask. If the seed (centroid) is outside the mask and the projections in X and Y axes do not cut the mask in any point, there is a crash. Solved using a random point inside the mask when this happens.

## [1.0.0] - 2023-04-23

### Added

- Results saving process, both raw data and statistical values.
- Script to create GIF from folder with images.
- Class to locate a centered point inside a mask.
- Notebook explaining how to fix the center of mass when it is outside the mask's boundaries.
- Notebook explaining how to extract slices from images and masks files.
- Helper class to show script execution summary.
- Helper class to save debug data.
- Ability to process all the slices in an image.
- Save SAM predictions to disk if debug enabled.
- Extended enumeration for SAM model.
- Notebook explaining image transformations.
- Project's first version.

### Changed

- Use all lungs as a positive prompt, and the slice center as a negative prompt.
- **README.md** has been improved.
- Load images, masks, image slice and masks slices in separate methods.
- Load the model in a separate function.
- The script **main.py** now works without transpose or swap operations.
- The notebook **image_transformations.ipynb** now loads a copy of the slice prepared for demonstration porpoises.
