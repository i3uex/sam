# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Shell script to process multiple CT images.
- Shell script to transform NIfTI data into NumPy data.

### Changed

- Parametrize shell scripts.
- Use same calling strategy for shell scripts.
- Debug data is now saved in the same folder the image being processed is.

### Removed

- Script **process_image.sh** is no longer necessary, it's been replaced by calling **process_image.py** with its parameters.

### Fixed

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
