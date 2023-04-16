# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Save SAM predictions to disk if debug enabled.
- Extended enumeration for SAM model.
- Transpose and rotation options to **nifti_to_numpy.py** script.
- Notebook explaining image transformations.
- Project's first version.

### Changed

- Load images, masks, image slice and masks slices in separate methods.
- Load the model in a separate function.
- The script **main.py** now works without transpose or swap operations.
- The notebook **image_transformations.ipynb** now loads a copy of the slice prepared for demonstration porpoises.
