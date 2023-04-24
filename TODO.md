# To Do

- Fix crash with image 8 (**image_coronacases.npy**), slice 26.
- Update notebooks to better describe the reality of the project.
  - **image_transformations.ipynb** works, but **process_image.py** doesn't do that.
- Use more than a point to prompt each lung.
- If all the slices are to be processed, say so in the summary.
- Download only the needed model weights, instead of all of them.
  - Ideally, this should be done from the Python script, via a constant.
  - If `model = "default"`, the script should try to download the corresponding weights, if they were not already there.
- Image orientation is defined in NIfTI files and [NiBabel is able to work with it][nibabel_orientation]. However, we are not using these features. Try to do it when showing the images.
- Parametrize **nifti_to_numpy.sh** script.
- Parametrize **process_images.sh** script.

## create_gif.py

- Move hard-coded values to variables.
- Configure through parameters.
- Improve documentation.

[nibabel_orientation]: https://nipy.org/nibabel/coordinate_systems.html "Coordinate systems and affines"
