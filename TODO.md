# To Do

- Fix errors in **process_image.py**.
  - I'm committing the changes just to have a backup.
- Get a center of mass that is inside the lung mask. Is this the centroid?
  - Use more than a point to prompt each lung.
- Maybe I should mark both lungs as foreground, and a point outside them as background?
- Add method to select ideal segmentation.
- Add method to calculate IoU between ground-truth and prediction.
- Store results in a CSV file.
- If all the slices are to be processed, say so in the summary.
- Download only the needed model weights, instead of all of them.
  - Ideally, this should be done from the Python script, via a constant.
  - If `model = "default"`, the script should try to download the corresponding weights, if they were not already there.
- Don't save debug prompt image if no masks are present.
- Image orientation is defined in NIfTI files and [NiBabel is able to work with it][nibabel_orientation]. However, we are not using these features. Try to do it when showing the images.

## create_gif.py

- Move hard-coded values to variables.
- Configure through parameters.
- Improve documentation.

[nibabel_orientation]: https://nipy.org/nibabel/coordinate_systems.html "Coordinate systems and affines"
