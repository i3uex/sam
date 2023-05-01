# To Do

- Improve segmentation:
  - Use contours' bounding boxes as prompts, besides the center point.
    - Explained in section "Combining points and boxes" of the notebook ["Object masks from prompts with SAM"][sam_notebook].
  - Use two more points as prompts. Divide the contour in three parts, find the center of each one of them.
- Update notebooks to better describe the reality of the project.
  - **image_transformations.ipynb** works, but **process_image.py** doesn't do that.
- Use more than a point to prompt each lung.
- If all the slices are to be processed, say so in the summary.
- Download only the needed model weights, instead of all of them.
  - Ideally, this should be done from the Python script, via a constant.
  - If `model = "default"`, the script should try to download the corresponding weights, if they were not already there.
- Image orientation is defined in NIfTI files and [NiBabel is able to work with it][nibabel_orientation]. However, we are not using these features. Try to do it when showing the images.
- Parallelize **process_image.py**.

## create_gif.py

- Move hard-coded values to variables.
- Configure through parameters.
- Improve documentation.

[sam_notebook]: https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb "Object masks from prompts with SAM"
[nibabel_orientation]: https://nipy.org/nibabel/coordinate_systems.html "Coordinate systems and affines"
