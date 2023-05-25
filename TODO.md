# To Do

- Fix statistical values calculation when combining results. The values obtained for each image are correct, but when aggregating them to obtain the results for the whole datasets, the calculations are wrong.
- Use a different marker for the prompt. A cross instead of a star could be more adequate, but try with different shapes.
- Update notebooks to better describe the reality of the project.
  - **image_transformations.ipynb** works, but **process_image.py** doesn't do that.
- Download only the needed model weights, instead of all of them.
  - Ideally, this should be done from the Python script, via a constant.
  - If `model = "default"`, the script should try to download the corresponding weights, if they were not already there.
- Image orientation is defined in NIfTI files and [NiBabel is able to work with it][nibabel_orientation]. However, we are not using these features. Try to do it when showing the images.
- Parallelize **process_image.py**.
- Wrapper to solve coordinates conventions disparities.

[nibabel_orientation]: https://nipy.org/nibabel/coordinate_systems.html "Coordinate systems and affines"
