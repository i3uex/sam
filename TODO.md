# To Do

- Use a different marker for the prompt. A cross instead of a star could be more adequate, but try with different shapes.
- Download only the needed model weights, instead of all of them.
  - Ideally, this should be done from the Python script, via a constant.
  - If `model = "default"`, the script should try to download the corresponding weights, if they were not already there.
- Image orientation is defined in NIfTI files and [NiBabel is able to work with it][nibabel_orientation]. However, we are not using these features. Try to do it when showing the images.
- Parallelize **process_image.py**.
- Wrapper to solve coordinates conventions disparities.

[nibabel_orientation]: https://nipy.org/nibabel/coordinate_systems.html "Coordinate systems and affines"

## Montgomery Dataset

- Something strange happens when downloading this dataset. Name resolution returns a series of IPs, but connection fails with most of them. As a result, download time is five times what it should be.
