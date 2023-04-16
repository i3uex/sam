# To Do

- Download only the needed model weights, instead of all of them.
  - Ideally, this should be done from the Python script, via a constant.
  - If `model = "default"`, the script should try to download the corresponding weights, if they were not already there.
- Improve **README.md**, including dataset and model information.
- The notebook **image_transformations.ipynb** loads a slice from a CT that is no longer wrong. Save the slice to disk, so the notebook shows the problems.
