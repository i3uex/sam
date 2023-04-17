# To Do

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
- Improve **README.md**, including dataset and model information.
- Guess why is PyCharm having trouble creating **debug.log**.
