#!/usr/bin/bash

PYTHON="conda run -n sam --no-capture-output python"
PROCESS_IMAGE_SCRIPT="process_image.py"
JOIN_RESULTS_SCRIPT="join_results.py"
WORKING_DATA_PATH="working_data"
RESULTS_PATH=$WORKING_DATA_PATH/"results"

IMAGE_NAMES=(
  "coronacases_001"
  "coronacases_002"
  "coronacases_003"
  "coronacases_004"
  "coronacases_005"
  "coronacases_006"
  "coronacases_007"
  "coronacases_008"
  "coronacases_009"
  "coronacases_010"
)

for IMAGE_NAME in "${IMAGE_NAMES[@]}"; do
  $PYTHON $PROCESS_IMAGE_SCRIPT \
    --image_file_path $WORKING_DATA_PATH/image_"$IMAGE_NAME".npy \
    --masks_file_path $WORKING_DATA_PATH/masks_"$IMAGE_NAME".npy
done

$PYTHON $JOIN_RESULTS_SCRIPT --results_folder_path $RESULTS_PATH
