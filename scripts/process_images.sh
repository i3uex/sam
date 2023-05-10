#!/usr/bin/bash

PYTHON="conda run -n sam --no-capture-output python"
PROCESS_IMAGE_SCRIPT="process_image.py"
JOIN_RESULTS_SCRIPT="join_results.py"
WORKING_DATA_PATH="working_data"
RESULTS_PATH=$WORKING_DATA_PATH/"results"
IMAGE_NAME_INDEX=0
APPLY_WINDOWING_INDEX=1

IMAGE_ITEMS=(
  "coronacases_001 true"
  "coronacases_002 true"
  "coronacases_003 true"
  "coronacases_004 true"
  "coronacases_005 true"
  "coronacases_006 true"
  "coronacases_007 true"
  "coronacases_008 true"
  "coronacases_009 true"
  "coronacases_010 true"
)

for IMAGE_ITEM in "${IMAGE_ITEMS[@]}"; do
  read -r -a IMAGE_ITEM_ARRAY <<< "$IMAGE_ITEM"
  IMAGE_NAME="${IMAGE_ITEM_ARRAY[$IMAGE_NAME_INDEX]}"
  APPLY_WINDOWING="${IMAGE_ITEM_ARRAY[$APPLY_WINDOWING_INDEX]}"
  if [ "$APPLY_WINDOWING" = true ]; then
    $PYTHON $PROCESS_IMAGE_SCRIPT \
      --image_file_path $WORKING_DATA_PATH/image_"$IMAGE_NAME".npy \
      --masks_file_path $WORKING_DATA_PATH/masks_"$IMAGE_NAME".npy \
      --apply_windowing
  else
    $PYTHON $PROCESS_IMAGE_SCRIPT \
      --image_file_path $WORKING_DATA_PATH/image_"$IMAGE_NAME".npy \
      --masks_file_path $WORKING_DATA_PATH/masks_"$IMAGE_NAME".npy
  fi
done

$PYTHON $JOIN_RESULTS_SCRIPT --results_folder_path $RESULTS_PATH
