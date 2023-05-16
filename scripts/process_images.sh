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
  "radiopaedia_4_85506_1 false"
  "radiopaedia_7_85703_0 false"
  "radiopaedia_10_85902_1 false"
  "radiopaedia_10_85902_3 false"
  "radiopaedia_14_85914_0 false"
  "radiopaedia_27_86410_0 false"
  "radiopaedia_29_86490_1 false"
  "radiopaedia_29_86491_1 false"
  "radiopaedia_36_86526_0 false"
  "radiopaedia_40_86625_0 false"
)

for IMAGE_ITEM in "${IMAGE_ITEMS[@]}"; do
  read -r -a IMAGE_ITEM_ARRAY <<< "$IMAGE_ITEM"
  IMAGE_NAME="${IMAGE_ITEM_ARRAY[$IMAGE_NAME_INDEX]}"
  APPLY_WINDOWING="${IMAGE_ITEM_ARRAY[$APPLY_WINDOWING_INDEX]}"
  if [ "$APPLY_WINDOWING" = true ]; then
    $PYTHON $PROCESS_IMAGE_SCRIPT \
      --image_file_path $WORKING_DATA_PATH/image_"$IMAGE_NAME".npy \
      --masks_file_path $WORKING_DATA_PATH/masks_"$IMAGE_NAME".npy \
      --apply_windowing \
      --use_bounding_box \
      --debug
  else
    $PYTHON $PROCESS_IMAGE_SCRIPT \
      --image_file_path $WORKING_DATA_PATH/image_"$IMAGE_NAME".npy \
      --masks_file_path $WORKING_DATA_PATH/masks_"$IMAGE_NAME".npy \
      --use_bounding_box \
      --debug
  fi
done

$PYTHON $JOIN_RESULTS_SCRIPT --results_folder_path $RESULTS_PATH
