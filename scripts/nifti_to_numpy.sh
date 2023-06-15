#!/usr/bin/bash

PYTHON="conda run -n sam --no-capture-output python"
SCRIPT="nifti_to_numpy.py"
DATASET_NAME="covid"
DATASETS_PATH="datasets"
DATASET_PATH=$DATASETS_PATH/$DATASET_NAME
IMAGES_PATH="COVID-19-CT-Seg_20cases"
MASKS_PATH="Lung_Mask"
WORKING_DATA_PATH="working_data"
OUTPUT_PATH=$WORKING_DATA_PATH/$DATASET_NAME

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
  "radiopaedia_4_85506_1"
  "radiopaedia_7_85703_0"
  "radiopaedia_10_85902_1"
  "radiopaedia_10_85902_3"
  "radiopaedia_14_85914_0"
  "radiopaedia_27_86410_0"
  "radiopaedia_29_86490_1"
  "radiopaedia_29_86491_1"
  "radiopaedia_36_86526_0"
  "radiopaedia_40_86625_0"
)

mkdir -p ./$WORKING_DATA_PATH

for IMAGE_NAME in "${IMAGE_NAMES[@]}"; do
  echo "Processing image \"$IMAGE_NAME\"..."
  $PYTHON $SCRIPT \
    --input_file_path $DATASET_PATH/$IMAGES_PATH/"$IMAGE_NAME".nii.gz \
    --output_file_path $OUTPUT_PATH/image_"$IMAGE_NAME".npz \
    --swap_axes
  echo "Processing masks \"$IMAGE_NAME\"..."
  $PYTHON $SCRIPT \
    --input_file_path $DATASET_PATH/$MASKS_PATH/"$IMAGE_NAME".nii.gz \
    --output_file_path $OUTPUT_PATH/masks_"$IMAGE_NAME".npz \
    --swap_axes
done
