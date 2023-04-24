#!/usr/bin/bash

PYTHON='conda run -n sam --no-capture-output python'
SCRIPT='nifti_to_numpy.py'
DATASET_PATH='datasets/zenodo/'
IMAGES_PATH='COVID-19-CT-Seg_20cases'
MASKS_PATH='Lung_Mask'
WORKING_DATA_PATH='working_data'

$PYTHON $SCRIPT --input_file_path $DATASET_PATH/$IMAGES_PATH/coronacases_001.nii.gz --output_file_path $WORKING_DATA_PATH/image_coronacases_001.npy
$PYTHON $SCRIPT --input_file_path $DATASET_PATH/$MASKS_PATH/coronacases_001.nii.gz --output_file_path $WORKING_DATA_PATH/masks_coronacases_001.npy

$PYTHON $SCRIPT --input_file_path $DATASET_PATH/$IMAGES_PATH/coronacases_002.nii.gz --output_file_path $WORKING_DATA_PATH/image_coronacases_002.npy
$PYTHON $SCRIPT --input_file_path $DATASET_PATH/$MASKS_PATH/coronacases_002.nii.gz --output_file_path $WORKING_DATA_PATH/masks_coronacases_002.npy

$PYTHON $SCRIPT --input_file_path $DATASET_PATH/$IMAGES_PATH/coronacases_003.nii.gz --output_file_path $WORKING_DATA_PATH/image_coronacases_003.npy
$PYTHON $SCRIPT --input_file_path $DATASET_PATH/$MASKS_PATH/coronacases_003.nii.gz --output_file_path $WORKING_DATA_PATH/masks_coronacases_003.npy

$PYTHON $SCRIPT --input_file_path $DATASET_PATH/$IMAGES_PATH/coronacases_004.nii.gz --output_file_path $WORKING_DATA_PATH/image_coronacases_004.npy
$PYTHON $SCRIPT --input_file_path $DATASET_PATH/$MASKS_PATH/coronacases_004.nii.gz --output_file_path $WORKING_DATA_PATH/masks_coronacases_004.npy

$PYTHON $SCRIPT --input_file_path $DATASET_PATH/$IMAGES_PATH/coronacases_005.nii.gz --output_file_path $WORKING_DATA_PATH/image_coronacases_005.npy
$PYTHON $SCRIPT --input_file_path $DATASET_PATH/$MASKS_PATH/coronacases_005.nii.gz --output_file_path $WORKING_DATA_PATH/masks_coronacases_005.npy

$PYTHON $SCRIPT --input_file_path $DATASET_PATH/$IMAGES_PATH/coronacases_006.nii.gz --output_file_path $WORKING_DATA_PATH/image_coronacases_006.npy
$PYTHON $SCRIPT --input_file_path $DATASET_PATH/$MASKS_PATH/coronacases_006.nii.gz --output_file_path $WORKING_DATA_PATH/masks_coronacases_006.npy

$PYTHON $SCRIPT --input_file_path $DATASET_PATH/$IMAGES_PATH/coronacases_007.nii.gz --output_file_path $WORKING_DATA_PATH/image_coronacases_007.npy
$PYTHON $SCRIPT --input_file_path $DATASET_PATH/$MASKS_PATH/coronacases_007.nii.gz --output_file_path $WORKING_DATA_PATH/masks_coronacases_007.npy

$PYTHON $SCRIPT --input_file_path $DATASET_PATH/$IMAGES_PATH/coronacases_008.nii.gz --output_file_path $WORKING_DATA_PATH/image_coronacases_008.npy
$PYTHON $SCRIPT --input_file_path $DATASET_PATH/$MASKS_PATH/coronacases_008.nii.gz --output_file_path $WORKING_DATA_PATH/masks_coronacases_008.npy

$PYTHON $SCRIPT --input_file_path $DATASET_PATH/$IMAGES_PATH/coronacases_009.nii.gz --output_file_path $WORKING_DATA_PATH/image_coronacases_009.npy
$PYTHON $SCRIPT --input_file_path $DATASET_PATH/$MASKS_PATH/coronacases_009.nii.gz --output_file_path $WORKING_DATA_PATH/masks_coronacases_009.npy

$PYTHON $SCRIPT --input_file_path $DATASET_PATH/$IMAGES_PATH/coronacases_010.nii.gz --output_file_path $WORKING_DATA_PATH/image_coronacases_010.npy
$PYTHON $SCRIPT --input_file_path $DATASET_PATH/$MASKS_PATH/coronacases_010.nii.gz --output_file_path $WORKING_DATA_PATH/masks_coronacases_010.npy
