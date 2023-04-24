#!/usr/bin/bash

PYTHON='conda run -n sam --no-capture-output python'
SCRIPT='process_image.py'
WORKING_DATA_PATH='working_data'

$PYTHON $SCRIPT --image_file_path $WORKING_DATA_PATH/image_coronacases_001.npy --masks_file_path $WORKING_DATA_PATH/masks_coronacases_001.npy --debug
$PYTHON $SCRIPT --image_file_path $WORKING_DATA_PATH/image_coronacases_002.npy --masks_file_path $WORKING_DATA_PATH/masks_coronacases_002.npy --debug
$PYTHON $SCRIPT --image_file_path $WORKING_DATA_PATH/image_coronacases_003.npy --masks_file_path $WORKING_DATA_PATH/masks_coronacases_003.npy --debug
$PYTHON $SCRIPT --image_file_path $WORKING_DATA_PATH/image_coronacases_004.npy --masks_file_path $WORKING_DATA_PATH/masks_coronacases_004.npy --debug
$PYTHON $SCRIPT --image_file_path $WORKING_DATA_PATH/image_coronacases_005.npy --masks_file_path $WORKING_DATA_PATH/masks_coronacases_005.npy --debug
$PYTHON $SCRIPT --image_file_path $WORKING_DATA_PATH/image_coronacases_006.npy --masks_file_path $WORKING_DATA_PATH/masks_coronacases_006.npy --debug
$PYTHON $SCRIPT --image_file_path $WORKING_DATA_PATH/image_coronacases_007.npy --masks_file_path $WORKING_DATA_PATH/masks_coronacases_007.npy --debug
$PYTHON $SCRIPT --image_file_path $WORKING_DATA_PATH/image_coronacases_008.npy --masks_file_path $WORKING_DATA_PATH/masks_coronacases_008.npy --debug
$PYTHON $SCRIPT --image_file_path $WORKING_DATA_PATH/image_coronacases_009.npy --masks_file_path $WORKING_DATA_PATH/masks_coronacases_009.npy --debug
$PYTHON $SCRIPT --image_file_path $WORKING_DATA_PATH/image_coronacases_010.npy --masks_file_path $WORKING_DATA_PATH/masks_coronacases_010.npy --debug
