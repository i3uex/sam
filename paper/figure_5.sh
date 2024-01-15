#!/usr/bin/bash

#MOVE_CENTROIDS_INSIDE_MASKS='False'
#INCLUDE_NEGATIVE_PROMPT='False'
#USE_BOUNDING_BOX='False'
#DEBUG_DRAW_BOUNDING_BOX='False'
#export MOVE_CENTROIDS_INSIDE_MASKS
#export INCLUDE_NEGATIVE_PROMPT
#export USE_BOUNDING_BOX
#export DEBUG_DRAW_BOUNDING_BOX
#
#conda run -n sam --no-capture-output \
#  python process_image.py \
#  --image_file_path working_data/image_coronacases_001.npy \
#  --masks_file_path working_data/masks_coronacases_001.npy \
#  --slice 177 \
#  --apply_windowing \
#  --use_bounding_box \
#  --debug
#cp working_data/debug/image_coronacases_001/slice_177_prediction.png paper/slice_177_prediction_1.png

#MOVE_CENTROIDS_INSIDE_MASKS='True'
#INCLUDE_NEGATIVE_PROMPT='False'
#USE_BOUNDING_BOX='False'
#DEBUG_DRAW_BOUNDING_BOX='False'
#export MOVE_CENTROIDS_INSIDE_MASKS
#export INCLUDE_NEGATIVE_PROMPT
#export USE_BOUNDING_BOX
#export DEBUG_DRAW_BOUNDING_BOX
#
#conda run -n sam --no-capture-output \
#  python process_image.py \
#  --image_file_path working_data/image_coronacases_001.npy \
#  --masks_file_path working_data/masks_coronacases_001.npy \
#  --slice 177 \
#  --apply_windowing \
#  --use_bounding_box \
#  --debug
#cp working_data/debug/image_coronacases_001/slice_177_prediction.png paper/slice_177_prediction_2.png
#
#MOVE_CENTROIDS_INSIDE_MASKS='True'
#INCLUDE_NEGATIVE_PROMPT='True'
#USE_BOUNDING_BOX='False'
#DEBUG_DRAW_BOUNDING_BOX='False'
#export MOVE_CENTROIDS_INSIDE_MASKS
#export INCLUDE_NEGATIVE_PROMPT
#export USE_BOUNDING_BOX
#export DEBUG_DRAW_BOUNDING_BOX
#
#conda run -n sam --no-capture-output \
#  python process_image.py \
#  --image_file_path working_data/image_coronacases_001.npy \
#  --masks_file_path working_data/masks_coronacases_001.npy \
#  --slice 177 \
#  --apply_windowing \
#  --use_bounding_box \
#  --debug
#cp working_data/debug/image_coronacases_001/slice_177_prediction.png paper/slice_177_prediction_3.png

MOVE_CENTROIDS_INSIDE_MASKS='True'
INCLUDE_NEGATIVE_PROMPT='True'
USE_BOUNDING_BOX='True'
DEBUG_DRAW_BOUNDING_BOX='True'
export MOVE_CENTROIDS_INSIDE_MASKS
export INCLUDE_NEGATIVE_PROMPT
export USE_BOUNDING_BOX
export DEBUG_DRAW_BOUNDING_BOX

conda run -n sam --no-capture-output \
  python process_image.py \
  --image_file_path working_data/image_coronacases_001.npy \
  --masks_file_path working_data/masks_coronacases_001.npy \
  --slice 177 \
  --apply_windowing \
  --use_bounding_box \
  --debug
cp working_data/debug/image_coronacases_001/slice_177_prediction.png letter/slice_177_prediction_4.png
