# Paper notes

This folder contains all the resources needed to generate the evidences (figures, statistical values, etcetera) the letter is based on.

## Figure 1

From the root folder of this project, execute the script **letter/1_figure_1.sh**. It will:

1. process the slice 122 from the image **coronacases_001.npy**, and copy it to the folder **letter**.

2. process the slice 171 from the image **radiopaedia_10_85902_3.npy**, and copy it to the folder **letter**.

Then, to prepare it for the letter, you need to remove the white frame from both. The resulting images will be 770 width by 770 height.

Rotate the coronacases one 180º, as it was originally upside down.

## Figure 2

From the root folder of this project, execute the script **letter/2_figure_2.sh**. It will:

1. Disable drawing SAM's prediction, masks contours, bounding box, and negative prompt.

2. Process the slice 98 from the image **coronacases_001.npy** while disabling moving the centroids inside the masks, and copy it to the folder **letter** with the name **centroid_outside.png**.

3. Process the slice 98 from the image **coronacases_001.npy**, moving the centroids inside the masks, and copy it to the folder **letter** with the name **centroid_inside.png**.

Then, to prepare it for the letter, you need to remove the white frame from both, and rotate them 180º, as they were originally upside down. The resulting images will be 770 width by 770 height.

## Table 1

To generate the data shown in Table 1:

1. Execute the script **scripts/process_images.sh** from the root folder of this project to process the whole Zenodo dataset.

2. Copy the data from the file **working_data/results/results.csv**. This file will have a time stamp suffix.

