# Paper notes

This folder contains all the resources needed to generate the evidences (figures, statistical values, etcetera) the letter is based on.

## Figure 1

From the root folder of this project, execute the script **letter/figure_1.sh**. It will:

1. process the slice 122 from the image **coronacases_001.npy**, and copy it to the folder **letter**.

2. process the slice 171 from the image **radiopaedia_10_85902_3.npy**, and copy it to the folder **letter**.

Then, to prepare it for the letter, you need to remove the white frame from both. The resulting images will be 770 width by 770 height.

Rotate the coronacases one 180º, as it was originally upside down.

## Figure 2

From the root folder of this project, execute the script **letter/figure_2.sh**. It will:

1. Disable drawing SAM's prediction, masks contours, bounding box, and negative prompt.

2. Process the slice 98 from the image **coronacases_001.npy** while disabling moving the centroids inside the masks, and copy it to the folder **letter** with the name **centroid_outside.png**.

3. Process the slice 98 from the image **coronacases_001.npy**, moving the centroids inside the masks, and copy it to the folder **letter** with the name **centroid_inside.png**.

Then, to prepare it for the letter, you need to remove the white frame from both, and rotate them 180º, as they were originally upside down. The resulting images will be 770 width by 770 height.

## Table 1

To generate the data shown in Table 1:

1. Execute the script **scripts/process_images.sh** from the root folder of this project to process the whole COVID-19 dataset.

2. Copy the data from the file **working_data/results/results.csv**. This file will have a time stamp suffix.

## Figure 3

To create the box plot for figure 3, you first need to process the whole dataset. The folder **working_data/results** will contain a folder for each image, and three summary CSV files.

Now, create three folders:

- **1-all**: copy everything in **working_data/results** there.
- **2-coronacases**: copy only the folders for the corresponding images there.
- **3-radiopaedia**: copy only the folders for the corresponding images there.

Now, execute the script **join_results.py** for the folders **2-coronacases** and **3-radiopaedia** to create the summary files for each subset. There is a run/debug configuration in PyCharm for the script.

Lastly, execute the script **plots/create_boxplot.py**, changing the paths for each folder's result file. Use the resulting PDF as figure 3.

## Figure 4

From the root folder of this project, execute the script **letter/figure_4.sh**. It will:

1. Process the slice 85 from the image **coronacases_001.npy**, and copy it to the folder **letter** with the name **slice_85_prediction.png**.

2. Displace the negative prompt, and process again the slice 85 from the image **coronacases_001.npy**. Now, and copy it to the folder **letter** with the name **slice_85_prediction_improved.png**.

3. Write down the values returned for Jaccard index and Dice score for both slices. These values are in the letter.

Then, to prepare it for the letter, you need to remove the white frame from both, and rotate them 180º, as they were originally upside down. The resulting images will be 770 width by 770 height.

## Figure 5

From the root folder of this project, execute the script **letter/figure_5.sh**. It will:

1. Process the slice 177 from the image **coronacases_001.npy** without improving the centroids location, without using a negative prompt, and without using a bounding box, and copy it to the folder **letter** with the name **slice_177_prediction_1.png**.

2. Process the slice 177 from the image **coronacases_001.npy** improving the centroids location, without using a negative prompt, and without using a bounding box, and copy it to the folder **letter** with the name **slice_177_prediction_2.png**.

3. Process the slice 177 from the image **coronacases_001.npy** improving the centroids location, using a negative prompt, and without using a bounding box, and copy it to the folder **letter** with the name **slice_177_prediction_3.png**.

4. Process the slice 177 from the image **coronacases_001.npy** improving the centroids location, using a negative prompt, and using a bounding box, and copy it to the folder **letter** with the name **slice_177_prediction_4.png**.

5. Write down the values returned for Jaccard index and Dice score for each slice. These values are in the letter.

Then, to prepare it for the letter, you need to remove the white frame from all of them, and rotate them 180º, as they were originally upside down. The resulting images will be 770 width by 770 height.
