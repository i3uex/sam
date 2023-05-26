# Paper notes

This folder contains all the resources needed to generate the evidences (figures, statistical values, etcetera) the letter is based on.

### Figure 1

From the root folder of this project, execute the script `letter/1_figure_1.sh`. It will:

1. process the slice 122 from the image **coronacases_001.npy**, and copy it to the folder **letter**.

2. process the slice 171 from the image **radiopaedia_10_85902_3.npy**, and copy it to the folder **letter**.

Then, to prepare it for the letter, you need to remove the white frame from both. The resulting images will be 770 width by 770 height.

Rotate the coronacases one 180º, as it was originally upside down.
