# Create folder for the datasets, if it doesn't exist

mkdir -p ./datasets

# Download and unpack Zenodo's dataset
# We are only CTs and lung masks

cd ./datasets || exit
mkdir ./zenodo
cd ./zenodo || exit

wget https://zenodo.org/record/3757476/files/COVID-19-CT-Seg_20cases.zip
wget https://zenodo.org/record/3757476/files/Lung_Mask.zip

unzip COVID-19-CT-Seg_20cases.zip -d COVID-19-CT-Seg_20cases
unzip Lung_Mask.zip -d Lung_Mask

rm COVID-19-CT-Seg_20cases.zip
rm Lung_Mask.zip
