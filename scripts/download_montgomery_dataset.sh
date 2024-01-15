#!/usr/bin/bash

# Create folder for the datasets, if it doesn't exist

mkdir -p ./datasets

# Download and unpack Montgomery's dataset
# We are only CTs and lung masks

cd ./datasets || exit
mkdir ./montgomery
cd ./montgomery || exit

wget https://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip

unzip NLM-MontgomeryCXRSet.zip -d NLM-MontgomeryCXRSet

rm NLM-MontgomeryCXRSet.zip

# Tidy content up

mv NLM-MontgomeryCXRSet/MontgomerySet/CXR_png .
mv NLM-MontgomeryCXRSet/MontgomerySet/ManualMask .
mv NLM-MontgomeryCXRSet/MontgomerySet/NLM-MontgomeryCXRSet-ReadMe.pdf .

rm -Rf NLM-MontgomeryCXRSet/

rm -f CXR_png/Thumbs.db
rm -f ManualMask/.DS_Store
rm -f ManualMask/leftMask/Thumbs.db
