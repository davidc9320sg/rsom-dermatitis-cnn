# Model Learning Analysis of 3D Optoacoustic Mesoscopic Images for the Classification of Atopic Dermatitis
This repository hosts the code used in the manuscript "**Model Learning Analysis of 3D Optoacoustic Mesoscopic Images for the Classification of Atopic Dermatitis**" (submitted for review)
by
Sojeong Park (1)\*, 
Shier Nee Saw (1)\*,
Xiuting Li (2)\*
et al. (\* co-first authors)

(1) Bioinformatics Institute, Agency of Science, Technology and Research, A\*STAR, 30 Biopolis Street, #07-01 Matrix, 138671, Singapore.

(2) Laboratory of Bio-Optical Imaging, Singapore Bioimaging Consortium, A\*STAR, 11 Biopolis Way, 138667, Singapore

# Structure
## Classic ML
The folder ```classic_ml``` contains the script used for the experiments that employ classical machine learning algorithm, namely Random Forest (RF) and Support Vector Machines (SVM) as mentioned in the manuscript.

## CNN
The folder ```cnn``` contains the script used for the three experiments - (i)CNN-bottleneck_LFHF, (ii) CNN-bottleneck_LFHF_3features and (iii) CNN-bottleneck_LFHF_features. The third folder "CNN-bottleneck_LFHF_features" is an experiment using RSOM images and all four features, as mentioned in the manuscript. 

1. main_GenerateCSVFiles.py - this is to generate the CSV file that contain the list of case that used in each cross validation
2. main_GenerateData.py.py - this is to generate RSOM data which include cropping and augmenting used as input to the CNN model. 
3. main_GenerateSpecificData.py - this is to generate additional data during evaluation procedure. As stated in manuscript, when a patient does not have final prediction due to having equal number of prediction of its' samples, one additional samples of that patient will be generated to obtain the final prediction. 
4. main_generate_confusion_matrix.py - this code is used to to obtain confusion matrix. 


## Skin Surface
The folder ```skin_surface``` contains the script used to detect the skin surface for cropping.  
