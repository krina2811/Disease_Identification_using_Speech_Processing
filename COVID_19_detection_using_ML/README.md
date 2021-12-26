# COVID_19_detection_using_ML
This project is mainly focus COVID-19 detection using the speech processing. So here first the features like mfccs, lpcs are extracted from the audio files. After that features will pass to the machine learning model (classifier).

FOR DATASET: https://drive.google.com/drive/u/1/folders/1oMdx7M2tBiJ1x8cMJWsHvC3LLlS3UR0w

In this project Coswara-Covid-19 dataset is used. Here each person need to record the 9 audio files. Breathing deep & shallow, Cough heavy & shallow, Count fast & slow and vowel a, e & o. Every person has different speech characterstics like pitch, volumn, rhythm, frequency etc. Base on this characterstic speech base features are extracted. 

This repositaory contains three files:

 covid_19.ipynb
 
 covid-19_mfcc_image.ipynb
 
 covid-19_using_lpc.ipynb
  
covid_19.ipynb -------> In this file first features like chroma stft, spectral centroid, spectral rolloff and mfccs like features are extracted. Then extracted features will pass to the machine learning models like k-NN and random forest base on different hyper parameters. 

covid-19_mfcc.ipynb ----> In this file mfcc images are extracted from the audio files.

covid-19_using_lpc.ipynb ----> In this file lpc features are extractd from the audio files.

SMOTE and RandomOverSampler are used for data balancing. GridsearchCV is used for hyper parameter tuning. 
  

