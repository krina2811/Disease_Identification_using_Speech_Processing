To develop respiratory_disease_detection_system, used ICBHI challange 2017 dataset. There is total 920 audio files. The dataset contains total 8 classes like COPD, URTI, LRTI, Bronchitis, Bronchiectasis, Asthma, Pneumonia, and healthy.

For DATASET: https://drive.google.com/drive/u/1/folders/1oMdx7M2tBiJ1x8cMJWsHvC3LLlS3UR0w

First step is data_preprocessing. First MFCCs features are extracted from the audio files. Mfccs features are stored into the json format shown in file data_preprocessing.py

Second part is model creation part. The extracted MFccs features are passed to the CNN model which classify the disease. The model creation and prediction part are shown into model.py.