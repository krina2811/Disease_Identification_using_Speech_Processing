In this project, COSWARA Covid-19 dataset is used. 

FOR DATASET: https://drive.google.com/drive/u/1/folders/1oMdx7M2tBiJ1x8cMJWsHvC3LLlS3UR0w

Here, the first step is prepare the dataset. Each person has total 9 audio files. So at the first step we extract the MFCCs features from audio files. The MFCCs features are stored into json file. All the pre-processing steps are in covid_19_data.py.

The second step is model creation part. The extracted MFCCs features are passed to the CNN model. The CNN model predict the person is suffering from COVID_19 or not. The model creation and prediction part are in covid_19_model.py

The small GUI based application is created. In that user need to record the 9 audio files. When user click on submit button application shows user has COVID-19 or not. The server.py shows how the flask service is created. 

For GUI Reference: https://drive.google.com/drive/u/1/folders/1-LB037Fes53TOxq9xMn1ogoifaajv8lt