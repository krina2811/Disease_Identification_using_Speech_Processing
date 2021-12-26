import csv
import os
import pandas as pd
import librosa
import math
import json


SAMPLE_RATE = 22050
DURATION = 15
Total_Sample = SAMPLE_RATE * DURATION

def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):

    data = {
            "id" : [],
            "mfcc" : [],
            "labels" : []
    }

    number_of_sample_per_segment = int(Total_Sample / num_segments)
    
    expected_mfcc_per_segment = math.ceil(number_of_sample_per_segment / hop_length)

    for file in os.listdir(dataset_path):
        signal, sr = librosa.load(os.path.join(dataset_path, file), sr=SAMPLE_RATE)
        index = int(file.split("_")[0])
        label_string = label_dict[index]
        label_index = label_encoder[label_string]
        print(file)
        for s in range(num_segments):
            start_sample = number_of_sample_per_segment * s
            finish_sample = number_of_sample_per_segment + start_sample

            mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample], 
                                        sr=sr, n_fft=n_fft, 
                                        n_mfcc= n_mfcc, hop_length=hop_length)
            
            mfcc = mfcc.T

            if len(mfcc) == expected_mfcc_per_segment:
                data["id"].append(index)
                data["mfcc"].append(mfcc.tolist())
                data["labels"].append(label_index)

    with open(json_path,'w') as fw:
        json.dump(data, fw, indent=4)


if __name__ == "__main__":

    dataset_path = "F:\\dataset\\Respiratory_Sound_Database\\Respiratory_Sound_Database\\files"
    json_path = "data.json"
    csv_file_path = "F:\\dataset\\Respiratory_Sound_Database\\Respiratory_Sound_Database\\patient_diagnosis.csv"
    df = pd.read_csv(csv_file_path, header=None)
    id = df.iloc[:, 0].tolist()
    labels = df.iloc[:,1].tolist()
    label_dict = {}
    for ids, labels in zip(id, labels):
        label_dict[ids] = labels

    label_encoder = {'Asthma': 0,
                    'Bronchiectasis': 1,
                    'Bronchiolitis': 2,
                    'COPD': 3,
                    'Healthy': 4,
                    'Pneumonia': 5,
                    'LRTI' : 6,
                    'URTI' : 7
                    }

    save_mfcc(dataset_path, json_path)