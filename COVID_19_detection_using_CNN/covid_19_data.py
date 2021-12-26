import csv
import os
import pandas as pd
import librosa
import math
import json
import wave
import contextlib

SAMPLE_RATE = 22050
DURATION = 3
Total_Sample = SAMPLE_RATE * DURATION

def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=1024, hop_length=256, num_segments=5):

    data = {
            "id" : [],
            "mfcc" : [],
            "labels" : []
    }
    c = 0
    count = 0
    number_of_sample_per_segment = int(Total_Sample / num_segments)
    
    expected_mfcc_per_segment = math.ceil(number_of_sample_per_segment / hop_length)

    for root, dir, files in os.walk(dataset_path):
        for d in dir:
            print(d)
            if d not in d_less_2:              
                c += 1
                for file in os.listdir(os.path.join(root, d)):
                    if file.endswith(".wav"):
                        print(file)
                        
                        signal, sr = librosa.load(os.path.join(root, d, file), sr=SAMPLE_RATE)
                        label_string = disease_dict[d]
                        label_index = label_encoder[label_string]

                        for s in range(num_segments):
                            start_sample = number_of_sample_per_segment * s
                            finish_sample = number_of_sample_per_segment + start_sample

                            mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample], 
                                                        sr=sr, n_fft=n_fft, 
                                                        n_mfcc= n_mfcc, hop_length=hop_length)
                            
                            mfcc = mfcc.T

                            if len(mfcc) == expected_mfcc_per_segment:
                                data["id"].append(d)
                                data["mfcc"].append(mfcc.tolist())
                                data["labels"].append(label_index)
                print("countttttttttttttttttt", c)
              
    with open(json_path,'w') as fw:
        json.dump(data, fw, indent=4)




if __name__ == "__main__":

    dataset_path = "D:\\COVID_19_dataset\\final"
    json_path = "covid_19_data2.json"


    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(dataset_path):    
        for file in f:
            if '.wav' in file:
                files.append(os.path.join(r,file))

    d_less_2 = set()
    d_2_to_3 = set()
    d_3_to_4 = set()
    d_4_to_5 = set()
    d_gra_5 = set()

    
    duration1 = []
    for fname in files:
        with contextlib.closing(wave.open(fname,'r')) as f:
            d = fname.split("\\")[3]
            
            frames = f.getnframes()
            rate = f.getframerate()
            duration = math.ceil(frames / float(rate))
            
            if(duration == 0 or duration == 2 or duration == 1 or duration == 3):
                d_less_2.add(d)
                              
            elif(duration == 3):
                d_2_to_3.add(d)
                            
            elif(duration == 4):
                d_3_to_4.add(d)
                
            elif(duration == 5):
                d_4_to_5.add(d)
                           
            # else:
            #     d_gra_5.add(d)
            #     d_gra_51.append(d)
                
    # print(duration1)

    # print("duration less 2", len(d_less_2))
    # print("duration 2 to 3", len(d_2_to_3))
    # print("duration 3 to 4", len(d_3_to_4))
    # print("duration 4 to 5", len(d_4_to_5))
    # print("duration gra 5", len(d_gra_5))
    # exit()


    disease_dict = {}
    col = ['id','a','covid_status','ep','g','l_c','l_l','l_s','rU','asthma','cough','smoker','test','ht','cold','diabetes','um','ihd','bd','st','fever','ftg','mp','loss_of_smell','test_status','diarrhoea','cld','pneumonia']
    disease = pd.read_csv("https://raw.githubusercontent.com/iiscleap/Coswara-Data/master/combined_data.csv")
    ids = disease['id'].tolist()
    labels = disease['covid_status'].tolist()
    for id, label in zip(ids, labels):
        disease_dict[id] = label
    
    label_encoder ={'healthy' : 0,
                    'no_resp_illness_exposed' : 1,
                    'positive_asymp' : 2,
                    'positive_mild' : 3,
                    'positive_moderate' : 4,
                    'recovered_full' : 5,
                    'resp_illness_not_identified' : 6}

    save_mfcc(dataset_path, json_path)




    ## first json was hop = 512, fft = 2024, segment = 5 and romove(0, 1 2 3 4), duration = 4
    ## second json was hop = 256, fft = 1024, segment = 5 and remove(0,1,2,3), duration = 3
    ## third json was hop = 256, fft = 1024, segment = 3 and remove(0, 1,2,3, 4), duration = 4
