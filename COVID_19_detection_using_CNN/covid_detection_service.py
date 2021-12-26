import tensorflow.keras as keras
import librosa
import numpy as np
import math
import warnings, os
import time
warnings.simplefilter("default") # Change the filter in this process
os.environ["PYTHONWARNINGS"] = "default"
deprecation_warnings=False


MODEL_PATH = 'covid.h5'
NUM_SAMPLES_TO_CONSIDER = 22050  # 1 sec

SAMPLE_RATE = 22050
DURATION = 3
Total_Sample = SAMPLE_RATE * DURATION
num_segments = 5

class _covid_detection_service:

    model = None
    _mappings = ['healthy',
                'no_resp_illness_exposed',
                'positive_asymp',
                'positive_mild',
                'positive_moderate',
                'recovered_full',
                'resp_illness_not_identified']

    _instance = None
    
    def preprocess(self, file_path, n_mfcc =13, n_fft=1024, hop_length=256):

        signal, rate = librosa.load(file_path)
        number_of_sample_per_segment = int(Total_Sample / num_segments)
    
        expected_mfcc_per_segment = math.ceil(number_of_sample_per_segment / hop_length)

        if len(signal) > number_of_sample_per_segment:
            signal = signal[:number_of_sample_per_segment]
        
        mfccs = librosa.feature.mfcc(signal, n_mfcc=13, n_fft=1024, hop_length=256 )

        return mfccs.T

    def predict(self, file_path):

        #extract mfcc
        MFCCs = self.preprocess(file_path)  #(segments, coff)

        #convert 2d mfccs array into 4d array: (samples, segments, coff, ch)
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

        # make prediction
        prediction_array = self.model.predict(MFCCs)
        prediction_index = np.argmax(prediction_array)

        # return self._mappings[prediction_index]
        
        value = prediction_array[0][prediction_index]
        
        return value, self._mappings[prediction_index]



def covid_detection_service():

    if _covid_detection_service._instance is None:
        _covid_detection_service._instance = _covid_detection_service()
    _covid_detection_service.model = keras.models.load_model(MODEL_PATH)
    return _covid_detection_service._instance

if __name__ == "__main__":

    
    cds = covid_detection_service()
    proba, index = [], []

    start = time.time()
    breath_deep, breath_deep_value = cds.predict("D:\\COVID_19_dataset\\final\\ffyNrzu5ZGXrZ0I7MbxBwqbtIln1\\breathing-deep.wav")
    proba.append(breath_deep)
    index.append(breath_deep_value)


    breath_shallow, breath_shallow_value = cds.predict("D:\\COVID_19_dataset\\final\\ffyNrzu5ZGXrZ0I7MbxBwqbtIln1\\breathing-shallow.wav")
    proba.append(breath_shallow)
    index.append(breath_shallow_value)

    cough_heavy, cough_heavy_value = cds.predict("D:\\COVID_19_dataset\\final\\ffyNrzu5ZGXrZ0I7MbxBwqbtIln1\\cough-heavy.wav")
    proba.append(cough_heavy)
    index.append(cough_heavy_value)

    cough_shallow, cough_shallow_value = cds.predict("D:\\COVID_19_dataset\\final\\ffyNrzu5ZGXrZ0I7MbxBwqbtIln1\\cough-shallow.wav")
    proba.append(cough_shallow)
    index.append(cough_shallow_value)

    count_fast, counting_fast_value = cds.predict("D:\\COVID_19_dataset\\final\\ffyNrzu5ZGXrZ0I7MbxBwqbtIln1\\counting-fast.wav")
    proba.append(count_fast)
    index.append(counting_fast_value)

    count_shallow, count_shallow_value = cds.predict("D:\\COVID_19_dataset\\final\\ffyNrzu5ZGXrZ0I7MbxBwqbtIln1\\counting-normal.wav")
    proba.append(count_shallow)
    index.append(count_shallow_value)

    vowel_a, vowel_a_value = cds.predict("D:\\COVID_19_dataset\\final\\ffyNrzu5ZGXrZ0I7MbxBwqbtIln1\\vowel-a.wav")
    proba.append(vowel_a)
    index.append(vowel_a_value)

    vowel_e, vowel_e_value = cds.predict("D:\\COVID_19_dataset\\final\\ffyNrzu5ZGXrZ0I7MbxBwqbtIln1\\vowel-e.wav")
    proba.append(vowel_e)
    index.append(vowel_e_value)

    vowel_o, vowel_o_value = cds.predict("D:\\COVID_19_dataset\\final\\ffyNrzu5ZGXrZ0I7MbxBwqbtIln1\\vowel-o.wav")
    proba.append(vowel_o)
    index.append(vowel_o_value)
    end = time.time()
    print("model took:{}".format(end-start))
    id = np.argmax(proba)
    print(index[id])
    print(proba)
    print(index)
    class_label = index[id]
    
    positive = ['positive_asymp', 'positive_mild', 'positive_moderate']
    if class_label in positive:
        class_label = "COVID-19 Positive"
    else:
        class_label = "Healthy"
    print("The person is {}".format(class_label))
