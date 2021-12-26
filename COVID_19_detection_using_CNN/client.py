import requests
import os



URL1 = "http://127.0.0.1:5000/upload_file"
URL2 = "http://127.0.0.1:5000/predict"
# breathing_deep = "test\\breathing-deep.wav"
# breathing_shallow = "test\\breathing-shallow.wav"
# cough_heavy = "test\\cough-heavy.wav"
# cough_shallow = "test\\cough-shallow.wav"
# count_fast = "test\\counting-fast.wav"
# count_slow = "test\\counting-normal.wav"
# vowel_a = "test\\vowel-a.wav"
# vowel_e = "test\\vowel-e.wav"
# vowel_o = "test\\vowel-o.wav"

folder_path = 'test'

if __name__ == "__main__":

    for file in os.listdir(folder_path):
        audio_file = open(os.path.join(folder_path, file), "rb")
        values = {"file": (os.path.join(folder_path, file), audio_file, "audio/wav")}
        res = requests.post(URL1, files=values)

    response = requests.get(URL2)
    data = response.json()
        
    print(f"The person is {data['label']}")