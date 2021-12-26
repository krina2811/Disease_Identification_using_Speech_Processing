from flask import Flask, jsonify, request, redirect, render_template
import os
from covid_detection_service import covid_detection_service
from werkzeug.utils import secure_filename
import numpy as np
import json
import random

folder_path = 'client_file'
app = Flask(__name__)   ## create the flask application

probability, label_list = [], []
@app.route("/")
def root():
    return render_template("index.html")

@app.route("/upload_file", methods=['POST','GET'])
def upload_file():    
        file = request.files["audio_data"]
        print(file)
        # f = open(file, 'rb') 
        # file_name = str(random.randint(0, 100000))
        file_name = secure_filename(file.filename)        
        print("file name is.............", file_name)
        with open(os.path.join(folder_path, file_name +'.wav'), 'wb') as audio:
            file.save(audio)
        
        print('File(s) successfully uploaded')
        return redirect('/')

### domain_name/predict request is send to server
@app.route("/predict", methods=['GET' , 'POST'])
def predict():

    ## covid_detecteion_service called
    # cds = covid_detection_service()

    # ## get request in form of audio-files and save it
    # for file in os.listdir(folder_path):
    #     ## make prediction
    #     proba_value, label = cds.predict(os.path.join(folder_path, file))
    #     print(os.path.join(folder_path, file))
    #     probability.append(proba_value)
    #     label_list.append(label)
    #     ## remve the audio file
    

    # index = np.argmax(probability)
    # class_label = label_list[index]
    # positive = ['positive_asymp', 'positive_mild', 'positive_moderate']

    # if class_label in positive:
    #     class_label = "COVID-19-Positive"
    #     data = 1
    # else:
    #     class_label = "Healthy"
    #     data = 0

    # print("&&&&&&&&&&&&&&7", class_label)
    # send response back to client
    # situation = {'label': str(class_label)}
    # if class_label =='Healthy':
    #     data = 0
    # else:
    #     data= 1
    data = 0
    return render_template('submit.html', data = data)
    

if __name__ == "__main__":

    app.run(debug=True)
