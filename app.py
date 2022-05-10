#///////////////////////////////////////////////////////////Import Libs////////////////////////////////////////////////////////////////////////////////////
import os
from flask import Flask, flash, url_for, request, jsonify, render_template, redirect
import pickle
import numpy as np
from PIL import Image
import glob
import tensorflow as tf
import pandas as pd

#/////////////////////////////////////////////////////////////////////Global Vars//////////////////////////////////////////////////////////////////////////   
UPLOAD_FOLDER = 'data/'
IMAGE_MODEL_FOLDER = 'models/image_model/'
NLP_MODEL_PATH1= 'models/nlp_model/rf_fe.pickle'
NLP_MODEL_PATH2= 'models/nlp_model/rf_nlp.pickle'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

dic = {'bab':'BABBLING', 'gue':'GUESS', 'uni':'UNINTELLIGIBLE', 'rep':'REPETITION', 'inq':'INCOMPLETION', 'ono': 'ONOMATOPOEIA', 'hes':'HESITATION', 'disf':'DISFLUENCY', 'mis':'MISSPELING'}
annotations_dic = {'bab': 0, 'gue': 0, 'uni': 0, 'rep': 0, 'inq': 0, 'ono': 0, 'hes': 0, 'mis': 0, 'disf': 0}
annotation_list =['bab', 'gue', 'uni', 'rep', 'inq', 'ono', 'hes', 'mis', 'disf']
pos_tags_to_keep = ['ADJ','ADV','AUX','INTJ','NOUN','PRON','ADJ','VERB']

#//////////////////////////////////////////////////////////////////Upload pretrained models/////////////////////////////////////////////////////////////////
image_model = tf.keras.models.load_model(IMAGE_MODEL_FOLDER)
text_model_1 = pickle.load(open(NLP_MODEL_PATH1, 'rb'))
text_model_2 = pickle.load(open(NLP_MODEL_PATH2, 'rb'))

#/////////////////////////////////////////////////////////////////////////Flask App//////////////////////////////////////////////////////////////////////
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#**********************************Predefined Functions*************************************

from modules import app_utils

#*****************************************app.route****************************************
@app.route("/")
def home():
    return render_template('home.html')

@app.route("/main")
def main():
    return render_template('main.html')

@app.route("/check", methods=['GET','POST'])
def check():
    if request.method == 'POST':
        #access the data from the main page
        if "name" in request.form :
            app_utils.clear_data_content()
            child_name = str(request.form["name"])
            birth_date =  str(request.form["birthdate"])
            sex =  str(request.form["sex"])
            app_utils.csv_register(child_name, birth_date, sex)
            return render_template("check.html", name = child_name, birth_date = birth_date, sex = sex)
        #get data from the saved csv file
        if "data_child.csv" in os.listdir(UPLOAD_FOLDER) :
            df_child = pd.read_csv("data/data_child.csv")
            child_name = df_child["name"][0]
            birth_date =  df_child["birth_date"][0]
            sex =  df_child["sex"][0]
        #get the uploaded picture from the check page	
        if "file" in request.files :
            file = request.files['file']
            # If the user does not select a file, the browser submits an
            # empty file without a filename.
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if file and app_utils.allowed_file(file.filename):
                filename = file.filename
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                return render_template("check.html", image_loaded = "Image loaded !", name = child_name)
        #get the speech text
        if 'speech' in request.form:
            speech = request.form['speech']
            if speech =='':
                flash('No detected text')
                return redirect(request.url)
            if speech != '':
                app_utils.speech_register(child_name, birth_date, sex, speech)
                return render_template("check.html", speech_loaded = "Speech saved !", name = child_name)
    # print(error message)
    return render_template("failure.html")


@app.route("/result", methods=['GET','POST'])
def result():
    if request.method == 'POST' :
        #///////////////////////////////////////////////////////////// photo-based prediction /////////////////////////////////
        # convert png file into jpg and delete png
        png_files = glob.glob(UPLOAD_FOLDER + "/*.png")
        if png_files :
            with Image.open(png_files[0]) as image :
                image.save(png_files[0][:-3]+"jpg")
            for f in os.listdir(UPLOAD_FOLDER):
                if f.endswith(".png") :
                    os.remove(os.path.join(UPLOAD_FOLDER, f))
                    
        # convert jpeg file into jpg and delete jpeg
        jpeg_files = glob.glob(UPLOAD_FOLDER + "/*.jpeg")
        if jpeg_files :
            with Image.open(jpeg_files[0]) as image :
                image.save(jpeg_files[0][:-4]+"jpg")
            for f in os.listdir(UPLOAD_FOLDER):
                if f.endswith(".jpeg") :
                    os.remove(os.path.join(UPLOAD_FOLDER, f))
        #load the saved image
        jpg_files = glob.glob(UPLOAD_FOLDER + "/*.jpg")
        image_file = jpg_files[0]
        with Image.open(image_file) as image :
            image_array = np.array(image)
        #normalize and reshape the loaded image
        image_normalized = image_array/255
        image_array_exp = np.expand_dims(image_normalized, axis=0)
        image_ready = tf.image.resize(image_array_exp, [300,300])
        #make prediction
        image_pred = image_model.predict(image_ready)[0][0]
        print(image_pred)
        #///////////////////////////////////////////////////////////// text-based prediction part 1 /////////////////////////////////
        #check if we saved a speech text
        if "data_speech_child.csv" in os.listdir(UPLOAD_FOLDER):
           df = pd.read_csv('data/data_speech_child.csv')
           df = app_utils.extract_features(df, annotations_dic)
           X = df[['sex','age_months','len_speech','len_meaningful_speech', 'len_structured_speech', 'n_bab', 'n_gue',
                   'n_uni', 'n_rep', 'n_inq', 'n_ono', 'n_hes', 'n_mis', 'n_disf', 'n_diff_words', 'density']]
           text_prediction_1 = text_model_1.predict_proba(X)
                  #//////////////////////////////////////////////////////////////////text-based prediction part 2/////////////////////////////////////////////////
           text_prediction_2 = text_model_2 .predict_proba(df['speech'])
           mean_p_1=(text_prediction_1[0][1]+text_prediction_2[0][1] + image_pred )/3
                 #/////////////////////////////////////////////////////////////combine and return final results /////////////////////////////////
           name = df['name'][0]
           result = ''
           if mean_p_1 < 0.5:
               result = '{} is not at risk of having Autism Spectrum Disorder'.format(name)
           elif mean_p_1>0.5 and mean_p_1<0.7:
               result = '{} is at potential risk of having Autism Spectrum Disorder'.format(name)
           else:
               result = '{} is at high risk of having Autism Spectrum Disorder'.format(name)
           result = result
        return render_template('result.html', prediction_text= result, name = name)
    # print(error message)
    return render_template("failure.html")


if __name__ == "__main__":
    app.secret_key = os.urandom(12) # in order for image load to run
    app.run(debug=True)
