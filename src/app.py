from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html') 
    else:
        data=CustomData(
        ID=int(request.form.get('ID')),
        N_Days=int(request.form.get('N_Days')),
        Drug=int(request.form.get('Drug')),
        Age=int(request.form.get('Age')),
        Sex=int(request.form.get('Sex')),
        Ascites=int(request.form.get('Ascites')),
        Hepatomegaly=int(request.form.get('Hepatomegaly')),
        Spiders=int(request.form.get('Spiders')),
        Edema=int(request.form.get('Edema')),
        Bilirubin=float(request.form.get('Bilirubin')),
        Cholesterol=float(request.form.get('Cholesterol')),
        Albumin=float(request.form.get('Albumin')),
        Copper=float(request.form.get('Copper')),
        Alk_Phos=float(request.form.get('Alk_Phos')),
        SGOT=float(request.form.get('SGOT')),
        Tryglicerides=float(request.form.get('Tryglicerides')),
        Platelets=float(request.form.get('Platelets')),
        Prothrombin=float(request.form.get('Prothrombin')),
        Stage=float(request.form.get('Stage'))
        )

        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)    