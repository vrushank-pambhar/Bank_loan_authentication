from flasgger import Swagger
from flask import Flask,request
import pandas as pd
import numpy as np 
import pickle

app=Flask(__name__)
Swagger(app)

pickle_in=open('classifier.pkl','rb')
classifier=pickle.load(pickle_in)

@app.route('/')
def welcome():
    return "welcome all"

@app.route('/predict')
def predict_note_authentication():
    """let's Authenticate Bank note
    this is using docstings using specification
    ---
    parameters:
        - name: variance
          in: query
          type: number
          required: true
        - name: skewness
          in: query
          type: number
          required: true    
        - name: curtosis
          in: query
          type: number
          required: true
        - name: entropy
          in: query
          type: number
          required: true
    responses:
        200:
            description: the output values
    """        
    variance=request.args.get('variance')
    skewness=request.args.get('skewness')
    curtosis=request.args.get('curtosis')
    entropy=request.args.get('entropy')
    prediction=classifier.predict([[variance,skewness,curtosis,entropy]])
    print(prediction)
    return 'the prediction is'+ str(prediction)

@app.route('/predict_file',methods=["POST"])
def predict_note_file():
    """let's Authenticate Bank note
    this is using docstings using specification
    ---
    parameters:
        - name: file
          in: formdata
          type: file
          required: true
    responses:
        200:
            description: the output values
    """ 
    df_test=pd.read_csv(request.files.get('file'))
    print(df_test.head())
    prediction=classifier.predict(df_test)
    return str(list(prediction))


if __name__=='__main__':
    app.run()
