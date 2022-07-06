
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import sklearn
import pickle

app = Flask(__name__)
f = open('LinearRegressionModel.pkl', 'rb')
model = pickle.load(f)
car = pd.read_csv("Cleaned_dataset_car.csv")

@app.route('/')
def index():
    companies= sorted(car['company'].unique())
    car_models= sorted(car['name'].unique())
    years= sorted(car['year'].unique(), reverse= True)
    fuel_type= sorted(car['fuel_type'].unique())
    companies.insert(0,"Select Company")

    return render_template('index.html',
                           companies= companies,
                           car_models= car_models,
                           years = years,
                           fuel_type=fuel_type)

@app.route('/predict', methods = ['POST'])
def predict():
    company = request.form.get('company')
    car_model = request.form.get('car_model')
    year = request.form.get('year')
    fuel_type = request.form.get('fuel_type')
    kms_driven = int(request.form.get('kilo_driven'))

    prediction=model.predict(pd.DataFrame([[car_model,company,year,kms_driven,fuel_type]],
                                          columns=['name','company','year','kms_driven','fuel_type']))
    prediction_value = round(prediction[0],2)
    prediction_value = str(prediction_value)
    return prediction_value

if __name__ == "__main__":
    app.run(debug = True)