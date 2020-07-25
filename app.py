from flask import Flask,render_template,url_for,request
import pandas as pd 

import pickle

# load the model from disk
loaded_model=pickle.load(open('random_forest_regression_model.pkl', 'rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    df=pd.read_csv('real_2017.csv')
    df=df.dropna()
    df1 = df.iloc[1060:1065,2:-1]
    df1=df1.dropna()
    my_prediction=loaded_model.predict(df1.values)
    my_prediction=my_prediction.tolist()
    return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)
