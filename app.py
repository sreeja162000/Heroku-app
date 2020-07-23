import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import datetime
import re

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def process():

    def convert_date_to_ordinal(date):
        return date.toordinal()
    features=[]
    y=0
    pattern=r'^[a-zA-Z].'

    for x in request.form.values():
        if x=='positive':
            features.append(int('1'))
        elif x=='negative':
            features.append(int('-1'))
        elif x=='neutral':
            features.append(int('0'))
        elif re.search(pattern, x):
            y=x
        else:
            x=datetime.datetime.strptime(x,'%m/%d/%Y')

            features.append(convert_date_to_ordinal(x))

    #int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)

    output = datetime.datetime.fromordinal(int(prediction[0]))
    z=datetime.datetime(2050,12,12)

    if output==z:
        return render_template('index.html', prediction_text='Customer {} does not churn'.format(y))
    else:    
        return render_template('index.html', prediction_text='Churn date of {} is : {}'.format(y,output.date()))


if __name__ == "__main__":
    app.run(debug=True)
