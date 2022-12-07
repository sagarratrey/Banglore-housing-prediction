
import pandas as pd
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
data = pd.read_excel('cleaned_data.xlsx')
pipe = pickle.load(open('bangalore_housing_pred.pkl', 'rb'))

@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html' , locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get('location')
    BHK = request.form.get('BHK')
    bath = request.form.get('bath')
    sqft = request.form.get('total_sqft')

    print(location, BHK, bath, sqft)
    input = pd.DataFrame([[location,sqft,bath,BHK]], columns=['location', 'total_sqft', 'bath','BHK'])
    prediction = pipe.predict(input)[0]

    return str(prediction)

if __name__ == "__main__":
    app.run(debug=True, port=5001)


