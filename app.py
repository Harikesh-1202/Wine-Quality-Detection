from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Load the Random Forest model
model = joblib.load('wine_quality_model.pkl')

# Define a route to render the HTML form
@app.route('/')

def one():
    return render_template('home_page.html')

# Define a route to handle form submission and make a prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the form
    param1 = float(request.form['param1'])
    param2 = float(request.form['param2'])
    param3 = float(request.form['param3'])
    param4 = float(request.form['param4'])
    param5 = float(request.form['param5'])
    param6 = float(request.form['param6'])
    param7 = float(request.form['param7'])
    param8 = float(request.form['param8'])
    param9 = float(request.form['param9'])
    param10 = float(request.form['param10'])
    param11 = float(request.form['param11'])
    

    # Make a prediction using the loaded model
    prediction = model.predict([[param1, param2, param3, param4, param5, param6, param7,param8,param9,param10,param11]])

    # Return the prediction as JSON
    # return jsonify({'prediction': prediction.tolist()[0]})
    return render_template('result_page.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True,port=8001)