from flask import Flask, render_template, request
import pandas as pd
import tensorflow as tf

app = Flask(__name__)

# Load the saved models
soc_model = tf.keras.models.load_model('tesla_SOC.h5')
soh_model = tf.keras.models.load_model('SOH.h5')
range_model = tf.keras.models.load_model('range.h5')

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the HTML form
    print(request.form)
    voltage = request.form['voltage']
    current = request.form['current']
    battery_temp = request.form['battery_temp']
    terminal_voltage = request.form['terminal_voltage']
    terminal_current = request.form['terminal_current']
    temperature = request.form['temperature']
    quantity = request.form['quantity']
    city = request.form['city']
    motor_way = request.form['motor_way']
    country_roads = request.form['country_roads']
    consumption = request.form['consumption']
    ac = request.form['ac']
    park_heating = request.form['park_heating']
    ecr_deviation = request.form['ecr_deviation']
    tire_type = request.form['tire_type']
    driving_style_fast = request.form['driving_style_fast']
    driving_style_moderate = request.form['driving_style_moderate']
    avg_speed = request.form['avg_speed']

    # Check which models to predict based on available input data
    models_to_predict = []
    if voltage and current and battery_temp:
        models_to_predict.append('SOC')

    if terminal_voltage and terminal_current and temperature:
        models_to_predict.append('SOH')

    if quantity and city and motor_way and country_roads and consumption and ac and park_heating and ecr_deviation and tire_type and driving_style_fast and driving_style_moderate and avg_speed:
        models_to_predict.append('Range')

    # Check that at least one model is selected for prediction
    if not models_to_predict:
        return render_template('index.html', error='Please provide input data for at least one model')

    # Convert the input data to a Pandas DataFrame
    input_data = {
        'voltage': [float(voltage)] if voltage else [None],
        'current': [float(current)] if current else [None],
        'battery_temp': [float(battery_temp)] if battery_temp else [None],
        'terminal_voltage': [float(terminal_voltage)] if terminal_voltage else [None],
        'terminal_current': [float(terminal_current)] if terminal_current else [None],
        'temperature': [float(temperature)] if temperature else [None],
        'quantity': [float(quantity)] if quantity else [None],
        'city': [float(city)] if city else [None],
        'motor_way': [float(motor_way)] if motor_way else [None],
        'country_roads': [float(country_roads)] if country_roads else [None],
        'consumption': [float(consumption)] if consumption else [None],
        'ac': [float(ac)] if ac else [None],
        'park_heating': [float(park_heating)] if park_heating else [None],
        'ecr_deviation': [float(ecr_deviation)] if ecr_deviation else [None],
        'tire_type': [float(tire_type)] if tire_type else [None],
        'driving_style_fast': [float(driving_style_fast)] if driving_style_fast else [None],
        'driving_style_moderate': [float(driving_style_moderate)] if driving_style_moderate else [None],
        'avg_speed': [float(avg_speed)] if avg_speed else [None]
    }
    input_df = pd.DataFrame.from_dict(input_data)

    # Make predictions using the selected models
    soc_prediction = None
    soh_prediction = None
    range_prediction = None
    if 'SOC' in models_to_predict:
        soc_prediction = 100 * soc_model.predict(input_df[['voltage', 'current', 'battery_temp']])[0][0]
        print(soc_prediction)
    if 'SOH' in models_to_predict:
        soh_prediction = 100 * soh_model.predict(input_df[['terminal_voltage', 'terminal_current', 'temperature']])[0][0]
        print(soh_prediction)
    if 'Range' in models_to_predict:
        range_prediction = 1*range_model.predict(input_df[['quantity', 'city', 'motor_way', 'country_roads', 'consumption', 'ac', 'park_heating', 'ecr_deviation', 'tire_type', 'driving_style_fast', 'driving_style_moderate', 'avg_speed']])[0][0]
        print(range_prediction)
    # Render the results template with the predictions
    return render_template('index.html', soc=soc_prediction, soh=soh_prediction, range=range_prediction)

if __name__ == '__main__':
    app.run(debug=True)



