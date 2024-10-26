import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn import metrics
import json
from decision_tree import RandomForest
from sklearn.preprocessing import StandardScaler
import socket
import os

app = Flask(__name__, template_folder='templates')

global content

# Define base directory for better path management
base_dir = os.path.dirname(os.path.abspath(__file__))


def score(data: list):
    print("SCORE!")
    input_arr = data
    print(f"Json: {input_arr}")

    # Load the model
    model_path = os.path.join(base_dir, "data", "rand_forest_model.pkl")
    model = RandomForest.load('', model_path)
    print(f"Pickle: {model}")

    # Standardize the input data
    X = StandardScaler().fit_transform([list(i[:-1]) for i in input_arr])
    # Convert target to int for accuracy calculation
    y = [int(i[-1]) for i in input_arr]
    print(f"X: {X}")
    print(f"y: {y}")

    # Calculate prediction
    predicted_np = model.predict(X)
    print(f"predicted: {predicted_np}")

    # Calculate accuracy
    accuracy = metrics.accuracy_score(y, predicted_np)
    print(f"accuracy: {accuracy}")

    # Save results
    result_path = os.path.join(base_dir, 'result', 'accuracy.json')
    with open(result_path, 'w') as outfile:
        json.dump(
            {"accuracy": accuracy, "prediction": predicted_np.tolist()}, outfile)

    return predicted_np


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    global content
    if request.method == 'POST':
        try:
            content = request.get_json()
            print(f"content: {content}")
            prediction = score(content["data"])
            return jsonify({"prediction": prediction.tolist()})
        except Exception as e:
            print(f"Error: {str(e)}")
            return jsonify({"message": str(e)}), 400
    return jsonify(content)


@app.route('/result', methods=['GET'])
def result():
    result_path = os.path.join(base_dir, 'result', 'accuracy.json')
    with open(result_path) as json_file:
        data = json.load(json_file)

    html_string = '''
    <html>
      <head><title>Results</title></head>
      <link rel="stylesheet" type="text/css" href="df_style.css"/>
      <body style="background: url('static/images/grapes_and_wines.jpg'); height: 300px; background-size: cover;">
        <h3><a href="/main">Return to main page</a></h3><br>
        <h1 style="color: blue">Results</h1><br/>
        <h2>Accuracy: {accuracy}<br>Prediction:</h2> {prediction}
      </body>
    </html>
    '''

    with open(os.path.join(base_dir, 'templates', 'results.html'), 'w') as f:
        f.write(html_string.format(
            accuracy=data['accuracy'], prediction=data['prediction']))

    return render_template('results.html')


@app.route('/raw_data', methods=['GET'])
def raw_data():
    df = pd.read_csv(os.path.join(base_dir, "data", "winequalityN.csv"))
    table = df.head(20).to_html(classes="table table-striped")
    pd.set_option('colheader_justify', 'center')

    html_string = '''
    <html>
      <head><title>Raw Data</title></head>
      <link rel="stylesheet" type="text/css" href="df_style.css"/>
      <body style="background: url('static/images/grapes_and_wines.jpg'); height: 300px; background-size: cover;">
        <h3><a href="/main">Return to main page</a></h3><br>
        <h1 style="color: blue">Raw Data</h1><br/>
        {table}
      </body>
    </html>
    '''

    with open(os.path.join(base_dir, 'templates', 'data.html'), 'w') as f:
        f.write(html_string.format(table=table))

    return render_template("data.html")


@app.route('/input', methods=['GET'])
def input_data():
    df = pd.read_csv(os.path.join(base_dir, "data", "instance_raw.csv"))
    table = df.head(20).to_html(classes="table table-striped")
    html_string = '''
    <html>
      <head><title>Input Test Data</title></head>
      <link rel="stylesheet" type="text/css" href="df_style.css"/>
      <body style="background: url('static/images/grapes_and_wines.jpg'); height: 300px; background-size: cover;">
        <h3><a href="/main">Return to main page</a></h3><br>
        <h1 style="color: blue">Input Test Data</h1><br/>
        {table}
      </body>
    </html>
    '''

    # OUTPUT AN HTML FILE
    with open(os.path.join(base_dir, 'templates', 'input.html'), 'w') as f:
        f.write(html_string.format(table=table))

    return render_template('input.html')


@app.route('/wrangler_input', methods=['GET'])
def input_wr_data():
    df = pd.read_csv(os.path.join(base_dir, "data", "instance_wrangler.csv"))
    table = df.head(20).to_html(classes="table table-striped")

    html_string = '''
    <html>
      <head><title>Input Test Data (Wrangler)</title></head>
      <link rel="stylesheet" type="text/css" href="df_style.css"/>
      <body style="background: url('static/images/grapes_and_wines.jpg'); height: 300px; background-size: cover;">
        <h3><a href="/main">Return to main page</a></h3><br>
        <h1 style="color: blue">Input Test Data (Wrangler)</h1><br/>
        {table}
      </body>
    </html>
    '''

    # OUTPUT AN HTML FILE
    with open(os.path.join(base_dir, 'templates', 'wrangler_input.html'), 'w') as f:
        f.write(html_string.format(table=table))

    return render_template('wrangler_input.html')


@app.route('/about', methods=['GET'])
def about():
    a_main = "ABOUT information"
    return render_template("about.html", a_main=a_main)


@app.route('/main', methods=['GET'])
def main():
    return render_template('main.html', ipaddress=local_ip)


@app.route('/', methods=['GET'])
def home():
    return render_template("home.html", ip_address=local_ip)


@app.route('/data_visualizing', methods=['GET'])
def reports():
    return render_template('reports.html')


if __name__ == '__main__':
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print(f"Ip address: {local_ip}")
    app.run(debug=True, host=local_ip, port=5000)
