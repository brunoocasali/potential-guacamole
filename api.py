from flask import Flask, request, jsonify, render_template
from sklearn.externals import joblib

import json

app = Flask(__name__)


@app.route('/run', methods=['POST'])
def run():
    classifier = joblib.load('sklearn.joblib')

    # TODO:
    # refactor the way to retrieve the data and predict it.
    req_data = request.get_json()

    print(req_data)

    return classifier.predict(req_data)

@app.route('/', methods=['GET'])
def index():
    return '-> classify using method <b>POST   /run</b> route'

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)
