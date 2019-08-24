from flask import Flask, request, jsonify, render_template
from sklearn.externals import joblib

import json

app = Flask(__name__)


@app.route('/run', methods=['POST'])
def run():
    classifier = joblib.load('sklearn.joblib')

    json_params = json.dumps(request.json, ensure_ascii=False)
    json = json.loads(json_params)

    #

    return classifier.predict(json)

@app.route('/', methods=['GET'])
def index():
    return '-> classify using method <b>POST   /run</b> route'

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)
