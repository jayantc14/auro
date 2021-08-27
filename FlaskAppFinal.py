# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 20:53:50 2021

@author: Jayant
"""
import json
import numpy as np
from flask import Flask, jsonify, request
from AuroScorefinal import TechnicalAnalysis
app = Flask(__name__)


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
@app.route('/get_technical_data')  # http://127.0.0.1:8088/get_technical_data?ticker='MSFT US Equity'
def get_technical_data():
    
    params_dict = request.args 
    ticker = params_dict['ticker']
    t = str(ticker)
    tana = TechnicalAnalysis(t)
    result_dict = tana.get_auro_score()
    #print ("FlaskAppFinal.get_technical_data: data received from TechnicalAnalysis module is {} ".format(result_dict))
    result_dict_json = json.loads(json.dumps(result_dict, cls=NumpyEncoder)) # Fix the Error - Object of type int64 is not JSON serializable
    return {'result' : result_dict_json}

app.run(host='0.0.0.0', port=8088)




