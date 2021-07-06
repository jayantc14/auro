# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 20:53:50 2021

@author: Jayant
"""
from flask import Flask, jsonify, request
from AuroScorefinal2 import get_auro_score
app = Flask(__name__)


@app.route('/get_technical_data')  # http://127.0.0.1:8088/get_technical_data?ticker='MSFT US Equity'
def get_technical_data():
    
    params_dict = request.args 
    ticker = params_dict['ticker']
    #ticker = str(ticker)
    t = str(ticker)
    #result_dict = {'auro_score': 0.86,'indicator_value': {'CCI': {155: 'overbought'}, 'MA': {155: 'Buysignal'}, 'EMA': {155: 'overbought'}, 'rsi': {155: 'neutral'}, 'PSY': {155: 'neutral'}, 'CMO': {155: 'neutral'}, 'ROC': {155: 'downwardtrend'}, 'PPO': {155: 'neutral'}, 'APO': {155: 'downwardtrend'}, 'WMSR': {155: 'overbought'}, 'macd': {155: 'overbought'}, 'AR': {155: 'upwardtrend'}, 'VR': {155: 'Buysignal'}, 'SAR': {155: 'Buysignal'}, 'KC': {155: 'neutral'}, 'KDJ': {155: 'neutral'}, 'BOLL': {155: 'neutral'}}   }
    
    result_dict = get_auro_score(t)
    
    return {'result' : result_dict}

app.run(host='0.0.0.0', port=8088)