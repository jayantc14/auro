# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 13:41:32 2021

@author: hnazk
"""
import pandas as pd 
import requests

DB_API_ACCESS_TOKEN = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6MTMwMSwiaWF0IjoxNjIwNjI0ODQwfQ.kOe7wSwZEHsefWiQFCGjL1sbVBZqpd_-G9nDcSrea_g'
DB_API_HEADERS = {'Authorization': 'access_token {}'.format(DB_API_ACCESS_TOKEN) }
DB_API_URI_OHLCV = "http://44.240.12.174/technical_analysis/get_pricing"
DB_OHLCV_COLUMN_MAPPING = {"close_price":'Close',"open_price":'Open',"high_price":'High', "low_price":'Low',"volume":'Volume'}

class DataReader():


    def __init__(self, ticker):
        self.ticker = ticker
        self.data_from_sql = True
    
    def get_ohlcv(self, ticker = None): 
        ticker = ticker if ticker is not None else self.ticker
        if self.data_from_sql:
            self.data_raw = self.get_ohlcv_from_database(ticker)
        else:
            self.data_raw = self.get_ohlcv_from_file(ticker)                
        return self.data_raw

    def get_ohlcv_from_file(self, ticker):
        self.dfc = pd.read_excel("ClosePrice.xlsx").set_index('Date')
        self.dfh = pd.read_excel("HighPrice.xlsx").set_index('Date')
        self.dfl = pd.read_excel("LowPrice.xlsx").set_index('Date')
        self.dfv = pd.read_excel("Volume.xlsx").set_index('Date')
            
        self.dfc_ticker = self.dfc[ticker].rename("Close")      
        self.dfh_ticker = self.dfh[ticker].rename("High")    
        self.dfl_ticker = self.dfl[ticker].rename("Low")
        self.dfv_ticker = self.dfv[ticker].rename("Volume")
                    
        self.data_raw = pd.concat([self.dfc_ticker,self.dfh_ticker,self.dfl_ticker,self.dfv_ticker], axis = 1)                
        return self.data_raw

    def get_ohlcv_from_database(self, ticker):
        payload = { "ticker": self.ticker}            
        self.ohlcv_json = requests.post(DB_API_URI_OHLCV, json = payload, headers=DB_API_HEADERS)
        self.ohlcv_db = self.ohlcv_json.json()["message"]
        self.ohlcv_db_df = pd.DataFrame(self.ohlcv_db)        
        self.ohlcv_df = self.ohlcv_db_df.set_index("date").drop(columns="ticker").rename(columns=DB_OHLCV_COLUMN_MAPPING)
        
        return self.ohlcv_df


if __name__ == '__main__':
    ticker = '1088 HK Equity'
    dr = DataReader(ticker)
    data = dr.get_ohlcv()
    print (data)
    
    
    