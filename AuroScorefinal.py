# -*- coding: utf-8 -*-
#!/usr/bin/env python
# coding: utf-8

# Data Manipulation
import numpy as np
import pandas as pd
import talib as ta
import os, datetime as dt

# Plotting graphs
#import matplotlib.pyplot as plt

# Machine learning
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import confusion_matrix, classification_report

import pandas_ta as tal
import warnings
from data_reader import DataReader

warnings.filterwarnings("ignore")

# Data fetching
class TechnicalAnalysis():
        
    def __init__(self, ticker):
        self.ticker = ticker
        self.dreader = DataReader(ticker)
        self.data_raw = self.get_data(ticker)
    
    def get_data(self, ticker):
        self.data_ohlcv = self.dreader.get_ohlcv()        
        self.data_raw = self.data_ohlcv.dropna()#.interpolate()

        return self.data_raw
    
    def compute_technical_indicators(self, data):    
        df_ta = data.copy()
        df_ta['CCI'] = ta.CCI(df_ta['High'],df_ta['Low'], df_ta['Close'],timeperiod = 14)
        df_ta['MA'] = ta.SMA(df_ta['Close'],20)
        df_ta['EMA'] = ta.EMA(df_ta['Close'], timeperiod = 20)
        df_ta['EMA5'] = ta.EMA(df_ta['Close'], timeperiod = 5)
        df_ta['EMA10'] = ta.EMA(df_ta['Close'], timeperiod = 10)
    
        df_ta['up_band_1dev'],df_ta['mid_band'],df_ta['low_band_1dev'] = ta.BBANDS(df_ta['Close'], timeperiod =20, nbdevup=1, nbdevdn=1)
        df_ta['up_band_2dev'],df_ta['mid_band'],df_ta['low_band_2dev'] = ta.BBANDS(df_ta['Close'], timeperiod =20, nbdevup=2, nbdevdn=2)
        df_ta['up_band_3dev'],df_ta['mid_band'],df_ta['low_band_3dev'] = ta.BBANDS(df_ta['Close'], timeperiod =20, nbdevup=3, nbdevdn=3)

        df_ta['rsi'] = ta.RSI(df_ta['Close'],14)
        df_ta['BIAS'] = tal.bias(df_ta['Close'])
        df_ta['PSY'] = tal.psl(df_ta['Close'])
        df_ta['CMO'] = ta.CMO(df_ta['Close'], timeperiod=20)
        df_ta['ROC'] = ta.ROC(df_ta['Close'], timeperiod=20)
        df_ta['PPO'] = ta.PPO(df_ta['Close'], fastperiod=12, slowperiod=26, matype=0)
        df_ta['APO'] = ta.APO(df_ta['Close'], fastperiod=12, slowperiod=26, matype=0)
        df_ta['WMSR'] = ta.WILLR(df_ta['High'],df_ta['Low'], df_ta['Close'], timeperiod=14)
        df_ta['macd'], df_ta['macdsignal'], macdhist = ta.MACD(df_ta['Close'])
        df_ta['AR'] = ta.AROONOSC(df_ta['High'],df_ta['Low'], timeperiod=14)
        df_ta["VR"] = tal.pvr(df_ta['Close'],df_ta['Volume'])
        kc = tal.kc(df_ta['High'],df_ta['Low'], df_ta['Close'])
        df_ta = df_ta.join(kc)
        k = tal.kdj(df_ta['High'],df_ta['Low'], df_ta['Close'])
        df_ta = df_ta.join(k)
        df_ta['SAR'] = ta.SAR(df_ta['High'],df_ta['Low'], acceleration=0, maximum=0)
        
        df_ta.dropna(inplace = True)    
        #df_ta.reset_index(inplace= True,drop = True)
        return df_ta
    
    def encode_technical_indicators(self, data):
        df_enc = data.copy()
        df_enc[['RSI_enc', 'KDJ_enc']] = np.nan
        
        df_enc['RSI_enc'] = pd.cut(df_enc['rsi'], bins=[0, 15, 30, 70, 85, 100], labels=[1,2,3,4,5])
        df_enc['WMSR_enc'] = pd.cut(df_enc['WMSR'], bins=[-100, -90, -80, -20, -10, 0], labels=[1,2,3,4,5])
        df_enc['CCI_enc'] = pd.cut(df_enc['CCI'], bins=[-500, -200, -100, 0, 100, 200, 500], labels=[1,2,3,4,5,6])
        df_enc['ROC_enc'] = pd.cut(df_enc['ROC'], bins=[-500, -50, -20, -10, -5, 0, 5, 10, 20, 50, 500], labels=[1,2,3,4,5,6,7,8,9,10])
        df_enc['AR_enc'] = pd.cut(df_enc['AR'], bins=[0, 5, 10, 50, 90, 95, 100], labels=[1,2,3,4,5,6])
        df_enc['PSY_enc'] = pd.cut(df_enc['PSY'], bins=[0, 10, 25, 50, 75, 100], labels=[1,2,3,4,5])
        df_enc['CMO_enc'] = pd.cut(df_enc['CMO'], bins=[-100, -75, -50, 0, 50, 75, 100], labels=[1,2,3,4,5,6])
        
        df_enc['KDJ_enc'] = pd.cut(df_enc['K_9_3'], bins=[0, 20, 80, 100], labels=[1,2,3])
        
        cond_list = [
                     df_enc['Close'] < df_enc['low_band_3dev'],
                     (df_enc['low_band_3dev'] <= df_enc['Close'] ) & ( df_enc['Close'] <= df_enc['low_band_2dev'] ),  
                     (df_enc['low_band_2dev'] <= df_enc['Close'] ) & ( df_enc['Close'] <= df_enc['low_band_1dev'] ),  
                     (df_enc['low_band_1dev'] <= df_enc['Close'] ) & ( df_enc['Close'] <= df_enc['mid_band'] ),  
                     (df_enc['mid_band'] <= df_enc['Close'] ) & ( df_enc['Close'] <= df_enc['up_band_1dev'] ),  
                     (df_enc['up_band_1dev'] <= df_enc['Close'] ) & ( df_enc['Close'] <= df_enc['up_band_2dev'] ),  
                     (df_enc['up_band_2dev'] <= df_enc['Close'] ) & ( df_enc['Close'] <= df_enc['up_band_3dev'] ),                          
                     df_enc['up_band_3dev'] < df_enc['Close'] , 
                     ]
        df_enc['BOLL_enc'] = np.select(cond_list, [0,1,2,3,4,5,6,7], default=np.nan)                

        cond_list = [df_enc['KCUe_20_2'] < df_enc['Close'] , (df_enc['KCLe_20_2'] <= df_enc['Close'] ) & ( df_enc['Close'] <= df_enc['KCUe_20_2'] ),  df_enc['Close'] < df_enc['KCLe_20_2']]
        df_enc['KC_enc'] = np.select(cond_list, [3,2,1], default=np.nan)

        cond_list = [
                     ( df_enc['macd'].shift(1) < 0 ) & ( df_enc['macd'] > 0 ) , 
                     ( df_enc['macd'].shift(1) > 0 ) & ( df_enc['macd'] < 0 ) , 
                     ( df_enc['macdsignal'].shift(1) < df_enc['macd'].shift(1) ) & ( df_enc['macdsignal'] > df_enc['macd'] ) , 
                     ( df_enc['macdsignal'].shift(1) > df_enc['macd'].shift(1) ) & ( df_enc['macdsignal'] < df_enc['macd'] ) ,                      
                     ]
        df_enc['MACD_enc'] = np.select(cond_list, [1,2,3,4], default=0)

        cond_list = [
                     ( df_enc['PPO'].shift(1) < 0 ) & ( df_enc['PPO'] < 0 ) , 
                     ( df_enc['PPO'].shift(1) < 0 ) & ( df_enc['PPO'] > 0 ) , 
                     ( df_enc['PPO'].shift(1) > 0 ) & ( df_enc['PPO'] > 0 ) , 
                     ( df_enc['PPO'].shift(1) > 0 ) & ( df_enc['PPO'] < 0 ) , 
                     ]
        df_enc['PPO_enc'] = np.select(cond_list, [1,2,3,4], default=0)

        cond_list = [
                     ( df_enc['APO'].shift(1) < 0 ) & ( df_enc['APO'] < 0 ) , 
                     ( df_enc['APO'].shift(1) < 0 ) & ( df_enc['APO'] > 0 ) , 
                     ( df_enc['APO'].shift(1) > 0 ) & ( df_enc['APO'] > 0 ) , 
                     ( df_enc['APO'].shift(1) > 0 ) & ( df_enc['APO'] < 0 ) , 
                     ]
        df_enc['APO_enc'] = np.select(cond_list, [1,2,3,4], default=0)


        cond_list = [
                     ( df_enc['MA'].shift(1) < df_enc['Close'].shift(1) ) & ( df_enc['MA'] < df_enc['Close'] ) , 
                     ( df_enc['MA'].shift(1) < df_enc['Close'].shift(1) ) & ( df_enc['MA'] > df_enc['Close'] ) , 
                     ( df_enc['MA'].shift(1) > df_enc['Close'].shift(1) ) & ( df_enc['MA'] > df_enc['Close'] ) , 
                     ( df_enc['MA'].shift(1) > df_enc['Close'].shift(1) ) & ( df_enc['MA'] < df_enc['Close'] ) , 
                     ]
        df_enc['MA_enc'] = np.select(cond_list, [1,2,3,4], default=np.nan)

        cond_list = [
                     ( df_enc['EMA5'].shift(1) < df_enc['EMA10'].shift(1) ) & ( df_enc['EMA5'] < df_enc['EMA10'] ) , 
                     ( df_enc['EMA5'].shift(1) < df_enc['EMA10'].shift(1) ) & ( df_enc['EMA5'] > df_enc['EMA10'] ) , 
                     ( df_enc['EMA5'].shift(1) > df_enc['EMA10'].shift(1) ) & ( df_enc['EMA5'] > df_enc['EMA10'] ) , 
                     ( df_enc['EMA5'].shift(1) > df_enc['EMA10'].shift(1) ) & ( df_enc['EMA5'] < df_enc['EMA10'] ) , 
                     ]
        df_enc['EMA_enc'] = np.select(cond_list, [1,2,3,4], default=np.nan)


        cond_list = [df_enc['Close'] >= df_enc['SAR'] , df_enc['Close'] < df_enc['SAR']]
        df_enc['SAR_enc'] = np.select(cond_list, [1,0], default=np.nan)
        
        return df_enc


    def preprocess_data(self, data):
        df_proc = data.copy()
        df_proc['Close_next1day'] = df_proc['Close'].shift(-1)
        df_proc['ret'] = df_proc['Close'].pct_change(periods=1)
        df_proc['ret_next1day'] = df_proc['ret'].shift(-1)
        df_proc['ret_next1day_enc'] = np.select([ df_proc['ret_next1day'] >= 0, df_proc['ret_next1day'] < 0 ], 
                                               [1, 0], default=np.nan)
        return df_proc
        
        
    def get_feature_response_variables_for_model(self, data_input):
        data = data_input.dropna()
        y = data['ret_next1day_enc']
        feature_list = ['MA', 'EMA', 'EMA5', 'EMA10', 'CMO', 'PPO', 'APO', 'WMSR', 'VR', 'SAR', 'RSI_enc', 'KC_enc', 'MACD_enc']
        X = data[feature_list]
        return X, y
    
    def fit_train_model(self, X, y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = 0.3, random_state = 4)
        
        # scaling the features
        scaler = RobustScaler()
        self.X_train_scaled = scaler.fit_transform(self.X_train)
        self.X_test_scaled = scaler.transform(self.X_test)
        
        self.model = LogisticRegression()
        self.model.fit(self.X_train_scaled, self.y_train)
            
        res = X.iloc[len(X)-1,:]
        res = res.to_frame()
        res = res.transpose()  
        result = self.model.predict_proba(res)
        
        result  = pd.DataFrame(result)
        result = result.loc[0,1] 
        #label = 1 rise in price #1 is column name for prediction proba. of label 1 
        
        return result

    def model_summary(self, y, y_pred):
        model = self.model
        cm_df = pd.DataFrame(confusion_matrix(y, y_pred).T, index=model.classes_, columns=model.classes_)
        cm_df.index.name = 'Predicted'
        cm_df.columns.name = 'True'
        print(cm_df)
        print(classification_report(y, y_pred))
    
    def get_individual_indicator_prediction(self, data, resultdic):
        data1 = data.copy()
        df = data1.iloc[-1,:]


        resultdic['SAR'] = {}               
        cond_list = [df['Close'] >= df['SAR'] , df['Close'] < df['SAR']]
        df['SAR_pred'] = np.select(cond_list, ['upwardtrend','downwardtrend'], default='')
        resultdic['SAR']['prediction'] = df['SAR_pred']

        resultdic['KC'] = {}
        cond_list = [df['KCUe_20_2'] < df['Close'] , (df['KCLe_20_2'] <= df['Close'] ) & ( df['Close'] <= df['KCUe_20_2'] ),  df['Close'] < df['KCLe_20_2']]
        df['KC_pred'] = np.select(cond_list, ['Buysignal','neutral', 'sellsignal'], default='')
        resultdic['KC']['prediction'] = df['KC_pred']
        
        resultdic['KDJ'] = {}
        cond_list = [ 80 < df['K_9_3'], 20 > df['D_9_3']]
        df['KDJ_pred'] = np.select(cond_list, ['overbought','oversold'], default='neutral')
        resultdic['KDJ']['prediction'] = df['KDJ_pred']


        resultdic['VR'] = {}
        cond_list = [ df['VR'] >= 2.5 , df['VR'] < 2.5 ]
        df['VR_pred'] = np.select(cond_list, ['sellsignal','Buysignal'], default='')
        resultdic['VR']['prediction'] = df['VR_pred']
        
                
        resultdic['CCI'] = {}
        cond_list = [100 < df['CCI'] , ( -100 <= df['CCI'] ) & ( df['CCI'] <= 100 ),  df['CCI'] < -100 ]
        df['CCI_pred'] = np.select(cond_list, ['overbought','neutral', 'oversold'], default='')
        resultdic['CCI']['prediction'] = df['CCI_pred']

        
        resultdic['PSY'] = {}
        cond_list = [ 70 < df['PSY'] , ( 30 <= df['PSY'] ) & ( df['PSY'] <= 70 ),  df['PSY'] < 30 ]
        df['PSY_pred'] = np.select(cond_list, ['overbought','neutral', 'oversold'], default='')
        resultdic['PSY']['prediction'] = df['PSY_pred']


        resultdic['WMSR'] = {}
        cond_list = [ -20 < df['WMSR'] , ( -80 <= df['WMSR'] ) & ( df['WMSR'] <= -20 ),  df['WMSR'] < -80 ]
        df['WMSR_pred'] = np.select(cond_list, ['overbought','neutral', 'oversold'], default='')
        resultdic['WMSR']['prediction'] = df['WMSR_pred']

        # TODOs: Correct the logic of Moving average crossover
        resultdic['MA'] = {}
        cond_list = [ df['Close'] >= df['MA'], df['Close'] < df['MA'] ]
        df['MA_pred'] = np.select(cond_list, ['Buysignal','sellsignal'], default='')
        resultdic['MA']['prediction'] = df['MA_pred']

        resultdic['BOLL'] = {}
        cond_list = [ df['up_band_2dev'] < df['Close'] , ( df['low_band_2dev'] <= df['Close'] ) & ( df['Close'] <= df['up_band_2dev'] ),  df['Close'] < df['low_band_2dev'] ]
        df['BOLL_pred'] = np.select(cond_list, ['overbought','neutral', 'oversold'], default='')
        resultdic['BOLL']['prediction'] = df['BOLL_pred']


        resultdic['RSI'] = {}
        cond_list = [ 70 < df['rsi'] , ( 30 <= df['rsi'] ) & ( df['rsi'] <= 70 ),  df['rsi'] < 30 ]
        df['RSI_pred'] = np.select(cond_list, ['overbought','neutral', 'oversold'], default='')
        resultdic['RSI']['prediction'] = df['RSI_pred']
        
        
        resultdic['PPO'] = {}
        cond_list = [ 10 < df['PPO'] , ( -10 <= df['PPO'] ) & ( df['PPO'] <= 10 ),  df['PPO'] < -10 ]
        df['PPO_pred'] = np.select(cond_list, ['overbought','neutral', 'oversold'], default='')
        resultdic['PPO']['prediction'] = df['PPO_pred']

        
        resultdic['APO'] = {}
        cond_list = [ df['APO'] >= 0, df['APO'] < 0 ]
        df['APO_pred'] = np.select(cond_list, ['upwardtrend','downwardtrend'], default='')
        resultdic['APO']['prediction'] = df['APO_pred']

        '''
        resultdic['AR'] = {}
        cond_list = [ df['AR'] >= 0, df['AR'] < 0 ]
        df['AR_pred'] = np.select(cond_list, ['upwardtrend','downwardtrend'], default='')
        resultdic['AR']['prediction'] = df['AR_pred']
        '''
        
        resultdic['ROC'] = {}
        cond_list = [ df['ROC'] >= 0, df['ROC'] < 0 ]
        df['ROC_pred'] = np.select(cond_list, ['upwardtrend','downwardtrend'], default='')
        resultdic['ROC']['prediction'] = df['ROC_pred']

        
        resultdic['CMO'] = {}
        cond_list = [ 50 < df['CMO'] , ( -50 <= df['CMO'] ) & ( df['CMO'] <= 50 ),  df['CMO'] < -50 ]
        df['CMO_pred'] = np.select(cond_list, ['upwardtrend','neutral', 'downwardtrend'], default='')
        resultdic['CMO']['prediction'] = df['CMO_pred']
      
        # TODOs: Correct the logic of MACD       
        resultdic['MACD'] = {}
        cond_list = [ df['macdsignal']  < df['macd'] ,  df['macd'] < df['macdsignal'] ]
        df['MACD_pred'] = np.select(cond_list, ['overbought', 'oversold'], default='')
        resultdic['MACD']['prediction'] = df['MACD_pred']
                
        # TODOs: Correct the logic of EMA         
        resultdic['EMA'] = {}
        cond_list = [ df['EMA10']  < df['EMA5'] ,  df['EMA5'] < df['EMA10'] ]
        df['EMA_pred'] = np.select(cond_list, ['overbought', 'oversold'], default='')
        resultdic['EMA']['prediction'] = df['EMA_pred']

        return resultdic
       
    
    def get_backtesting_result(self, data, resultdic):
        indicators ={  
                    "MA" : 20,
                    "EMA" : 20,
                    "BOLL" :20, #Done
                    "RSI":14,  # Done
                    "MACD":9, 
                    "CCI":14,  # Done   
                    #BIAS
                    #PER
                    #MAVOL
                    "PSY": 0,
                    "WMSR": 20,  # Done
                    "CMO": 20,
                    "ROC":20,    # Done
                    "PPO":26,
                    "APO":26,
                    #"AR" : 14,
                    "KDJ" :0,  # Done
                    "KC":0,    # Done
                    "VR":0,
                    "SAR":0    # Done    
                    }
    
        df = data
         
        df['avg'] = ((df['Close'].shift(-1)-df['Close'])/df['Close'])*100
        max_increase = df['avg'].max()
        
        max_decline = df['avg'].min()
        avg_change = df['avg'].mean()
        
        for indicator, time in indicators.items():
            length = len(df)
            count = 0
            count_rise = 0
            count_sell = 0
            
            if(indicator == 'SAR'):                        
                current_state =  df['SAR_enc'].iloc[-1]
                df_current_state = df[df['SAR_enc'] == current_state]
                nextperiod_return_s = df_current_state['ret_next1day']
    
            elif(indicator == 'KC' ):
                current_state =  df['KC_enc'].iloc[-1]
                df_current_state = df[df['KC_enc'] == current_state]
                nextperiod_return_s = df_current_state['ret_next1day']
 
            elif(indicator == 'KDJ'):
                current_state =  df['KDJ_enc'].iloc[-1]
                df_current_state = df[df['KDJ_enc'] == current_state]
                nextperiod_return_s = df_current_state['ret_next1day']
 
            elif(indicator == 'VR'):
                current_state =  df['VR'].iloc[-1]
                df_current_state = df[df['VR'] == current_state]
                nextperiod_return_s = df_current_state['ret_next1day']
 
            elif(indicator == 'CCI'):
                current_state =  df['CCI_enc'].iloc[-1]
                df_current_state = df[df['CCI_enc'] == current_state]
                nextperiod_return_s = df_current_state['ret_next1day']
 
            elif(indicator == 'PSY'):
                current_state =  df['PSY_enc'].iloc[-1]
                df_current_state = df[df['PSY_enc'] == current_state]
                nextperiod_return_s = df_current_state['ret_next1day']
 
            elif(indicator == 'WMSR'):
                current_state =  df['WMSR_enc'].iloc[-1]
                df_current_state = df[df['WMSR_enc'] == current_state]
                nextperiod_return_s = df_current_state['ret_next1day']
                     
            elif(indicator == "RSI"):
                current_state =  df['RSI_enc'].iloc[-1]
                df_current_state = df[df['RSI_enc'] == current_state]
                nextperiod_return_s = df_current_state['ret_next1day']
                 
            elif(indicator == "PPO"):
                current_state =  df['PPO_enc'].iloc[-1]
                df_current_state = df[df['PPO_enc'] == current_state]
                nextperiod_return_s = df_current_state['ret_next1day']
                     
            elif(indicator == "APO"):
                current_state =  df['APO_enc'].iloc[-1]
                df_current_state = df[df['APO_enc'] == current_state]
                nextperiod_return_s = df_current_state['ret_next1day']
            
            elif(indicator == "AR"):
                current_state =  df['AR_enc'].iloc[-1]
                df_current_state = df[df['AR_enc'] == current_state]
                nextperiod_return_s = df_current_state['ret_next1day']
         
            elif(indicator == "ROC"):
                current_state =  df['ROC_enc'].iloc[-1]
                df_current_state = df[df['ROC_enc'] == current_state]
                nextperiod_return_s = df_current_state['ret_next1day']
                     
            elif(indicator == "CMO"):
                current_state =  df['CMO_enc'].iloc[-1]
                df_current_state = df[df['CMO_enc'] == current_state]
                nextperiod_return_s = df_current_state['ret_next1day']
 
            elif(indicator == "MA"):
                current_state =  df['MA_enc'].iloc[-1]
                df_current_state = df[df['MA_enc'] == current_state]
                nextperiod_return_s = df_current_state['ret_next1day']
 
            elif(indicator == "BOLL"):
                current_state =  df['BOLL_enc'].iloc[-1]
                df_current_state = df[df['BOLL_enc'] == current_state]
                nextperiod_return_s = df_current_state['ret_next1day']
                     
            elif(indicator == "MACD"):
                current_state =  df['MACD_enc'].iloc[-1]
                df_current_state = df[df['MACD_enc'] == current_state]
                nextperiod_return_s = df_current_state['ret_next1day']
                         
            elif(indicator == "EMA"):
                current_state =  df['EMA_enc'].iloc[-1]
                df_current_state = df[df['EMA_enc'] == current_state]
                nextperiod_return_s = df_current_state['ret_next1day']
 

            count = df_current_state['ret_next1day'].count()
            count_rise = ( df_current_state['ret_next1day'] > 0).sum()
            count_sell = ( df_current_state['ret_next1day'] <= 0).sum()
            fall_rate = count_sell/count                 
            avg_change = nextperiod_return_s.mean()
            max_increase = nextperiod_return_s.max()
            max_decline = nextperiod_return_s.min()


            resultdic[indicator]['Count'] = count
            resultdic[indicator]['NextDayRise'] = count_rise
            resultdic[indicator]['NextDayFall'] = count_sell
            resultdic[indicator]['FallRate'] = float (count_sell/count) if count != 0 else 0                 
            resultdic[indicator]['AVG_Change']= avg_change
            resultdic[indicator]['MAX_Increase']= max_increase
            resultdic[indicator]['MAX_Decline']= max_decline
    
        return resultdic  
    
    def get_auro_score(self):
        #data_raw = get_data(s)
        
        self.data_ta = self.compute_technical_indicators(self.data_raw)
        self.data_ta_encoded = self.encode_technical_indicators(self.data_ta)
        self.data_processed = self.preprocess_data(self.data_ta_encoded)
        self.X, self.y  = self.get_feature_response_variables_for_model(self.data_processed)
        self.result = self.fit_train_model(self.X, self.y)
    
        self.resultdic = {}
        self.resultdic['Tickername'] = self.ticker
        self.resultdic['Auroscore'] = self.result
    
        self.resultdic = self.get_individual_indicator_prediction(self.data_processed, self.resultdic)
        self.resultdic = self.get_backtesting_result(self.data_processed, self.resultdic)      
    
        return self.resultdic 

if __name__ == "__main__":
    from pprint import pprint
    
    s = 'INDIGO IN Equity' #'PAG LN Equity'
    tana = TechnicalAnalysis(s)
    result_dict = tana.get_auro_score()
    pprint(result_dict)

    output_fpath = os.path.join( os.getcwd(), 'output', "result_ta_"+str(dt.date.today()) )
    with open(output_fpath, 'w') as f:
        pprint(result_dict, stream=f) #print(result_dict, file=f)
            
    #prediction code on the test dataset
    y_pred = tana.model.predict(tana.X_test_scaled)
    tana.model_summary(tana.y_test, y_pred)
