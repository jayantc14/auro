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
warnings.filterwarnings("ignore")

# Data fetching
class TechnicalAnalysis():
    
    
    def __init__(self, s):
        self.s = s
        self.data_raw = self.get_data(s)
    
    def get_data(self, s):
        self.dfc = pd.read_excel("ClosePrice.xlsx").set_index('Date')
        self.dfh = pd.read_excel("HighPrice.xlsx").set_index('Date')
        self.dfl = pd.read_excel("LowPrice.xlsx").set_index('Date')
        self.dfv = pd.read_excel("Volume.xlsx").set_index('Date')
            
        self.dfc_ticker = self.dfc[s].rename("Close")      
        self.dfh_ticker = self.dfh[s].rename("High")    
        self.dfl_ticker = self.dfl[s].rename("Low")
        self.dfv_ticker = self.dfv[s].rename("Volume")
                    
        self.data_raw = pd.concat([self.dfc_ticker,self.dfh_ticker,self.dfl_ticker,self.dfv_ticker], axis = 1)
        
        self.data_raw = self.data_raw.dropna()#.interpolate()

        return self.data_raw
    
    def compute_technical_indicators(self, data):    
        df_ta = data.copy()
        df_ta['CCI'] = ta.CCI(df_ta['High'],df_ta['Low'], df_ta['Close'],timeperiod = 14)
        df_ta['MA'] = ta.SMA(df_ta['Close'],20)
        df_ta['EMA'] = ta.EMA(df_ta['Close'], timeperiod = 20)
        df_ta['EMA5'] = ta.EMA(df_ta['Close'], timeperiod = 5)
        df_ta['EMA10'] = ta.EMA(df_ta['Close'], timeperiod = 10)
    
        df_ta['up_band'],df_ta['mid_band'],df_ta['low_band'] = ta.BBANDS(df_ta['Close'], timeperiod =20)
        df_ta['rsi'] = ta.RSI(df_ta['Close'],14)
        df_ta['BIAS'] = tal.bias(df_ta['Close'])
        df_ta['PSY'] = tal.psl(df_ta['Close'])
        df_ta['CMO'] = ta.CMO(df_ta['Close'], timeperiod=20)
        df_ta['ROC'] = ta.ROC(df_ta['Close'], timeperiod=20)
        df_ta['PPO'] = ta.PPO(df_ta['Close'], fastperiod=12, slowperiod=26, matype=0)
        df_ta['APO'] = ta.APO(df_ta['Close'], fastperiod=12, slowperiod=26, matype=0)
        df_ta['WMSR'] = ta.WILLR(df_ta['High'],df_ta['Low'], df_ta['Close'], timeperiod=14)
        macd, macdsignal, macdhist = ta.MACD(df_ta['Close'])
        df_ta['macd'] = macd
        df_ta['macdsignal'] = macdsignal
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

        df_enc['KDJ_enc'] = pd.cut(df_enc['K_9_3'], bins=[0, 20, 80, 100], labels=[1,2,3])
        
        cond_list = [df_enc['up_band'] < df_enc['Close'] , (df_enc['low_band'] <= df_enc['Close'] ) & ( df_enc['Close'] <= df_enc['up_band'] ),  df_enc['Close'] < df_enc['low_band']]
        df_enc['BOLL_enc'] = np.select(cond_list, [3,2,1], default=np.nan)                

        cond_list = [df_enc['KCUe_20_2'] < df_enc['Close'] , (df_enc['KCLe_20_2'] <= df_enc['Close'] ) & ( df_enc['Close'] <= df_enc['KCUe_20_2'] ),  df_enc['Close'] < df_enc['KCLe_20_2']]
        df_enc['KC_enc'] = np.select(cond_list, [3,2,1], default=np.nan)

        cond_list = [df_enc['macdsignal'] < df_enc['macd'] , df_enc['macd'] < df_enc['macdsignal']]
        df_enc['MACD_enc'] = np.select(cond_list, [3,1], default=np.nan)
       
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
        data = data_input.copy()

        data.drop(['rsi'],axis=1,inplace = True)   
        data.drop(['macd','macdsignal'],axis=1,inplace = True) 
        data.drop(['up_band','low_band','mid_band'],axis=1,inplace = True) 
        data.drop(["KCUe_20_2","KCLe_20_2","KCBe_20_2"],axis = 1,inplace = True )
        data.drop(["K_9_3","D_9_3","J_9_3"],axis = 1,inplace = True )
    
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
        df['SAR_pred'] = np.select(cond_list, ['Buysignal','sellsignal'], default='')
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
        cond_list = [ df['up_band'] < df['Close'] , ( df['low_band'] <= df['Close'] ) & ( df['Close'] <= df['up_band'] ),  df['Close'] < df['low_band'] ]
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


        resultdic['AR'] = {}
        cond_list = [ df['AR'] >= 0, df['AR'] < 0 ]
        df['AR_pred'] = np.select(cond_list, ['upwardtrend','downwardtrend'], default='')
        resultdic['AR']['prediction'] = df['AR_pred']

        
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
                    "BOLL" :20,
                    "RSI":14,  # Done
                    "MACD":9, 
                    "CCI":14,  # Done   
                    #BIAS
                    #PER
                    #MAVOL
                    "PSY": 0,
                    "WMSR": 20,  # Done
                    "CMO": 20,
                    "ROC":20,
                    "PPO":26,
                    "APO":26,
                    "AR" : 14,
                    "KDJ" :0,
                    "KC":0, 
                    "VR":0,
                    "SAR":0        
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
                        
                for epoch in df.index: #range(length):
        
                    if df.loc[epoch,"Close"]>df.loc[epoch, indicator] :
        
                        count_rise = count_rise+1
                        count = count+1
                    elif df.loc[epoch,"Close"]<df.loc[epoch, indicator] :
        
                        count_sell = count_sell+1
                        count = count+1
            elif(indicator == 'KC' ):
                current_state =  df['KC_enc'].iloc[-1]
                df_current_state = df[df['KC_enc'] == current_state]
                nextperiod_return_s = df_current_state['ret_next1day']
 
                count = df_current_state['ret_next1day'].count()
                count_rise = ( df_current_state['ret_next1day'] > 0).sum()
                count_sell = ( df_current_state['ret_next1day'] <= 0).sum()
                fall_rate = count_sell/count                 
                avg_change = nextperiod_return_s.mean()
                max_increase = nextperiod_return_s.max()
                max_decline = nextperiod_return_s.min()

            elif(indicator == 'KDJ'):
                for epoch in df.index: #range(length ):   
                    if 90<df.loc[epoch ,'K_9_3'] :
                        count_rise = count_rise+1
                        count = count+1
        
                    elif 20 >df.loc[epoch, 'D_9_3'] :
                        count_sell = count_sell+1
                        count = count+1
            elif(indicator == 'VR'):
                for epoch in df.index: #range(length ):   
                    if 2.5<df.loc[epoch ,'VR'] :
                        count_rise = count_rise+1
                        count = count+1
        
                    elif 2.5 >df.loc[epoch, 'VR'] :
                        count_sell = count_sell+1
                        count = count+1

            elif(indicator == 'CCI'):
                current_state =  df['CCI_enc'].iloc[-1]
                df_current_state = df[df['CCI_enc'] == current_state]
                nextperiod_return_s = df_current_state['ret_next1day']
 
                count = df_current_state['ret_next1day'].count()
                count_rise = ( df_current_state['ret_next1day'] > 0).sum()
                count_sell = ( df_current_state['ret_next1day'] <= 0).sum()
                fall_rate = count_sell/count                 
                avg_change = nextperiod_return_s.mean()
                max_increase = nextperiod_return_s.max()
                max_decline = nextperiod_return_s.min()


            elif(indicator == 'PSY'):
                for epoch in df.index: #range(length ):   
                    if 70<df.loc[epoch ,'PSY'] :
                        count_rise = count_rise+1
                        count = count+1
        
                    elif 30 >df.loc[epoch, 'PSY'] :
                        count_sell = count_sell+1
                        count = count+1
            elif(indicator == 'WMSR'):
                current_state =  df['WMSR_enc'].iloc[-1]
                df_current_state = df[df['WMSR_enc'] == current_state]
                nextperiod_return_s = df_current_state['ret_next1day']
 
                count = df_current_state['ret_next1day'].count()
                count_rise = ( df_current_state['ret_next1day'] > 0).sum()
                count_sell = ( df_current_state['ret_next1day'] <= 0).sum()
                fall_rate = count_sell/count                 
                avg_change = nextperiod_return_s.mean()
                max_increase = nextperiod_return_s.max()
                max_decline = nextperiod_return_s.min()

            elif(indicator == 'MA'):
                for epoch in df.index: #range(length ):   
                    if -20<df.loc[epoch ,'WMSR'] :
                        count_rise = count_rise+1
                        count = count+1
        
                    elif -80>df.loc[epoch, 'WMSR'] :
                        count_sell = count_sell+1
                        count = count+1
                    
            elif(indicator == "RSI"):
                current_state =  df['RSI_enc'].iloc[-1]
                df_current_state = df[df['RSI_enc'] == current_state]
                nextperiod_return_s = df_current_state['ret_next1day']
 
                count = df_current_state['ret_next1day'].count()
                count_rise = ( df_current_state['ret_next1day'] > 0).sum()
                count_sell = ( df_current_state['ret_next1day'] <= 0).sum()
                fall_rate = count_sell/count                 
                avg_change = nextperiod_return_s.mean()
                max_increase = nextperiod_return_s.max()
                max_decline = nextperiod_return_s.min()
                                    
                
            elif(indicator == "PPO"):
                for epoch in df.index: #range(length ):   
                    if 10<df.loc[epoch ,"PPO"] :
                        count_rise = count_rise+1
                        count = count+1
        
                    elif -10>df.loc[epoch, "PPO"] :
                        count_sell = count_sell+1
                        count = count+1
                    
            elif(indicator == "APO"):
                for epoch in df.index: #range(length ):   
                    if 0<df.loc[epoch ,"APO"] :
                        count_rise = count_rise+1
                        count = count+1
        
                    elif 0>df.loc[epoch, "APO"] :
                        count_sell = count_sell+1
                        count = count+1
           
            elif(indicator == "AR"):
                for epoch in df.index: #range(length ):   
                    if 0 <df.loc[epoch ,"AR"] :
                        count_rise = count_rise+1
                        count = count+1
        
                    elif 0>df.loc[epoch, "AR"] :
                        count_sell = count_sell+1
                        count = count+1
        
            elif(indicator == "ROC"):
                for epoch in df.index: #range(length ):   
                    if 0<df.loc[epoch ,"ROC"] :
                        count_rise = count_rise+1
                        count = count+1
        
                    elif 0>df.loc[epoch, "ROC"] :
                        count_sell = count_sell+1
                        count = count+1
                    
            elif(indicator == "CMO"):
                for epoch in df.index: #range(length ):   
                    if 50<df.loc[epoch ,"CMO"] :
                        count_rise = count_rise+1
                        count = count+1
        
                    elif -50>df.loc[epoch, "CMO"] :
                        count_sell = count_sell+1
                        count = count+1
            elif(indicator == "MA"):
                for epoch in df.index: #range(length ):
        
                    if df.loc[epoch,"Close"]>df.loc[epoch ,"MA"] :
                        count_rise = count_rise+1
                        count = count+1
                    elif df.loc[epoch ,"MA"]<df.loc[epoch,"Close"] :
                        count_sell = count_sell+1
                        count = count+1
            elif(indicator == "BOLL"):
                for epoch in df.index: #range(length ):
        
                    if df.loc[epoch,"Close"]>df.loc[epoch ,"up_band"] :
                        count_rise = count_rise+1
                        count = count+1
                    elif df.loc[epoch ,"low_band"]<df.loc[epoch,"Close"] :
                        count_sell = count_sell+1
                        count = count+1
                    
            elif(indicator == "MACD"):
                for epoch in df.index: #range(length ):
        
                    if df.loc[epoch,"macd"]>df.loc[epoch ,"macdsignal"] :
                        count_rise = count_rise+1
                        count = count+1
                    elif df.loc[epoch ,"macd"]<df.loc[epoch,"macdsignal"] :
                        count_sell = count_sell+1
                        count = count+1
                        
            elif(indicator == "EMA"):
                for epoch in df.index: #range(length ):
                    
                        
                    if df.loc[epoch, "EMA5"]>df.loc[epoch,"EMA10"] :
                        count_rise = count_rise+1
                        count = count+1                    
                    
                    elif df.loc[epoch,"EMA10"]<df.loc[epoch, "EMA5"] :
                         count_sell = count_sell+1
                         count = count+1
                         
            resultdic[indicator]['Count'] = count
            resultdic[indicator]['NextDayRise'] = count_rise
            resultdic[indicator]['NextDayFall'] = count_sell
            if count != 0:
                
               resultdic[indicator]['FallRate'] = float (count_sell/count)
            else:
                 resultdic[indicator]['FallRate'] = float (0)
                 
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
        self.resultdic['Tickername'] = self.s
        self.resultdic['Auroscore'] = self.result
    
        self.resultdic = self.get_individual_indicator_prediction(self.data_processed, self.resultdic)
        self.resultdic = self.get_backtesting_result(self.data_processed, self.resultdic)      
    
        return self.resultdic 

if __name__ == "__main__":
    from pprint import pprint
    
    s = 'PAG LN Equity'
    tana = TechnicalAnalysis(s)
    result_dict = tana.get_auro_score()
    pprint(result_dict)

    output_fpath = os.path.join( os.getcwd(), 'output', "result_ta_"+str(dt.date.today()) )
    with open(output_fpath, 'w') as f:
        pprint(result_dict, stream=f) #print(result_dict, file=f)
            
    # prediction code on the test dataset
    #self.y_pred = self.model.predict(self.X_test_scaled)
    #self.model_summary(self.y_test, self.y_pred)
