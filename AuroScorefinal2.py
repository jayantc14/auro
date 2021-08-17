# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 12:44:11 2021

@author: Jayant
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 10:41:51 2021

@author: Jayant
"""

#!/usr/bin/env python
# coding: utf-8

# In[49]:


# Data Manipulation
import numpy as np
import pandas as pd

# Technical Indicators
import talib as ta

# Plotting graphs
#import matplotlib.pyplot as plt

# Machine learning
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import pandas_ta as tal
import warnings
warnings.filterwarnings("ignore")

# Data fetching


# In[50]:

def get_auro_score(s):
    
   
    
    dfc = pd.read_excel("ClosePrice.xlsx")
    dfc.rename(columns = {'USD':'Date'}, inplace = True)
    dfh = pd.read_excel("HighPrice.xlsx")
    dfl = pd.read_excel("LowPrice.xlsx")
    dfv = pd.read_excel("Volume.xlsx")
        
    dff1 = dfl
    df1 = dfc  
    
    df2 = dfh    
    df3 = dfl
    df4 = dfv
    
    
    df2[s].rename("High",inplace = True)
    df3[s].rename("Low",inplace = True)
    df1[s].rename("Close",inplace = True)
    df4[s].rename("Volume",inplace = True)
    
    data = pd.concat([df2['Date'],df2[s],df3[s],df1[s],df4[s]],axis = 1)
    
    data.reset_index(drop = True,inplace = True)
    
    
    data['High'].interpolate(inplace = True)
    data['Close'].interpolate(inplace = True)
    data['Low'].interpolate(inplace = True)
    data['Volume'].interpolate(inplace = True)
    
    #if(indicator == "CCI"):
    
    data['CCI'] = ta.CCI(data['High'],data['Low'], data['Close'],timeperiod = 14)
    #data[['Close','CCI']].plot(figsize=(12,12))
    #plt.show()
    
    data['MA'] = ta.SMA(data['Close'],20)
    #data[['Close','MA']].plot(figsize=(12,12))
    #plt.show()
    
    data['EMA'] = ta.EMA(data['Close'], timeperiod = 20)
    #data[['Close','EMA']].plot(figsize=(12,12))
    #plt.show()
    
    data['up_band'],data['mid_band'],data['low_band'] = ta.BBANDS(data['Close'], timeperiod =20)
    #data[['Close','up_band','mid_band','low_band']].plot(figsize=(12,12))
    #plt.show()
    #data['BOLL'] = up_band.subtract(data['Close'])
    
    
    data['rsi'] = ta.RSI(data['Close'],14)
    #data[['Close','RSI']].plot(figsize=(12,12))
    #plt.show()
    #data['RSI'] = data['RSI']-70 
    
    data['BIAS'] = tal.bias(data['Close'])
    #daa[['Close','BIAS']].plot(figsize=(12,12))
    #plt.show()
    
    data['PSY'] = tal.psl(data['Close'])
    
    data['CMO'] = ta.CMO(data['Close'], timeperiod=20)
    
    data['ROC'] = ta.ROC(data['Close'], timeperiod=20)
    
    data['PPO'] = ta.PPO(data['Close'], fastperiod=12, slowperiod=26, matype=0)
    
    data['APO'] = ta.APO(data['Close'], fastperiod=12, slowperiod=26, matype=0)
    
    data['WMSR'] = ta.WILLR(data['High'],data['Low'], data['Close'], timeperiod=14)
    #data[['Close','WMSR']].plot(figsize=(12,12))
    
    macd, macdsignal, macdhist = ta.MACD(data['Close'])
    data['macd'] = macd
    data['macdsignal'] = macdsignal
    #data['macdsignal'] =  macdsignal
    #data[['macd','macdsignal']].plot(figsize=(12,12))
    
    #data['EMA5'] = ta.EMA(data['Close'], timeperiod = 5)
    #data['EMA10'] = ta.EMA(data['Close'], timeperiod = 10)
    
    data['AR'] = ta.AROONOSC(data['High'],data['Low'], timeperiod=14)
    
    data["VR"] = tal.pvr(data['Close'],data['Volume'])
    
    kc = tal.kc(data['High'],data['Low'], data['Close'])
    data = data.join(kc)
    #data['KC']= data['KCUe_20_2'].subtract(data['Close'])
    #data.drop(['KCUe_20_2','KCBe_20_2','KCLe_20_2'],axis = 1,inplace = True)
    #data = data.join(kc)
    
    
    k =tal.kdj(data['High'],data['Low'], data['Close'])
    data = data.join(k)
    #data["KDJ"] = data["K_9_3"]-80
    #data.drop(["K_9_3","D_9_3","J_9_3"],axis = 1,inplace = True )
    #data = data.join(k)
    
    data['SAR'] = ta.SAR(data['High'],data['Low'], acceleration=0, maximum=0)
    
    data1 = data.copy()
    data2 = data.copy()
    
    data.dropna(inplace = True)
    
    data.reset_index(inplace= True,drop = True)  
    
    
    
    
    
    data['RSI'] = 0.0
    length = len(data)
    for epoch in range(length):
        if data.loc[epoch,'rsi']> 70 :
            data.loc[epoch,'RSI'] = 3
        elif data.loc[epoch,'rsi']< 30:
            data.loc[epoch,'RSI'] = 1
        else:
            data.loc[epoch,'RSI'] = 2
     
    
    data.drop(['rsi'],axis=1,inplace = True)   
    
    
    # In[86]:
    
    
    data['MACD'] = 0.0
    length = len(data)
    for epoch in range(length):
        if data.loc[epoch,'macd']> data.loc[epoch,'macdsignal'] :
            data.loc[epoch,'MACD'] = 3
        elif data.loc[epoch,'macd']< data.loc[epoch,'macdsignal']:
            data.loc[epoch,'MACD'] = 1
        else:
            data.loc[epoch,'MACD'] = 2
     
    
    data.drop(['macd','macdsignal'],axis=1,inplace = True) 
    
    
    # In[87]:
    
    
    data['BOLL'] = 0.0
    length = len(data)
    for epoch in range(length):
        if data.loc[epoch,'Close']> data.loc[epoch,'up_band'] :
            data.loc[epoch,'BOLL'] = 3
        elif data.loc[epoch,'Close']< data.loc[epoch,'low_band']:
            data.loc[epoch,'BOLL'] = 1
        else:
            data.loc[epoch,'BOLL'] = 2
     
    
    data.drop(['up_band','low_band','mid_band'],axis=1,inplace = True) 
    
    
    # In[88]:
    
    
    data['KC'] = 0.0
    length = len(data)
    for epoch in range(length):
        
        if data.loc[epoch,'Close']>data.loc[epoch,'KCUe_20_2'] :
            data.loc[epoch,'KC'] = 3
        elif data.loc[epoch,'Close']<data.loc[epoch,'KCLe_20_2'] :
            data.loc[epoch,'KC'] = 1
        else:
            data.loc[epoch,'KC'] = 2
    data.drop(["KCUe_20_2","KCLe_20_2","KCBe_20_2"],axis = 1,inplace = True )
    
    
    
    
    
    data['KDJ'] = 0.0
    length = len(data)
    for epoch in range(length):
        
        if 80 <data.loc[epoch,"K_9_3"] :
            data.loc[epoch,'KDJ'] = 3
        elif 20 >data.loc[epoch,"D_9_3"] :
            data.loc[epoch,'KDJ'] = 1
        else:
            data.loc[epoch,'KDJ'] = 2
    data.drop(["K_9_3","D_9_3","J_9_3"],axis = 1,inplace = True )
    
    X = data.iloc[:,5:25]
    
    y = np.where(data['Close'].shift(-1) > data['Close'],1,0)
    data['Target'] = pd.Series(y)
    
    
    X.drop(['BIAS','KDJ','BOLL','PSY','AR','ROC'],axis = 1,inplace = True)  #0.5
    #X.drop(['MACD'],axis = 1,inplace = True)
    
    
    # In[105]:
    
    
    #split = int(0.7*len(data))
    #X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 4)
    
    
    
    # scaling the features
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    
    # In[107]:
    
    
    model = LogisticRegression()
    model.fit(X_train_scaled,y_train)
    #print(list(zip(model.coef_, model_features)))
    #print(r.summary())
    #print(X.columns , 
          #np.transpose(r.coef_))
    #coefficients = pd.DataFrame({"Feature":X.columns,"Coefficients":(r.coef_)})
    #c = pd.Series(data = r.coef_[0],index = X.columns)
    #print(c.sort_values())
    #print(coefficients)
    
    d = len(X)

    res = X.loc[d-1,:]
  
    res = res.to_frame()
  
    res = res.transpose()  
    #print(res)
    
    
    result = model.predict_proba(res)
    
    
    
    
    # In[176]:
    
    
    result  = pd.DataFrame(result)
    
    
    
    
   # n = len(result)
    #result['prediction'] = 'neutral'
    
    #if (result.loc[0,0]>result.loc[0,1]):
        #if result.loc[0,0] >= 0.6:
            #result.loc[0,'prediction'] = 'oversold'
        #else:
             #prob.loc[epoch,'prediction'] = 'neutral'
    #else:
        #if result.loc[0,1] >= 0.6:
            #result.loc[0,'prediction'] = 'overbought'
        #else:
             #prob.loc[epoch,'prediction'] = 'neutral'
    
    
    # In[180]:
    result = result.loc[0,1] 
    #label = 1 rise in price #1 is column name for prediction proba. of label 1 
    #result.drop([0,1],axis =1,inplace = True)
    #print(result.to_dict())  ########step1
    resultdic = {}
    resultdic['Tickername'] = s
    resultdic['Auroscore'] = result
    
    data1.reset_index(inplace= True,drop = True)
    data1.dropna(inplace= True)
    t = len(data1)
    data1['EMA5'] = ta.EMA(data['Close'], timeperiod = 5)
    data1['EMA10'] = ta.EMA(data['Close'], timeperiod = 10)
    df = data1.loc[t-1,:]
    dff = df.to_frame()
    dff = dff.transpose()
    
    dff['Prediction'] = "NULL"
    resultdic['SAR'] = {}
           
    if dff.loc[t-1,"Close"]>dff.loc[t-1, 'SAR'] :
        dff.loc[t-1, 'SAR'] = 'Buysignal'
        resultdic['SAR']['prediction'] = 'Buysignal'
    elif dff.loc[t-1,"Close"]<dff.loc[t-1, 'SAR'] :
        dff.loc[t-1, 'SAR'] = 'sellsignal'
        resultdic['SAR']['prediction'] = 'sellsignal'
    else:
        dff.loc[t-1, 'SAR'] = 'neutral'
        resultdic['SAR']['prediction'] = 'neutral'
    
    resultdic['KC'] = {}
    if dff.loc[t-1,"Close"]>dff.loc[t-1 ,'KCUe_20_2'] :
        dff.loc[t-1,'KC'] = 'Buysignal'
        resultdic['KC']['prediction'] = 'Buysignal'
    elif dff.loc[t-1,"Close"]<dff.loc[t-1,'KCLe_20_2'] :
        dff.loc[t-1,'KC'] = 'sellsignal'
        resultdic['KC']['prediction'] = 'sellsignal'
    else:
        dff.loc[t-1,'KC'] = 'neutral'
        resultdic['KC']['prediction'] = 'neutral'
    dff.drop(["KCLe_20_2","KCBe_20_2","KCUe_20_2"],axis = 1,inplace = True )
    
    resultdic['KDJ'] = {}
    if 20 >dff.loc[t-1, 'D_9_3'] :
        dff.loc[t-1,'KDJ'] = 'oversold'
        resultdic['KDJ']['prediction'] = 'oversold'
    elif 80<dff.loc[t-1, 'K_9_3'] :
        dff.loc[t-1,'KDJ'] = 'overbought'
        resultdic['KDJ']['prediction'] = 'overbought'
    else:
        dff.loc[t-1,'KDJ'] = 'neutral'
        resultdic['KDJ']['prediction'] = 'neutral'
    dff.drop(['K_9_3','D_9_3','J_9_3'],axis = 1,inplace = True )
    
    resultdic['VR'] = {}
    if dff.loc[t-1,"VR"]>2.5:
        dff.loc[t-1,"VR"] = 'sellsignal'
        resultdic['VR']['prediction'] = 'sellsignal'
    elif dff.loc[t-1,"VR"]<2.5:
        dff.loc[t-1,"VR"] = 'Buysignal'
        resultdic['VR']['prediction'] = 'Buysignal'
    else:
        dff.loc[t-1,"VR"] = 'neutral'
        resultdic['VR']['prediction'] = 'neutral'
    
    
    
    resultdic['CCI'] = {}
    if dff.loc[t-1,"CCI"]>100:
        dff.loc[t-1,"CCI"] = 'overbought'
        resultdic['CCI']['prediction'] = 'overbought'
    elif dff.loc[t-1,"CCI"]<-100:
        dff.loc[t-1,"CCI"] = 'oversold'
        resultdic['CCI']['prediction'] = 'oversold'
    else:
        dff.loc[t-1,"CCI"] = 'neutral'
        resultdic['CCI']['prediction'] = 'neutral'
    
    resultdic['PSY'] = {}
    if dff.loc[t-1,"PSY"]>70:
        dff.loc[t-1,"PSY"] = 'overbought'
        resultdic['PSY']['prediction'] = 'overbought'
    elif dff.loc[t-1,"PSY"]<30:
        dff.loc[t-1,"PSY"] = 'oversold'
        resultdic['PSY']['prediction'] = 'oversold'
    else:
        dff.loc[t-1,"PSY"] = 'neutral'
        resultdic['PSY']['prediction'] = 'neutral'
    resultdic['WMSR'] = {}
    if dff.loc[t-1 , "WMSR"]>-20:
         dff.loc[t-1 , "WMSR"] = 'overbought'
         resultdic['WMSR']['prediction'] = 'overbought'
    elif dff.loc[t-1 , "WMSR"]<-80:
         dff.loc[t-1 , "WMSR"] = 'oversold'
         resultdic['WMSR']['prediction'] = 'oversold'
    else:
         dff.loc[t-1 , "WMSR"] = 'neutral'
         resultdic['WMSR']['prediction'] = 'neutral'
    
    
    
    resultdic['MA'] = {}
    if dff.loc[t-1,"Close"]>dff.loc[t-1 ,"MA"] :
        dff.loc[t-1 ,"MA"] = 'Buysignal'
        resultdic['MA']['prediction'] = 'Buysignal'
    elif dff.loc[t-1,"Close"]<dff.loc[t-1 ,"MA"] :
        dff.loc[t-1 ,"MA"] = 'sellsignal'
        resultdic['MA']['prediction'] = 'sellsignal'
    else:
        dff.loc[t-1 ,"MA"] = 'neutral'
        resultdic['MA']['prediction'] = 'neutral'
    
    resultdic['BOLL'] = {}
    if dff.loc[t-1,"Close"]>dff.loc[t-1,"up_band"] :
        dff.loc[t-1,"BOLL"] = 'overbought'
        resultdic['BOLL']['prediction'] = 'overbought'
    elif dff.loc[t-1,"Close"]<dff.loc[t-1,"low_band"] :
        dff.loc[t-1,"BOLL"] = 'oversold'
        resultdic['BOLL']['prediction'] = 'oversold'
    else:
        dff.loc[t-1,"BOLL"] = 'neutral'
        resultdic['BOLL']['prediction'] = 'neutral'
    dff.drop(["up_band","low_band","mid_band"],axis=1,inplace = True)
    
    
    # In[3635]:
    
    resultdic['RSI'] = {}
    if dff.loc[t-1 ,"rsi"]>70:
        dff.loc[t-1 ,"rsi"] = 'overbought'
        resultdic['RSI']['prediction'] = 'overbought'
    elif dff.loc[t-1 ,"rsi"]<30:
        dff.loc[t-1 ,"rsi"] = 'oversold'
        resultdic['RSI']['prediction'] = 'oversold'
    else:
        dff.loc[t-1 ,"rsi"] = 'neutral'
        resultdic['RSI']['prediction'] = 'neutral'
    
    
    resultdic['PPO'] = {}
    if dff.loc[t-1,"PPO"]>10:
        dff.loc[t-1,"PPO"] = 'overbought'
        resultdic['PPO']['prediction'] = 'overbought'
    elif dff.loc[t-1,"PPO"]<-10:
        dff.loc[t-1,"PPO"] = 'oversold'
        resultdic['PPO']['prediction'] = 'oversold'
    else:
        dff.loc[t-1,"PPO"] = 'neutral'
        resultdic['PPO']['prediction'] = 'neutral'
    
    
    # In[3639]:
    
    resultdic['APO'] = {}
    if dff.loc[t-1,"APO"]>0:
        dff.loc[t-1,"APO"] = 'upwardtrend'
        resultdic['APO']['prediction'] = 'upwardtrend'
        
    elif dff.loc[t-1,"APO"]<0:
        dff.loc[t-1,"APO"] = 'downwardtrend'
        resultdic['APO']['prediction'] = 'downwardtrend'
    else:
        dff.loc["APO","Prediction"] = 'neutral'
        resultdic['APO']['prediction'] = 'neutral'
    resultdic['AR'] = {}
    if dff.loc[t-1,"AR"]>0:
        dff.loc[t-1,"AR"] = 'upwardtrend'
        resultdic['AR']['prediction'] = 'upwardtrend'
    elif dff.loc[t-1,"AR"]<0:
        dff.loc[t-1,"AR"] = 'downwardtrend'
        resultdic['AR']['prediction'] = 'downwardtrend'
    else:
        dff.loc[t-1,"AR"] = 'neutral'
        resultdic['AR']['prediction'] = 'neutral'
        
    
    # In[3640]:
    
    resultdic['ROC'] = {}
    if dff.loc[t-1,"ROC"]>0:
        dff.loc[t-1,"ROC"] = 'upwardtrend'
        resultdic['ROC']['prediction'] = 'upwardtrend'
    elif dff.loc[t-1,"ROC"]<0:
        dff.loc[t-1,"ROC"] = 'downwardtrend'
        resultdic['ROC']['prediction'] = 'downwardtrend'
    else:
        dff.loc[t-1,"ROC"] = 'neutral'
        resultdic['ROC']['prediction'] = 'neutral'
    
    
    # In[3641]:
    
    resultdic['CMO'] = {}
    if dff.loc[t-1,"CMO"]>50:
        dff.loc[t-1,"CMO"] = 'upwardtrend'
        resultdic['CMO']['prediction'] = 'upwardtrend'
    elif dff.loc[t-1,"CMO"]<-50:
        dff.loc[t-1,"CMO"] = 'downwardtrend'
        resultdic['CMO']['prediction'] = 'downwardtrend'
    else:
        dff.loc[t-1,"CMO"] = 'neutral'
        resultdic['CMO']['prediction'] = 'neutral'
    
    
    # In[3642]:
    
    
    resultdic['MACD'] = {}
    if dff.loc[t-1,"macd"]>dff.loc[t-1,"macdsignal"] :
        dff.loc[t-1,"macd"] = 'overbought'
        resultdic['MACD']['prediction'] = 'overbought'
    elif dff.loc[t-1,"macd"]<dff.loc[t-1,"macdsignal"] :
        dff.loc[t-1,"macd"] = 'oversold'
        resultdic['MACD']['prediction'] = 'oversold'
    else:
        dff.loc[t-1,"macd"] = 'neutral'
        resultdic['MACD']['prediction'] = 'neutral'
    dff.drop(["macdsignal"],axis = 1,inplace = True )
    
    
    # In[3643]:
    
    resultdic['EMA'] = {}
    if dff.loc[t-1, "EMA5"]>dff.loc[t-1,"EMA10"] :
        dff.loc[t-1, "EMA"] = 'overbought'
        resultdic['EMA']['prediction'] = 'overbought'
    elif dff.loc[t-1,"EMA10"]<dff.loc[t-1, "EMA5"] :
        dff.loc[t-1, "EMA"] = 'oversold'
        resultdic['EMA']['prediction'] = 'oversold'
    else:
        dff.loc[t-1, "EMA"] = 'neutral'
        resultdic['EMA']['prediction'] = 'neutral'
    dff.drop(["EMA5","EMA10"],axis = 1,inplace = True )
    
    
    dff.drop(["Date","High","Low","Close","Volume","BIAS","Prediction"],axis=1,inplace = True)
    
   
    #dff = dff.to_dict(orient = 'list')
    #resultdic.update(dff)
    df = data2
    df['EMA5'] = ta.EMA(data['Close'], timeperiod = 5)
    df['EMA10'] = ta.EMA(data['Close'], timeperiod = 10)
    df.dropna(inplace = True)
    df.reset_index(inplace= True,drop = True)
    
    
    indicators ={  
    "MA" : 20,
    "EMA" : 20,
    "BOLL" :20,
    "RSI":14,
    "MACD":9, 
    "CCI":14,     
    #BIAS
    #PER
    #MAVOL
    "PSY": 0,
    "WMSR": 20,    
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
    ###step3 
    #mydict = {}
     
    df['avg'] = ((df['Close'].shift(-1)-df['Close'])/df['Close'])*100
    maxval = df['avg'].max()
    
    minval = df['avg'].min()
    avgchange = df['avg'].mean()
    
    for indicator, time in indicators.items():
        length = len(df)
        count = 0
        count_rise = 0
        count_sell = 0
        if(indicator == 'SAR'):
                    
            for epoch in range(length):
    
                if df.loc[epoch,"Close"]>df.loc[epoch, indicator] :
    
                    count_rise = count_rise+1
                    count = count+1
                elif df.loc[epoch,"Close"]<df.loc[epoch, indicator] :
    
                    count_sell = count_sell+1
                    count = count+1
        elif(indicator == 'KC' ):
            for epoch in range(length ):   
                if df.loc[epoch,"Close"]>df.loc[epoch ,'KCUe_20_2'] :
                    count_rise = count_rise+1
                    count = count+1
    
                elif df.loc[epoch,"Close"]<df.loc[epoch,'KCLe_20_2'] :
                    count_sell = count_sell+1
                    count = count+1
        elif(indicator == 'KDJ'):
            for epoch in range(length ):   
                if 90<df.loc[epoch ,'K_9_3'] :
                    count_rise = count_rise+1
                    count = count+1
    
                elif 20 >df.loc[epoch, 'D_9_3'] :
                    count_sell = count_sell+1
                    count = count+1
        elif(indicator == 'VR'):
            for epoch in range(length ):   
                if 2.5<df.loc[epoch ,'VR'] :
                    count_rise = count_rise+1
                    count = count+1
    
                elif 2.5 >df.loc[epoch, 'VR'] :
                    count_sell = count_sell+1
                    count = count+1
        elif(indicator == 'CCI'):
            for epoch in range(length ):   
                if 100<df.loc[epoch ,'CCI'] :
                    count_rise = count_rise+1
                    count = count+1
    
                elif -100 >df.loc[epoch, 'CCI'] :
                    count_sell = count_sell+1
                    count = count+1
        elif(indicator == 'PSY'):
            for epoch in range(length ):   
                if 70<df.loc[epoch ,'PSY'] :
                    count_rise = count_rise+1
                    count = count+1
    
                elif 30 >df.loc[epoch, 'PSY'] :
                    count_sell = count_sell+1
                    count = count+1
        elif(indicator == 'WMSR'):
            for epoch in range(length ):   
                if -20<df.loc[epoch ,'WMSR'] :
                    count_rise = count_rise+1
                    count = count+1
    
                elif -80>df.loc[epoch, 'WMSR'] :
                    count_sell = count_sell+1
                    count = count+1
        elif(indicator == 'MA'):
            for epoch in range(length ):   
                if -20<df.loc[epoch ,'WMSR'] :
                    count_rise = count_rise+1
                    count = count+1
    
                elif -80>df.loc[epoch, 'WMSR'] :
                    count_sell = count_sell+1
                    count = count+1
                
        elif(indicator == "RSI"):
            for epoch in range(length ):   
                if 70<df.loc[epoch ,"rsi"] :
                    count_rise = count_rise+1
                    count = count+1
    
                elif 30>df.loc[epoch, "rsi"] :
                    count_sell = count_sell+1
                    count = count+1
            
        elif(indicator == "PPO"):
            for epoch in range(length ):   
                if 10<df.loc[epoch ,"PPO"] :
                    count_rise = count_rise+1
                    count = count+1
    
                elif -10>df.loc[epoch, "PPO"] :
                    count_sell = count_sell+1
                    count = count+1
                
        elif(indicator == "APO"):
            for epoch in range(length ):   
                if 0<df.loc[epoch ,"APO"] :
                    count_rise = count_rise+1
                    count = count+1
    
                elif 0>df.loc[epoch, "APO"] :
                    count_sell = count_sell+1
                    count = count+1
       
        elif(indicator == "AR"):
            for epoch in range(length ):   
                if 0 <df.loc[epoch ,"AR"] :
                    count_rise = count_rise+1
                    count = count+1
    
                elif 0>df.loc[epoch, "AR"] :
                    count_sell = count_sell+1
                    count = count+1
    
        elif(indicator == "ROC"):
            for epoch in range(length ):   
                if 0<df.loc[epoch ,"ROC"] :
                    count_rise = count_rise+1
                    count = count+1
    
                elif 0>df.loc[epoch, "ROC"] :
                    count_sell = count_sell+1
                    count = count+1
                
        elif(indicator == "CMO"):
            for epoch in range(length ):   
                if 50<df.loc[epoch ,"CMO"] :
                    count_rise = count_rise+1
                    count = count+1
    
                elif -50>df.loc[epoch, "CMO"] :
                    count_sell = count_sell+1
                    count = count+1
        elif(indicator == "MA"):
            for epoch in range(length ):
    
                if df.loc[epoch,"Close"]>df.loc[epoch ,"MA"] :
                    count_rise = count_rise+1
                    count = count+1
                elif df.loc[epoch ,"MA"]<df.loc[epoch,"Close"] :
                    count_sell = count_sell+1
                    count = count+1
        elif(indicator == "BOLL"):
            for epoch in range(length ):
    
                if df.loc[epoch,"Close"]>df.loc[epoch ,"up_band"] :
                    count_rise = count_rise+1
                    count = count+1
                elif df.loc[epoch ,"low_band"]<df.loc[epoch,"Close"] :
                    count_sell = count_sell+1
                    count = count+1
                
        elif(indicator == "MACD"):
            for epoch in range(length ):
    
                if df.loc[epoch,"macd"]>df.loc[epoch ,"macdsignal"] :
                    count_rise = count_rise+1
                    count = count+1
                elif df.loc[epoch ,"macd"]<df.loc[epoch,"macdsignal"] :
                    count_sell = count_sell+1
                    count = count+1
                    
        elif(indicator == "EMA"):
            for epoch in range(length ):
                
                    
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
             
        resultdic[indicator]['AVG_Change']= avgchange
        resultdic[indicator]['MAX_Increase']= maxval
        resultdic[indicator]['MAX_Decline']= minval
        
        
                        
        
        
       

    #resultdic.update(mydict)
    
    
    
    
    
    return resultdic #########step2


