import pandas as pd
import datetime as dt
import numpy as np
import glob
import os

gmtoff = pd.read_csv('/bigdata/casus/atmos/ML_model/aqs_sites.csv')
gmtoff['County Code'] = gmtoff['County Code'].apply(lambda x: str(x).zfill(3))
gmtoff['Site Number'] = gmtoff['Site Number'].apply(lambda x: str(x).zfill(4))
gmtoff['AQSID'] = gmtoff['State Code'] + gmtoff['County Code'] + gmtoff['Site Number']

models = ['2rf','denseXS','denseS','denseM','denseL','denseXL']
timelen = ['fullyear','warmmonths','coldmonths']


outputdir_base='/home/lee45/play/DLair/code-compare-ml-dl/modeloutput_'
exp = ['p5split','p5split_less_features','p5split_less_features_minmaxlog','p5split_less_features_quantile','p65split','p8split']

# save all the output from here to allexp_results
plotdir_base='/home/lee45/play/DLair/code-compare-ml-dl/results_postproc/'

try:
    os.makedirs(plotdir_base, exist_ok = True)
    print("Directory '%s' created successfully" %plotdir_base)
except OSError as error:
    print("Directory '%s' can not be created" %plotdir_base)
    


for d in exp:
    for t in timelen:
        for m in models:
            dt_files = glob.glob(outputdir_base+d+'/prd-'+m+'-pm25*'+t+'*.csv')
            print('test file',dt_files[0:2])
            tmp3 = pd.DataFrame()
            for f in dt_files:
                tmp = pd.read_csv(f)
                aqsid = f.split("-")[-2]
                aqsid = aqsid.zfill(9)
                tmp['index'] = pd.to_datetime(tmp['index'])
                tmp.index = tmp['index']
                delta = np.timedelta64(abs(gmtoff.loc[gmtoff['AQSID']==aqsid,'GMT Offset'].values[0]),'h')
                tmp.index = tmp.index - delta
                #compute daily average
                tmp = tmp[['truth', 'prediction']]
                r = pd.date_range(start=tmp.index.min(), end=tmp.index.max(), freq='1H')
                tmp = tmp.reindex(r)
                tmp['PMavg24hr'] = tmp['truth'].rolling(24, min_periods=18).mean()
                tmp['PM25_pred'] = tmp['prediction'].rolling(24, min_periods=18).mean()
                tmp['PMavg24hr_org'] = tmp['PMavg24hr'].shift(-23)
                tmp['PMavg24hr_pred'] = tmp['PM25_pred'].shift(-23)

                tmp2 = tmp[(tmp.index.hour == 0)]#.dropna(how='all')

                tmp2['AQI_day'] = pd.cut(round(tmp2['PMavg24hr_org'],1),
                                        [-np.inf, 12, 35.4, 55.4, 150.4, 250.4, np.inf],
                                        labels=[1, 2, 3, 4, 5, 6])
                tmp2['AQI_pred_day'] = pd.cut(round(tmp2['PMavg24hr_pred'],1),
                                        [-np.inf, 12, 35.4, 55.4, 150.4, 250.4, np.inf],
                                        labels=[1, 2, 3, 4, 5, 6])

                tmp2.index = tmp2.index.normalize()

                tmp2['site'] = aqsid

                tmp3 = pd.concat([tmp3, tmp2[['PMavg24hr_org','PMavg24hr_pred', 'AQI_day', 'AQI_pred_day', 'site']].dropna()])

            # Kai's R script has a problem with text with number, so it is renamed to have only text    
            if m == "2rf": 
                tmp3.to_csv(plotdir_base+'daily_pm25_twoRF_'+t+'_'+d+'.csv', index=True)
            else:
                tmp3.to_csv(plotdir_base+'daily_pm25_'+m+'_'+t+'_'+d+'.csv', index=True)

for d in exp:
    for t in timelen:
        for m in models:
            dt_files = glob.glob(outputdir_base+d+'/prd-'+m+'-o3*'+t+'*.csv')
            print(outputdir_base+d+'/prd-'+m+'-o3*'+t+'*.csv')
            tmp3 = pd.DataFrame()
            for f in dt_files:
                tmp = pd.read_csv(f)
                #print("debug tmp ", tmp[0:24])
                aqsid = f.split("-")[-2]
                aqsid = aqsid.zfill(9)
                tmp['index'] = pd.to_datetime(tmp['index'])
                tmp.index = tmp['index']
                delta = np.timedelta64(abs(gmtoff.loc[gmtoff['AQSID']==aqsid,'GMT Offset'].values[0]),'h')
                tmp.index = tmp.index - delta
                #compute daily average
                tmp = tmp[['truth', 'prediction']]*1000
                r = pd.date_range(start=tmp.index.min(), end=tmp.index.max(), freq='1H')
                tmp = tmp.reindex(r)
                tmp['O3_obs'] = tmp['truth'].rolling(8, min_periods=6).mean()
                
                #print("debug tmp df", tmp['O3_obs'][0:24],tmp['truth'][0:24] )
                
                tmp['O3_pred'] = tmp['prediction'].rolling(8, min_periods=6).mean()
                tmp['O3avg8hr_org'] = tmp['O3_obs'].shift(-7)
                tmp['O3avg8hr_pred'] = tmp['O3_pred'].shift(-7)
                tmp['O3_obs.maxdaily8hravg'] = tmp['O3avg8hr_org'].rolling(17, min_periods=13).max()
                tmp['O3_pred.maxdaily8hravg'] = tmp['O3avg8hr_pred'].rolling(17, min_periods=13).max() 
                
                #shift columns
                tmp['O3_obs.maxdaily8hravg'] = tmp['O3_obs.maxdaily8hravg'].shift(-16)
                tmp['O3_pred.maxdaily8hravg'] = tmp['O3_pred.maxdaily8hravg'].shift(-16)
                tmp2 = tmp[(tmp.index.hour == 7)]#.dropna(how='all')

                tmp2['AQI_day'] = pd.cut(round(tmp2['O3_obs.maxdaily8hravg']),
                                        [0, 54, 70, 85, 105, 200, np.inf],
                                        labels=[1, 2, 3, 4, 5, 6])
                tmp2['AQI_pred_day'] = pd.cut(round(tmp2['O3_pred.maxdaily8hravg']),
                                        [0, 54, 70, 85, 105, 200, np.inf],
                                        labels=[1, 2, 3, 4, 5, 6])
                
                tmp2.index = tmp2.index.normalize()
                
                tmp2['site'] = aqsid

                tmp3 = pd.concat([tmp3, tmp2[['O3_obs.maxdaily8hravg','O3_pred.maxdaily8hravg', 'AQI_day', 'AQI_pred_day', 'site']].dropna()])

            if m == "2rf": 
                tmp3.to_csv(plotdir_base+'daily_o3_twoRF_'+t+'_'+d+'.csv', index=True)        
            else:
                tmp3.to_csv(plotdir_base+'daily_o3_'+m+'_'+t+'_'+d+'.csv', index=True)

#compute AIRPACT daily PM2.5

print(plotdir_base+'daily_o3_denseXS_fullyear_'+exp[0]+'.csv')
print(plotdir_base+'daily_pm25_denseXS_fullyear_'+exp[0]+'.csv')

sites = pd.read_csv(plotdir_base+'daily_pm25_denseXS_fullyear_'+exp[0]+'.csv')
            
tmp3 = pd.DataFrame()
for s in set(sites['site']):
    s = str(s).zfill(9)
    print(s)
    for y in range(2018,2021):
        try:
            tmp = pd.read_csv('http://lar.wsu.edu/R_apps/'+str(y)+'ap5/data/byAQSID/'+s+'.apan')
        except:
            continue
        aqsid = s
        tmp['index'] = pd.to_datetime(tmp['DateTime'])
        tmp.index = tmp['index']
        delta = np.timedelta64(abs(gmtoff.loc[gmtoff['AQSID']==aqsid,'GMT Offset'].values[0]),'h')
        tmp.index = tmp.index - delta
        tmp = tmp[~tmp.index.duplicated(keep='first')]
        #compute daily average
        tmp = tmp[['PM2.5ap']]
        r = pd.date_range(start=tmp.index.min(), end=tmp.index.max(), freq='1H')
        tmp = tmp.reindex(r)
        tmp['PM25_ap'] = tmp['PM2.5ap'].rolling(24, min_periods=18).mean()
        tmp['PMavg24hr_ap'] = tmp['PM25_ap'].shift(-23)
        
        tmp2 = tmp[(tmp.index.hour == 0)]#.dropna(how='all')
        
        tmp2['AQI_ap_day'] = pd.cut(round(tmp2['PMavg24hr_ap'],1),
                                        [-np.inf, 12, 35.4, 55.4, 150.4, 250.4, np.inf],
                                        labels=[1, 2, 3, 4, 5, 6])
        
        tmp2.index = tmp2.index.normalize()
        
        tmp2['site'] = aqsid
        
        tmp3 = pd.concat([tmp3, tmp2[['PMavg24hr_ap','AQI_ap_day', 'site']].dropna()])

tmp3.to_csv(plotdir_base+'daily_pm25_AIRPACT.csv', index=True)

#compute AIRPACT MDA8
sites = pd.read_csv(plotdir_base+'daily_o3_denseXS_fullyear_'+exp[0]+'.csv')
                
tmp3 = pd.DataFrame()
for s in set(sites['site']):
    s = str(s).zfill(9)
    print(s)
    for y in range(2018,2021):
        try:
            tmp = pd.read_csv('http://lar.wsu.edu/R_apps/'+str(y)+'ap5/data/byAQSID/'+s+'.apan')
        except:
            continue
        aqsid = s
        tmp['index'] = pd.to_datetime(tmp['DateTime'])
        tmp.index = tmp['index']
        delta = np.timedelta64(abs(gmtoff.loc[gmtoff['AQSID']==aqsid,'GMT Offset'].values[0]),'h')
        tmp.index = tmp.index - delta
        tmp = tmp[~tmp.index.duplicated(keep='first')]
        #compute daily average
        tmp = tmp[['OZONEap']]
        r = pd.date_range(start=tmp.index.min(), end=tmp.index.max(), freq='1H')
        tmp = tmp.reindex(r)
        tmp['O3_ap'] = tmp['OZONEap'].rolling(8, min_periods=6).mean()
        tmp['O3avg8hr_ap'] = tmp['O3_ap'].shift(-7)
        tmp['O3_ap.maxdaily8hravg'] = tmp['O3avg8hr_ap'].rolling(17, min_periods=13).max() 
        
        #shift columns
        tmp['O3_ap.maxdaily8hravg'] = tmp['O3_ap.maxdaily8hravg'].shift(-16)
        tmp2 = tmp[(tmp.index.hour == 7)]#.dropna(how='all')
        
        tmp2['AQI_ap_day'] = pd.cut(round(tmp2['O3_ap.maxdaily8hravg']),
                                        [0, 54, 70, 85, 105, 200, np.inf],
                                        labels=[1, 2, 3, 4, 5, 6])
        
        tmp2.index = tmp2.index.normalize()
        
        tmp2['site'] = aqsid
        
        tmp3 = pd.concat([tmp3, tmp2[['O3_ap.maxdaily8hravg', 'AQI_ap_day', 'site']].dropna()])

tmp3.to_csv(plotdir_base+'daily_o3_AIRPACT.csv', index=True)