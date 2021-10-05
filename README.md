
# DNN-based air quality predictions

The model predicts air quality prediction based on Dense Neural Network, with varying # of layers. It compares the DNN-based prediction against the random forest-based and CMAQ predictions.  It uses the AQS observation data from the Pacific Northwest, U.S.



The below is how to execute our model:

```python
python main.py --flag2rf 0 --flagdensexl 0 --flagdensel 0 --flagdensem 0 --flagdenses 0 --flagdensexs 0 --flagnasmlp 1 --flagnasres 1
```

For example:

```python
python main.py --wrf_path put_WRF_data_path  --aqs_path put_AQS_data_path  --out_path put_output_path
```

Important note that WRF and AQS data are pre-processed data, not the original/raw data format. 



### Contributors: 
Drs. Zongru (Doris) Shao, Kai Fan, and Yunha Lee at CASUS, HZDR in Germany. 



