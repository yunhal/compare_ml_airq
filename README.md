
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

### model structure

Input

Output. 


This script generates csv or json output files for each site, each model, each pollution species, each list in the save_prefixes (below): 
save_prefixes = ('feature-statistics', 'prd', 'permuted-train', 'permuted-test', 'mae', 'feature-importance')


Turning the --pipeploteval flag on, it runs "review_evaluation" module that reads all the prd*csv output files under the runoutput and computes the evaluation statistics (e.g., NME,NMB and R2). It generates statistics_for_allsites_allspecies_allmodels.csv and barplots (in pdf files) that compare the model performance for each statistic. 

Turning the --pipeplotimportance flag on, it runs "review_feature_importance" module that reads all feature-importance-*json output files under the runoutput. feature-importance is based on "permuted_importance" for train and test. lime_feature_importance is currently commented out (not sure if it working).  



### Contributors: 
Drs. Zongru (Doris) Shao, Kai Fan, and Yunha Lee at CASUS, HZDR in Germany. 



