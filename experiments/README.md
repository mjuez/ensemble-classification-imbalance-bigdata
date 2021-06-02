# Experimental results

The CSV structure for the experimental results is as follows. The listed elements
correspond to the CSV-file columns from left to right:

- **dataset**: The name of the dataset.
    - covtype_0vs2
    - covtype_0vs3
    - covtype_0vs4
    - covtype_1vs2
    - covtype_1vs3
    - covtype_1vs4
    - susy_ir4
    - susy_ir16
    - higgs_ir4
    - higgs_ir16
    - hepmass_ir4
    - hepmass_ir16
    - kddcup_normal_vs_dos
    - kddcup_dos_vs_r2l
    - ecbdl14_1m
    - ecbdl14_10m
- **algorithm**: The sampling method and the algorithm used. For resampling before
training the ensemble experiments, the syntax is \[resampling\]\_\[algorithm\]; while
for resampling within the ensemble, the syntax is \[algorithm\]\_\[resampling\].
    - **Resampling choices**: none, ros, rus, smote (only when resampling before), 
    rose (only when resampling before), ranbal (only when resampling within the ensemble).
    - **Algorithm choices**:
        - **Gini**: bag (bagging), rf (random forest), gbt (gradient boosting trees), 
        imbbag (resampling within bagging), imbgbt (resampling within gradient boosting 
        trees)
        - **Weighted Gini**: wbag (bagging), wrf (random forest), wgbt (gradient boosting 
        trees).
- **repetition**: Cross-validation repetition \[1-5\].
- **fold**: Cross-validation fold \[1-2\].
- **tp**: The number of true positives.
- **tn**: The number of true negatives.
- **fp**: The number of false positives.
- **fn**: The number of false negatives.
- **train_time**: The training time in nanoseconds.
- **predict_time**: The predict time in nanoseconds.
