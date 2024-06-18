## 'preprocessed_*' datafile descriptions

NOTE: On 15.7 a bug in duplicate checking was discovered. This bug caused UAV designs with different controller indices to be considered as unique. This was fixed, and all data re-preprocessed.

Pre-processed datafiles contain UAV designs from the original Hyform dataset. They are aggregates of files 'designerAI.csv', 'dronedb.csv', 'dronedb_log.csv', 'uavdb.csv' and 'ai_designer_sequences_updated.csv'
located in the 'raw' data folder.

### preprocessed_allunique.csv
Contains all unique UAV designs found in the raw datafiles. Uniqueness in this case defined as unique UAV design strings.
Has ~~17362~~ 16050 UAV designs. Also contains metrics (range, cost, velocity) from simulator, computed by the HyForm authors.

TLDR: Use the 'aggregated_uav_designs.csv' for the largest dataset of UAV design

### preprocessed_validunique.csv
Contains all unique UAV designs that are a form of valid UAV Grammar, as defined by the UAVGrammar class (data/datamodels/Grammar)
Has ~~17230~~ 15912 valid UAV designs. 138 invalid designs discarded from the 16050 designs in 'preprocessed_allunique.csv'. 
Please see valid_drone_pplog.txt for specific invalidation grounds.

Also contains metrics (range, cost, velocity) from simulator, computed by the HyForm authors.

### simresults_preprocessed_validunique.csv

Contains all the designs from preprocessed_validunique.csv, but with simulator results
**computed by us**. I.e, contains metrics from the simulator, and the final result (Success or failure type).

Outcome statistics:
- ~~3746~~ 3685 designs with simulator outcome 'Success'
- ~~12555~~ 11329 designs with simulator outcome 'CouldNotStabilize'
- ~~444~~ 418 designs with simulator outcome 'HitBoundary'
- 486 designs with simulator outcome 'Error' (Simulator exited w/o an outcome)
- 1 designs with simulator outcome 'Timeout' (Simulator hanged for 30 seconds and was terminated)


## 'aggregated_*' datafile descriptions

Aggregated datafiles are aggregates of the designs from the original HyForm dataset, and generated designs.
I.e these datafiles (ought to) contain all UAV designs we're aware of currently.

### aggregated_uav_designs.csv

Outcome statistics:
-  9077 designs with simulator outcome 'Success'
-  21819 designs with simulator outcome 'CouldNotStabilize'
-  1273 designs with simulator outcome 'HitBoundary'
-  1051 designs with simulator outcome 'Error' (Simulator exited w/o an outcome)
-  6160 designs with simulator outcome 'Timeout' (Simulator hanged for 30 seconds and was terminated)

Contains an aggregate of data from:
- 'simresults_preprocessed_validunique.csv'
- '062722152144_sample1.csv'
- '062722152144_sample2.csv'
- '062722152144_sample3.csv'

Design count: 39375

### filtered_aggregated_uav_designs.csv

Version of 'aggregated_uav_designs.csv' w/o designs that got 'Timeout' or 'Error' result from the simulator.
Range column values truncated to 0 if they are negative.

UPDATE 26/8/22: Now also contains designs from the DQN experiments, total count: 114141

Design count: 32166

### balanced_filtered_aggregated_uav_designs.csv

A version of 'filtered_aggregated_uav_designs.csv' with a 50/50 class distribution of 'Success' vs. failure conditions (CouldNotStabilize, HitBoundary)

Design count: 18152



