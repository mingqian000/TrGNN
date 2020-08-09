# Traffic Flow Prediction with Vehicle Trajectories

This repository is a PyTorch implementation of the model TrGNN in the paper [Traffic Flow Prediction with Vehicle Trajectories] (pending release).

model figure...


## Requirements

* PyTorch=0.4.1
* Python=3.7
* numpy=1.16.5
* scipy=1.3.1
* pandas=0.23.4
* folium=0.10.0
* geopy=1.20.0
* networkx=2.1
* statsmodels=0.9.0 (optional for VAR)
* scikit-learn=0.21.3 (optional for RF)



## Pipeline

> RawTaxiData --- `map matching` ---> ParsedTaxiData
> ParsedTaxiData --- `trajectory.py` ---> recovered_trajectory_df
> recovered_trajectory_df --- `trajectory_transition.py` ---> trajectory_transition
> recovered_trajectory_df --- `flow.py` ---> flow
> road_list & road_graph --- `train_model.py` ---> road_adj
> trajectory_transition & flow & road_adj --- `train_model.py` ---> TrGNN


## Dataset description

### 1. Road Network

The list of road segments are indexed in file `data/road_list.csv`:
| road_id |
|:----------:|
|103103595|
|103103090|
|...|

The road graph is constructed with [NetworkX](https://networkx.github.io/documentation/stable/tutorial.html) and saved in GML format in `data/road_graph.gml`. Each node represents a road segment (`label: road_id, length: in km`), and each directed edge represents the adjacency between to road segments (weight: exponential decay of distance).

### 2. Trajectories

Trajectories after map matching （refer to [Hidden Markov Map Matching](https://www.microsoft.com/en-us/research/publication/hidden-markov-map-matching-noise-sparseness/)） are saved at `data/ParsedTaxiData_YYYYMMDD.csv`:

| vehicle_id | time | matched_road_id |
|:----------:|:----------:|:----------:|
|XXXXXXX|14/03/2016 00:00:00|103047123|
|XXXXXXX|14/03/2016 00:00:05|103063511|
|...|...|...|

Note:
Files in the `data` folder contain dummy data for demo purpose. Real data have not been published due to confidentiality.


## Prepare input

### 1. Trajectory cleansing

Run '''python trajectory.py -d 20160314 >> log/trajectory0314.log'''.

Results are saved at `data/recovered_trajectory_df_20160314_20160314.csv`:

| vehicle_id |trajectory_id| time | road_id |scenario|
|:----------:|:----------:|:----------:|:----------:|:----------:|
|XXXXXXX|0|14/03/2016 00:00:00|103047123|0.1|
|XXXXXXX|0|14/03/2016 00:00:05|103063511|3.1|
|...|...|...|...|...|

Similarly for other dates.
 
Note: the `scenario` column is for reference only (as documented in `trajectory.py`) and can be ignored.


### 2. Flow aggregation

Run '''python flow.py -d 20160314 -i 15 >> log/flow0314.log'''.

Flows are aggregated in 15-minute intervals, and are saved at `data/flow_20160314_20160314.csv`:
|   | road_id_0 | road_id_1 | ... |
|:----------:|:----------:|:----------:|:----------:|
|14/03/2016 00:00:00| 33 | 67 | ... |
|14/03/2016 00:15:00| 21 | 89 | ... |
|...|...|...|...|

Similarly for other dates.


## Baseline approaches (optional)

For HA, run '''python baseline.py -m HA''' with correct `start_date` and `end_date`. Note: the dataset should cover more than 14 days.

For MA, run '''python baseline.py -m MA -D demo''' for demo purpose.

For VAR, run '''python baseline.py -m VAR -H 5 -D demo''' in 5-hop neighborhood for demo purpose.

For RF, run '''python baseline.py -m RF -H 5 -n 100 -D demo''' in 5-hop neighborhood with 100 trees for demo purpose. Note: the code takes longer to complete as it trains one model for each road segment separately. 

For the test set, ground truth results are saved at `result/MODEL_Y_true.pkl`, and predicted results are saved at `result/MODEL_Y_pred.pkl`.

For DCRNN, refer to its [PyTorch implementation](https://github.com/chnsh/DCRNN_PyTorch).


## TrGNN

### 1. Trajectory transition

(Optional) Run '''python trajectory_transition.py -d1 20160314 -d2 20160314 >> log/transition0314.log''' for one single date. Similarly for other dates.

The result is a tensor of shape `96 (# 15-minute intervals of day), 2404 (# road segments), 2404 (# road segments)` and is saved at `data/trajectory_transition_20160314_20160314.pkl`.

Run '''python trajectory_transition.py -d1 START_DATE -d2 END_DATE >> log/transition0314.log''' for the training period.


### 2.  Train and test TrGNN

To train and test TrGNN, run '''python train_model.py -m TrGNN -D demo''' with GPU for demo purpose.
To train and test TrGNN-, run '''python train_model.py -m TrGNN- -D demo''' with GPU for demo purpose.

Trained models are saved at `model/[MODEL]_[TIMESTAMP]_[EPOCH]epoch.cpt` whenever the validation MAE breaks through. For the test set, ground truth results are save at `result/[MODEL]_[TIMESTAMP]_Y_true.pkl`, and predicted results are saved at `result/[MODEL]_[TIMESTAMP]_[EPOCH]epoch_Y_pred.pkl`. Results are of shape `# test intervals, # road segments`.


### 3. Experimental result

We run this repository on `SG-TAXI` dataset (not released) and results are summarized in the paper (not released).


## Citation
(pending...)