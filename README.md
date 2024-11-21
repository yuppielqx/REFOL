# REFOL: Resource-Efficient Federated Online Learning for Traffic Flow Forecasting
 
This is an official PyTorch implementation of "REFOL: Resource-Efficient Federated Online 
Learning for Traffic Flow Forecasting"

## Data

One can download the Traffic data files as per [`DCRNN`](https://github.com/liyaguang/DCRNN/blob/master/README.md). Then extract it to the root directory of the repository:

Run the following commands to generate training dataset at  `data/{METR-LA,PEMS-BAY}/*_series.npz`.
```bash
# Create data directories
mkdir -p data/{METR-LA,PEMS-BAY}

cd data

# METR-LA
python -m split_data_increment.py --pred_steps=1 --output_dir=./METR-LA --traffic_df_filename=./metr-la.h5

# PEMS-BAY
python -m split_data_increment.py --pred_steps=1 --output_dir=./PEMS-BAY --traffic_df_filename=./pems-bay.h5
```

## Model Train

```bash
cd ..

python run.py
```
One can adjust the configurations in `default_config.yaml`.



## Acknowledgements

We appreciate the following repository for sharing the valuable code base and datasets:

https://github.com/mengcz13/KDD2021_CNFGNN/tree/master

https://github.com/liyaguang/DCRNN/blob/master