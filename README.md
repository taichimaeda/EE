# EE Experiments

- In each directory, `main.ipynb` is the notebook to begin experiments with on a Google Colaboratory instance, and `summary.ipynb` is the notebook for processing the graphs
  - All the data is stored in `data/`
  - All the Python codes are stored in `src/`
  
- Some reference materials are stored in `/references/`

## Hyperparameter Optimization

- Go to `/hyperopt/`
- All the hyperparameter tuning data and programs are stored in `benchmarks/`for Part 1 and `networks/` for Part 2

### For Part 1:

- The optimized values of hyperparameter are stored in `data/[dataset name]_[optimizer name]_[model name]/result.csv`
- The log is stored in `data/[dataset name]_[optimizer name]_[model name]/log.csv`

### For Part 2:

- The optimized values of hyperparameter are stored in `data/[dataset name]_[model name]_[optimizer name]/result.csv`
- The log is stored in `data/[dataset name]_[model name]_[optimizer name]/log.csv`

## Benchmark Comparison

- Go to `/benchmarks/`
- The distance of each starting point is stored in `data/[benchamrk name]_[optimizer name]/results.csv` 
- The average time duration required for each step is stored in `data/[benchmark name]_[optimizer name]/time.csv` 

## Network Comparison

- Go to `networks/`

- All the data is stored in `data/[dataset name]_[optimizer name]_[model name]/[trial number]/`

  - The learning metrics (acc, loss, val_acc and val_loss) recorded by `CSVLogger` is stored in `result.csv`
  - The time duration taken for each epoch is stored in `time.csv`
  - TensorBoard logs are stored in `tensorboard/`

- We used a library called Hyperdash for monitoring the experiments. In order to run the programs, the API key must be provided in `config.json in the following format:

  ```json
  // config.json
  {
    "hyperdash": {
      "api_key": "YOUR API KEY"
    }
  }
  ```