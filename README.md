# EE GitHub Repository

- All the data is stored in `data/`
- All the graphs are stored in `graphs/`
- All the Python codes are stored in `src/`

## Part 1: Benchmark Comparison

- Go to `benchmarks/`
- All the data on hyperparameter tuning are stored in `params/[dataset name]_[optimizer name]_[model name]/`
  - The optimized values of hyperparameter are stored in `result.csv`
  - The log is stored in `log.csv`
- All the data are stored in `data/[benchamrk name]_[optimizer name]/`
  - The distance of each starting point is stored in `result.csv` 
  - The average time taken for each step is stored in `time.csv` 

## Part 2: Network Comparison

- Go to `networks/`

- All the data on hyperparameter tuning are stored in `params/[dataset name]_[model name]_[optimizer name]/`

  - The optimized values of hyperparameter are stored in result.csv`
  - The log is stored in `log.csv`

- All the data are stored in `data/[dataset name]_[optimizer name]_[model name]/[trial number]/`

  - The learning metrics (acc, loss, val_acc and val_loss) recorded by `CSVLogger` is stored in `result.csv`
  - The average time taken for each epoch is stored in `time.csv`
  - TensorBoard logs are stored in `tensorboard/`

- Hyperdash was used for monitoring the experiments. In order to run the programs, the API key must be provided in `config.json in the following format:

  ```json
  // config.json
  {
    "hyperdash": {
      "api_key": "YOUR API KEY"
    }
  }
  ```