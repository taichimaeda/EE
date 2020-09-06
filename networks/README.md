- All the data is stored in `data/[dataset name]_[optimizer name]_[model name]/[trial number]/`
  - The metrics (acc, loss, val_acc and val_loss) recorded by CSV Logger is in `result.csv`
  - The time duration required for each epoch is stored in `time.csv`
  - The TensorBoard logs are stored in `tensorboard/`
  
- For monitoring the experiments, a library called Hyperdash was used. In order to use the codes, the API key must be provided in `config.json` as the following:

  ```json
  {
    "hyperdash": {
      "api_key": "YOUR API KEY"
    }
  }
  ```

  
