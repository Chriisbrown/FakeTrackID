# FakeTrackID
## Data Preparation
Data conversion from root files to h5 files is needed to run the ML model training:
run python datasaver.py
### Prerequisites:
* Uproot
* Pandas

## Training
To run use python training_scheme.py X
where X is:
* NN for Neural Network training
* NNOptimiser for hyperparameter optimisation

* GBDT for GBDT training
* GBDTOptimiser for hyperparameter optimisation
### Prerequisites
For NN:
* Tensorflow 2.0
* Tensorflow_model_optimization (for pruning)
* Keras
* qkeras (for quantised aware training)
For GBDT:
* xgboost
* joblib (saving models in pickle format)

For Both:
* Pandas
* Sklearn (for metrics and utility functions)
* Comet_ml (for parameter logging, will also need an account and API key to log metrics)
* yaml (for parameter file parsing)






