# Model A: structured-credit baselines

Run: `pip install -r requirements-model-a.txt` then `python src/model_a/train_models.py --csv data/raw/accepted_2007_to_2018Q4.csv`.

Trains Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM and CatBoost, and writes metrics and fitted models to `results/model_a/`.

**Research limitations:** This first benchmark predicts terminal Charged Off vs Fully Paid among resolved loans; it is not a calibrated 12-month probability of default. Filtering resolved loans can introduce outcome-maturity and selection bias. Before making credit decisions, define a fixed performance horizon, restrict to mature cohorts, validate features at decision time, tune hyperparameters on a separate temporal validation period, check fairness and calibration, and perform an untouched out-of-time test. Rejected applications have no observed repayment outcome and are not merged into the supervised target.
