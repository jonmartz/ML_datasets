import h2o
from h2o.automl import H2OAutoML

dataset_path = 'datasets/ml_features.csv'
balance_classes = False

# Start the H2O cluster (locally)
h2o.init()

# Import a sample binary outcome train/test set into H2O
train = h2o.import_file(dataset_path, header=1)

# Identify predictors and response
x = train.columns
y = "status"
x.remove(y)

# For binary classification, response should be a factor
train[y] = train[y].asfactor()

# Run AutoML for 20 base models
aml = H2OAutoML(max_models=20, seed=1, balance_classes=balance_classes)
aml.train(x=x, y=y, training_frame=train, fold_column="FOLD")

# View the AutoML Leaderboard
lb = aml.leaderboard
lb.head(rows=lb.nrows)  # Print all rows instead of default (10 rows)

# The leader model is stored here
print(f'leader: {aml.leader}')
