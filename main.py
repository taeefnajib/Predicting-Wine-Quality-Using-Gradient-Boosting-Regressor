import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from dataclasses_json import dataclass_json
from dataclasses import dataclass


@dataclass_json
@dataclass
class Hyperparameters(object):
    filepath: str = "WineQT.csv"
    test_size: float = 0.25
    random_state: int = 6
    n_estimators: int = 300
    learning_rate: float = 0.009
    max_depth: int = 3
    loss: str = "squared_error"
    min_samples_split: int = 5

hp = Hyperparameters()

# Collecting and preparing data
def create_dataframe(filepath):
    df = pd.read_csv(filepath)
    df.drop("Id", axis=1, inplace = True)
    return df


# Splitting train and test dataset
def split_dataset(df, test_size, random_state):
    X = df.drop(["quality"], axis=1)
    y = df["quality"]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# Building and fitting the model
def train_model(X_train, y_train, n_estimators, learning_rate, max_depth, random_state, min_samples_split, loss):
    model = GradientBoostingRegressor(n_estimators = n_estimators, learning_rate= learning_rate, max_depth= max_depth, random_state= random_state, min_samples_split= min_samples_split, loss= loss)
    return model.fit(X_train, y_train)


# Running the workflow
def run_wf(hp: Hyperparameters) -> GradientBoostingRegressor:
    df = create_dataframe(filepath=hp.filepath)
    X_train, X_test, y_train, y_test = split_dataset(df=df,
                                                    test_size=hp.test_size, 
                                                    random_state=hp.random_state)
    return train_model(X_train=X_train, y_train=y_train, n_estimators=hp.n_estimators, learning_rate=hp.learning_rate, max_depth=hp.max_depth, random_state=hp.random_state, loss=hp.loss, min_samples_split=hp.min_samples_split)

    

if __name__=="__main__":
    run_wf(hp=hp)