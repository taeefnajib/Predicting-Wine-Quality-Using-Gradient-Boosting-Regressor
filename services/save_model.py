import pickle
import bentoml

with open("/userRepoData/taeefnajib/Predicting-Wine-Quality-Using-Gradient-Boosting-Regressor/sidetrek/models/76e4ff187d783fbb1ebbb0ea205b8ada", "rb") as pickle_file:
    model = pickle.load(pickle_file)
    saved_model = bentoml.sklearn.save_model("model", model)
    print(saved_model) # This is required!
