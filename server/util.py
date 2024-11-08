import json
import numpy as np
import pickle
import joblib
import sklearn.linear_model  # Make sure to import the module

# Fix for deprecated sklearn module
if not hasattr(sklearn.linear_model, 'base'):
    sklearn.linear_model.base = sklearn.linear_model._base

__locations = None
__data_columns = None
__model = None

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Remap the old module path to the new one
        if module == 'sklearn.linear_model.base' and name == 'LinearRegression':
            module = 'sklearn.linear_model._base'
        return super().find_class(module, name)

def get_estimated_price(location, sqft, bhk, bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return round(__model.predict([x])[0], 2)

def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __data_columns
    global __locations
    global __model

    # Load the columns data
    with open("./artifacts/columns.json", "r") as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]  # First 3 columns are other features

    # Load the model using CustomUnpickler
    with open("./artifacts/banglore_home_prices_model.pickle", "rb") as f:
        __model = CustomUnpickler(f).load()

    print("loading saved artifacts...done")

def get_location_names():
    return __locations

def get_data_columns():
    return __data_columns