from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    MinMaxScaler
)
from sklearn.pipeline import Pipeline

STANDARD_SCALAR_PIPELINE = Pipeline([
    ('standard_scaler', RobustScaler()),
    ('min_max_scaler', MinMaxScaler()) 
])

def preprocess_data(pipeline, data, columns, train=False, return_scaler=False):
    
    if train == True:
        
        data[columns] = pipeline.fit_transform(data[columns])

        if return_scaler:
        
            return data, pipeline

        else:

            return data
        
    else:
        
        data[columns] = pipeline.transform(data[columns])
        return data