import os
os.environ['TF_USE_LEGACY_KERAS'] = 'True'
import ktrain
from ktrain import vision as vis
import re
import numpy as np

import pandas as pd
from make_model_prediction import make_model_prediction

from sklearn.model_selection import train_test_split

def build_default_cnn():
    model_name = 'default_cnn'
    operation = 'mean'
    
    bean_parameters = ["L", "a", "b"]
    learning_rates = [0.001, 0.002, 0.003, 0.005, 0.008, 0.013, 0.021, 0.1, 0.2, 0.3, 0.5]
    epochs = [30, 70]
    datasets = ["CD1", "CD2", "CD3"]

    #testings params
    # learning_rates = [0.001, 0.002]
    # epochs = [2]
    # datasets = ["CD1", "CD2", "CD3"]

    # remove previous results
    if os.path.exists("results/" + model_name + ".csv"):
        os.remove("results/" + model_name + ".csv")

    if os.path.exists("results/" + model_name + "_mae.csv"):
        os.remove("results/" + model_name + "_mae.csv")
    
    for bean_parameter in bean_parameters:
        for dataset in datasets:
            for learning_rate in learning_rates:
                for epoch in epochs:
                    DATADIR = "features/" + dataset + "/" + bean_parameter + '_' + operation + '/'
                    PATTERN = r'(\d+(\.\d+)?)\.jpg$'
                    p = re.compile(PATTERN)
                    
                    data_aug = vis.get_data_aug(horizontal_flip=True, vertical_flip=True)
                    (train_data, val_data, preproc) = vis.images_from_fname(DATADIR, pattern = PATTERN, data_aug = data_aug, val_pct=0.67, is_regression=True, random_state=42)

                    # # Convert the train_data DataFrameIterator to a DataFrame
                    # train_data_df = pd.DataFrame(data=train_data, columns=train_data)

                    # # Split the DataFrame into train and validation subsets
                    # train_data_subset, val_data_subset = train_test_split(train_data_df, train_size=0.67, test_size=0.33, random_state=42)

                    # # Convert back the train_data_subset to a DataFrameIterator object
                    # train_data_subset_iterator = DataFrameIterator(data=train_data_subset.values, columns=train_data_subset.columns)

                    model = vis.image_regression_model(model_name, train_data, val_data)

                    learner = ktrain.get_learner(model=model, train_data=train_data, val_data=val_data, workers=8, use_multiprocessing=False, batch_size=16)
                    learner.autofit(learning_rate, epoch, reduce_on_plateau=5)

                    predictor = ktrain.get_predictor(learner.model, preproc)
                    predictor.save("models/" + model_name + "/" + dataset + "/" + bean_parameter + "/" + str(epoch) + "_" + str(learning_rate))

                    make_model_prediction(predictor, bean_parameter, learning_rate, epoch, dataset, model_name)

                    print("Model built successfully.")

