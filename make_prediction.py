import os
import numpy as np
import pandas as pd
import ktrain

# def make_prediction(bean_parameter, operation):
def make_prediction(bean_parameter):
    DATADIR = "./datasets/CD3/images/"
    original_data = pd.read_csv('./datasets/CD3/data.csv')
    predictions = []
    
    #load predictor
    predictor = ktrain.load_predictor('./models/regressor_L_mean')
    model = ktrain.get_predictor(predictor.model, predictor.preproc)

    for filename in os.listdir(DATADIR):
        if filename.endswith(".jpg"):
            img_path = os.path.join(DATADIR, filename)
            print('predicting ' + img_path)
            prediction = model.predict_filename(img_path=img_path)[0]
            # Find original value corresponding to the filename
            original_value = original_data.loc[original_data['filename'] == filename.split('.')[0], bean_parameter].values[0]
            # Calculate difference between prediction and original value
            difference = abs(prediction - original_value)
            print('data: ' + str(original_value) + ' prediction: ' + str(prediction) + ' difference: ' + str(difference))
            predictions.append(difference)

    # Calculate average difference
    average_difference = np.mean(predictions)
    print(f"Average difference between predictions and original data: {average_difference}")
