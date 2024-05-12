import os
import numpy as np
import pandas as pd
import ktrain
import csv

from scipy.stats import rankdata
from scipy.stats import norm

def save_to_csv(data, filename):
    df = pd.DataFrame(data, columns=['id', 'attribute', 'original', 'prediction', 'difference'])
    df.to_csv(filename, index=False)

def make_model_prediction(predictor, bean_parameter, learning_rate, epoch, dataset, model_name):
    data = []
    columns = ['id', 'dataset', 'bean_parameter', 'epoch', 'learning_rate', 'original', 'prediction', 'difference']
    mae_columns = ['dataset', 'bean_parameter', 'epoch', 'learning_rate', 'difference']
    
    DATADIR = "./datasets/" + dataset + "/images/"
    original_data = pd.read_csv('./datasets/' + dataset + '/data.csv')

    model = ktrain.get_predictor(predictor.model, predictor.preproc)

    predictions = []
    for filename in os.listdir(DATADIR):
        print('filename', filename)
        if filename.endswith(".jpg"):
            img_path = os.path.join(DATADIR, filename)
            print('predicting ' + img_path)
            prediction = model.predict_filename(img_path=img_path)[0]
            # Find original value corresponding to the filename
            values = original_data.loc[original_data['filename'] == filename.split('.')[0], bean_parameter].values
            print('values', values, filename.split('.')[0])

            original_value = calculate_operation(values, "mean")
            # Calculate difference between prediction and original value
            difference = abs(prediction - original_value)
            print('data: ' + str(original_value) + ' prediction: ' + str(prediction) + ' difference: ' + str(difference))

            predictions.append(difference)
            data.append([filename.split('.')[0], dataset, bean_parameter, epoch, learning_rate, original_value, prediction, difference])

    # Calculate average difference
    average_difference = np.mean(predictions)
    print(bean_parameter + f": {average_difference}")

    file_path = 'results/' + model_name + '.csv'
    file_mae_path = 'results/' + model_name + '_mae.csv'

    if not os.path.isfile(file_path):
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(columns)

    if not os.path.isfile(file_mae_path):
        with open(file_mae_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(mae_columns)

    with open(file_mae_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([str(dataset), str(bean_parameter), str(epoch), str(learning_rate), str(average_difference)])

    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)
        
def calculate_CD(k, N, alpha=0.05):
    q = norm.ppf(1 - alpha / 2)
    return q * np.sqrt(k * (k + 1) / (6 * N))
        
def calculate_friedman(all_predictions):
    ranks = np.array([rankdata(-np.array(p)) for p in all_predictions])
    N = len(all_predictions[0])
    k = len(all_predictions)
    CD = calculate_CD(k, N)
    mean_ranks = np.mean(ranks, axis=0)
    pairwise_diff = np.subtract.outer(mean_ranks, mean_ranks)
    pairwise_diff = np.abs(pairwise_diff)
    
    return CD, pairwise_diff
        
        
def calculate_operation(data, operation):
    if operation == 'mean':
        return sum(data) / len(data)
    elif operation == 'max':
        return max(data)
    elif operation == 'min':
        return min(data)
    else:
        return None
