import os
import shutil
import numpy as np
import pandas as pd

DATASET_DIR = './datasets/CD3/images/'

def build_features():
    bean_parameter = input("Choose a bean parameter (L, a or b): ")
    operation = input("Choose the operation (min, max or mean): ")
    dataset = input("Choose the dataset (1, 2 or 3): ")

    parameters = ['L', 'a', 'b']
    operators = ['min', 'max', 'mean']
    datasets = ['1', '2', '3']
    
    if bean_parameter not in parameters or operation not in operators or dataset not in datasets:
        print("Invalid input.")
        return

    data = pd.read_csv('./datasets/CD3/data.csv')
    DATADIR = "features/" + bean_parameter + '_' + operation + '/'
    directory = f'./{bean_parameter}_{operation}/'
    
    # Remove existing directory
    shutil.rmtree(directory, ignore_errors=True)
    
    if not os.path.exists('./' + DATADIR):
        os.makedirs('./'+ DATADIR)

    print('Estimating ' + bean_parameter + ' using ' + operation)
    
    if operation == 'max':
        dts = data.groupby(['filename'], as_index=False ).max()
    elif operation == 'min':
        dts = data.groupby(['filename'], as_index=False ).min()
    elif operation == 'mean':
        dts = data.groupby(['filename'], as_index=False ).mean()
    
    dts = dts[['filename', bean_parameter]]
    for i in range(dts.shape[0]):
        shutil.copy2(DATASET_DIR + str(dts.loc[i,'filename']) + '.jpg', './' + DATADIR+str(np.round(dts.loc[i,bean_parameter],2)) + '.jpg')
