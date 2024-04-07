import os
os.environ['DISABLE_V2_BEHAVIOR'] = '1'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0" 
import ktrain
from ktrain import vision as vis
import numpy as np
import pandas as pd
import shutil
import re
import sys
from tensorflow import keras

DATASET_DIR = './datasets/CD3/images/'

def build_model(model_name, bean_p, operation):
  
  DATADIR = "features/" + bean_p + '_' + operation + '/'
  PATTERN = r'(\d+(\.\d+)?)\.jpg$'
  p = re.compile(PATTERN)
  
  data_aug = vis.get_data_aug(horizontal_flip=True, vertical_flip=True)
  (train_data, val_data, preproc) = vis.images_from_fname(DATADIR, pattern = PATTERN, data_aug = data_aug, val_pct=0.5, is_regression=True, random_state=42)  

  model = vis.image_regression_model(model_name, train_data, val_data)

  learner = ktrain.get_learner(model=model, train_data=train_data, val_data=val_data, workers=8, use_multiprocessing=False, batch_size=16)
  #learner.fit_onecycle(0.0001, 50)
  learner.autofit(0.05, 5, reduce_on_plateau=5)
  #learner.fit_onecycle(0.0001, 10)

  predictor = ktrain.get_predictor(learner.model, preproc)
  
  prediction = predictor.predict_filename(img_path='C:/Users/ogabr/OneDrive/Documentos/TCC/datasets/CD3/images/a1.jpg')
  print(prediction)

  predictor.save("models/regressor_" + bean_p + "_" + operation)
  return predictor

def img_prediction(predictor, fname):
  print(fname)
  predicted = float(predictor.predict_filename(fname)[0])
  return predicted

def print_models():
  vis.print_image_regression_models()
  
def build_features(bean_parameter, operation, dataset):
  parameters = ['L','a', 'b']
  operators = ['min', 'max', 'mean']
  datasets = ['1', '2', 3]
  
  if bean_parameter not in parameters:
    print('Choose a valid bean parameter')
    
  if operation not in operators:
    print('Choose a valid operator')
    
  if dataset not in datasets:
    print('Choose a valid dataset')

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
  if operation == 'min':
    dts = data.groupby(['filename'], as_index=False ).min()
  if operation == 'mean':
    dts = data.groupby(['filename'], as_index=False ).mean()
  
  dts = dts[['filename', bean_parameter]]
  for i in range(dts.shape[0]):
    shutil.copy2(DATASET_DIR + str(dts.loc[i,'filename']) + '.jpg', './' + DATADIR+str(np.round(dts.loc[i,bean_parameter],2)) + '.jpg')
  
def make_prediction():
  print('making prediction')
  
def main():
  while True:
    print("\nMenu:")
    print("1 - Build model")
    print("2 - Make prediction")
    print("3 - Build features")
    print("4 - Exit")

    choice = input("Enter your choice: ")

    if choice == "1":
      model_name = input("Enter model name: ")
      bean_parameter = input("Choose a bean parameter (L, a or b): ")
      operation = input("Choose the operation (min, max or mean): ")
      build_model(model_name, bean_parameter, operation)
    elif choice == "2":
      make_prediction()
    elif choice == "3":
      bean_parameter = input("Choose a bean parameter (L, a or b): ")
      operation = input("Choose the operation (min, max or mean): ")
      dataset = input("Choose the dataset (1, 2 or 3): ")
      build_features(bean_parameter, operation, dataset)
    elif choice == "4":
      print("Exiting the program.")
      break
    else:
      print("Invalid choice. Please choose again.")


if __name__ == "__main__":
  main()
  