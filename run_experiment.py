import os
os.environ['DISABLE_V2_BEHAVIOR'] = '1'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="0" 
import ktrain
from ktrain import vision as vis
import numpy as np
import pandas as pd
import shutil
import re

DATASET_DIR = './datasets/CD3/images/'

def buildModel(bean_p, operation, data, PATTERN):

  DATADIR = bean_p+'_'+operation+'/'
  os.makedirs('./'+DATADIR)
  print('Estimating '+bean_p+' using '+operation)
  
  if operation == 'max':
    print('MAX')
    dts = data.groupby(['filename'], as_index=False ).max()
  if operation == 'min':
    print('MIN')
    dts = data.groupby(['filename'], as_index=False ).min()
  if operation == 'mean':
    print('MEAN')
    dts = data.groupby(['filename'], as_index=False ).mean()
  
  dts = dts[['filename', bean_p]]
  for i in range(dts.shape[0]):
    shutil.copy2(DATASET_DIR + str(dts.loc[i,'filename']) + '.jpg', './' + DATADIR+str(np.round(dts.loc[i,bean_p],2)) + '.jpg')

  data_aug = vis.get_data_aug(horizontal_flip=True, vertical_flip=True)
  (train_data, val_data, preproc) = vis.images_from_fname(DATADIR, pattern = PATTERN, data_aug = data_aug, val_pct=0.1, is_regression=True, random_state=42)  

  model = vis.image_regression_model('mobilenet', train_data, val_data)

  learner = ktrain.get_learner(model=model, train_data=train_data, val_data=val_data, workers=8, use_multiprocessing=False, batch_size=16)
  #learner.fit_onecycle(0.0001, 50)
  learner.autofit(0.05, 5, reduce_on_plateau=5)
  #learner.fit_onecycle(0.0001, 10)

  predictor = ktrain.get_predictor(learner.model, preproc)

  return predictor

def img_prediction(predictor, fname):
    print(fname)
    predicted = float(predictor.predict_filename(fname)[0])
    return predicted


if __name__ == "__main__":

  dt = pd.read_csv('./datasets/CD3/data.csv')
  parameters = ['L','a', 'b']
  operators = ['min', 'max']
  PATTERN = r'(\d+(\.\d+)?)\.jpg$'
  p = re.compile(PATTERN)

  #with zipfile.ZipFile("beans.zip","r") as zip_ref:
  #  zip_ref.extractall("./")

  shutil.rmtree('./a_min/', ignore_errors=True)
  shutil.rmtree('./L_min/', ignore_errors=True)
  shutil.rmtree('./L_mean/', ignore_errors=True)
  shutil.rmtree('./b_min/', ignore_errors=True)
  shutil.rmtree('./L_max/', ignore_errors=True)
  shutil.rmtree('./a_max/', ignore_errors=True)
  shutil.rmtree('./b_max/', ignore_errors=True)

  #parser = argparse.ArgumentParser()
  #parser.add_argument('exp', type=str)
  #parser.add_argument('method', type=str)
  #parser.add_argument('--it', type=int, default=10)
  #parser.add_argument('cal', type=str)
  
  #args = parser.parse_args()
  #for i in os.listdir('./trainedArtifacts/'):
  #  if not i.startswith('.'):
  #experimentRunner(dataset=args.exp, method_name=args.method, niterations=3, calibrated=args.cal)

  #model = buildModel('L', 'min', dt, PATTERN)
  #model.save("models/regressor_L_min")
  #model = buildModel('L', 'max', dt, PATTERN)
  #model.save("models/regressor_L_max")
  model = buildModel('L', 'mean', dt, PATTERN)
  model.save("models/regressor_L_mean")

  #model = buildModel('a', 'min', dt, PATTERN)
  #model.save("models/regressor_a_min")
  #model = buildModel('a', 'max', dt, PATTERN)
  #model.save("models/regressor_a_max")
  #model = buildModel('a', 'mean', dt, PATTERN)
  #model.save("models/regressor_a_mean")

  #model = buildModel('b', 'min', dt, PATTERN)
  #model.save("models/regressor_b_min")
  #model = buildModel('b', 'max', dt, PATTERN)
  #model.save("models/regressor_b_max")
  #model = buildModel('b', 'mean', dt, PATTERN)
  #model.save("models/regressor_b_mean")

  #learner = keras.models.load_model('models/regressor_L_min/tf_model.h5')
  #predictor = ktrain.load_predictor('models/regressor_L_min')
  #print(predictor.summary())
  #print(predictor.predict_filename('beans/e1.jpg')[0])
  #preproc = keras.models.load_model('models/regressor_L_min')
  #predictor = ktrain.get_predictor(learner)
  #print(img_prediction(predictor, 'beans/e1.jpg'))
  