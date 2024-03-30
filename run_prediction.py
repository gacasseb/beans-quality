import os
os.environ['DISABLE_V2_BEHAVIOR'] = '1'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import ktrain
from ktrain import vision as vis
import numpy as np
import pandas as pd
import re

model = vis.image_regression_model('pretrained_resnet50', train_data, val_data)

learner = ktrain.get_learner(model=model, train_data=train_data, val_data=val_data,
                             workers=8, use_multiprocessing=False, batch_size=16)

# get a Predictor instance that wraps model and Preprocessor object
predictor = ktrain.get_predictor(learner.model, preproc)

# how to get validation filepaths
val_data.filenames

def img_prediction(predictor, fname):
    fname = DATADIR+'/'+fname
    predicted = float(predictor.predict_filename(fname)[0])
    actual = float(p.search(fname).grou  p(1))
    #vis.show_image(fname)
    #print('predicted:%s | actual: %s' % (predicted, actual))
    return [predicted, actual]

re = []
k = 0
for i in val_data.filenames:
  pred, act = img_prediction(predictor, i)
  err = np.abs(pred - act)
  print(k, "-" ,i, err, "-", pred)
  k = k + 1
  re.append(err)

np.mean(re)

def show_prediction(predictor, fname):
    print(fname)
    predicted = float(predictor.predict_filename(fname)[0])
    actual = float(p.search(fname).group(1))
    vis.show_image(fname)
    print('predicted:%s | actual: %s' % (predicted, actual))

show_prediction(predictor, DATADIR +'/'+ val_data.filenames[8])

#dtP['a'] = np.round(dtP['a'],2)
dtP[dtP['a']==24.01]

predictor.explain( DATADIR +'/'+ val_data.filenames[8])
