import os
os.environ['TF_USE_LEGACY_KERAS'] = 'True'
import ktrain
from ktrain import vision as vis
import re

def build_model():
    model_name = input("Enter model name: ")
    operation = input("Choose the operation (min, max or mean): ")
    
    bean_parameters = ["L", "a", "b"]
    
    for bean_parameter in bean_parameters:
        DATADIR = "features/" + bean_parameter + '_' + operation + '/'
        PATTERN = r'(\d+(\.\d+)?)\.jpg$'
        p = re.compile(PATTERN)
        
        data_aug = vis.get_data_aug(horizontal_flip=True, vertical_flip=True)
        (train_data, val_data, preproc) = vis.images_from_fname(DATADIR, pattern = PATTERN, data_aug = data_aug, val_pct=0.5, is_regression=True, random_state=42)  

        model = vis.image_regression_model(model_name, train_data, val_data)

        learning_rate = 0.04
        epochs = 70

        learner = ktrain.get_learner(model=model, train_data=train_data, val_data=val_data, workers=8, use_multiprocessing=False, batch_size=16)
        learner.autofit(learning_rate, epochs, reduce_on_plateau=5)

        predictor = ktrain.get_predictor(learner.model, preproc)
        predictor.save("models/" + model_name + "_" + str(learning_rate) + "_" + str(epochs) + "/regressor_" + bean_parameter + "_" + operation)

        print("Model built successfully.")
        
    # prediction = predictor.predict_filename(img_path='C:/Users/ogabr/OneDrive/Documentos/TCC/datasets/CD3/images/a1.jpg')
    # print(prediction)
