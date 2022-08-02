import numpy as np
import pandas as pd
from .config import random_forest_param_grid, random_forest_save_path
import pickle as cPickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


class RandomForestTrainer():
    """
    Trainer used to predict and then test the Random Forest Classifiers developed in this study for predicting an MRI's contrast and orientation
    based on the DICOM metadata associated.

    Note from Authors: To obtain partial access to the dataset used, please reach out to me at nsk367@nyu.edu
    """
    def __init__(self,
                param_grid = random_forest_param_grid,
                save_path = random_forest_save_path):
    
        self.param_grid = param_grid
        self.save_path = save_path

    def train_classifiers(self,X,y_contrasts,y_orientations):
        self.train_contrast_classifier(X,y_contrasts)
        self.train_orientation_classifier(X,y_orientations)
        return True

    def train_contrast_classifier(self,X,y):
        rfc = RandomForestClassifier(random_state=42)
        rf_contrast = GridSearchCV(estimator=rfc, param_grid=self.param_grid, cv= 5)
        rf_contrast.fit(X, y)

        # save best model
        with open(f'{random_forest_save_path}/contrast_random_forest.pkl', 'wb') as f:
            cPickle.dump(rf_contrast.best_estimator_, f)

    def train_orientation_classifier(self,X,y):
        rfc = RandomForestClassifier(random_state=42)
        rf_orientation = GridSearchCV(estimator=rfc, param_grid=self.param_grid, cv= 5)
        rf_orientation.fit(X, y)
        # save best model
        with open(f'{random_forest_save_path}/orientation_random_forest.pkl', 'wb') as f:
            cPickle.dump(rf_orientation.best_estimator_, f)




if __name__ == '__main__':
    model = RandomForestTrainer(param_grid=random_forest_param_grid, save_path=random_forest_save_path)
    model.train_classifiers(X,y_contrasts,y_orientations) # for this data, please reach out to nsk367@nyu.edu
    print('done')

