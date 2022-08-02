root = '/gpfs/scratch/nsk367/mri-content-detection/dat/'
random_forest_save_path = '/gpfs/scratch/nsk367/mri-content-detection/outputs'


random_forest_param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [None],
    'criterion' :['gini', 'entropy']
}

