from randomForestFile import random_forest_from_file, random_forest_to_file
import os

if not os.path.exists('random_forest.joblib'):
	random_forest_to_file(file_name='random_forest', n_estimators=200, n_jobs=10, verbose=1, random_state=678)
