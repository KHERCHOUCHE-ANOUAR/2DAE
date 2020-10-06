import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from joblib import dump, load




npzfile = np.load('data_training.npz')

svclassifier = SVC(kernel='sigmoid', gamma='auto')
svclassifier.fit(npzfile['X'], npzfile['Y'])
dump(svclassifier, 'svm_sigmoid_training.joblib')
