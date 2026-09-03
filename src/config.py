import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FOLDER = os.path.join(PROJECT_ROOT, 'data')

Random_State = 42
REPLACE_NAN = 0
EMBEDDING_METHODS = ['tsne', 'pca']
