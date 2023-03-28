import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score

# Cargamos el dataset
df = pd.read_csv('dataset_malaria.csv')

# Visualizamos las primeras 5 filas

df.head()

# Visualizamos las dimensiones del dataset

df.shape

# Visualizamos la cantidad de datos nulos
df.isnull().sum()

# Creamos una nueva columna con el nombre de la clase

