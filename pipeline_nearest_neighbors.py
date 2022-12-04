import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv('dataset.csv')
features = data.drop(columns='target')
x_train, x_test, y_train, y_test = \
    train_test_split(features, data['target'], random_state=1)

# Score on training set for this model: 0.872782874617737
exported_pipeline = KNeighborsClassifier(6, weights='distance')

exported_pipeline.fit(x_train, y_train)
results = exported_pipeline.predict(x_test)