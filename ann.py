import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.models import model_from_json
from keras.layers import Dense


def save_model():
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")

    joblib.dump(scaler, 'scaler.save')

    print("Saved model to disk")

def load_model():
    global model
    global scaler

    with open('model.json', 'r') as json_file:
        loaded_model_json = json_file.read()
    
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("model.h5")

    scaler = joblib.load('scaler.save')
    
    print("Loaded model from disk")

def predict(m0, m1, m2):
    data = pd.DataFrame([[m0, m1, m2, 0]])
    scaled = scaler.transform(data)
    # print(scaled)
    # print(type(scaled))
    x=pd.DataFrame([list(scaled[0][:-1])])
    # print(x)
    return model.predict(x)

try:
    load_model()
except:
    fields = ['0', '1', '2', 'target']

    raw = pd.read_csv('processed.csv', names=fields, sep=',', skipinitialspace=True)
    dataset = raw.copy()

    target = ('target')

    print(dataset)

    # train_dataset = dataset.sample(frac=0.8, random_state=0)
    # test_dataset = dataset.drop(train_dataset.index)

    # print(train_dataset)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_train = scaler.fit_transform(dataset)
    scaled_train_df = pd.DataFrame(scaled_train, columns=dataset.columns.values)

    model = Sequential()

    model.add(Dense(10, activation='softmax'))
    model.add(Dense(50, activation='softmax'))
    model.add(Dense(10, activation='softmax'))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')

    X = scaled_train_df.drop(target, axis=1).values
    Y = scaled_train_df[target].values

    # Train the model
    model.fit(
        X[10:],
        Y[10:],
        epochs=500,
        shuffle=True,
        verbose=2
    )

    save_model()

prediction = predict(0,0,0)

for p in prediction:
    p[0] = (p[0]-scaler.min_[3])/scaler.scale_[3]
    print('%.2f'%(p[0]))