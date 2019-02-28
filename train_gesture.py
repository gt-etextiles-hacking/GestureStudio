from __future__ import print_function
from matplotlib import pyplot as plt

import numpy as np
import seaborn as sns
import coremltools
from scipy import stats

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import preprocessing

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils

import sys
import time


verbose = False

N_FEATURES = 15
LABELS = ['Undetected', 'ForceTouch ']

print('keras version ', keras.__version__)

# The number of steps within one time segment
TIME_PERIODS = 45
# The steps to take from one segment to the next; if this value is equal to
# TIME_PERIODS, then there is no overlap between the segments
STEP_DISTANCE = 15

data = np.genfromtxt('./data/annotated/{0}'.format(sys.argv[1]), delimiter=",")

rows, cols = data.shape
test_data_size = int(rows / 5)
print('Dataset comprises of {0} rows, of which {1} are used for testing'.format(rows, test_data_size))

train = data[test_data_size:,:]
test = data[:test_data_size, :]


def create_segments_and_labels(data, time_steps, step):

    # x, y, z acceleration as features

    # Number of steps to advance in each iteration (for me, it should always
    # be equal to the time_steps in order to have no overlap between segments)
    # step = time_steps
    segments = []
    labels = []
    for i in range(0, data.shape[0] - time_steps, step):
        # xs = df['x-axis'].values[i: i + time_steps]
        # ys = df['y-axis'].values[i: i + time_steps]
        # zs = df['z-axis'].values[i:i + time_steps]
        segment = data[i:i + time_steps, :-1]

        # Retrieve the most often used label in this segment
        label = stats.mode(data[i:i + time_steps, -1])[0][0]
        segments.append(segment)
        labels.append(label)

    # Bring the segments into a better shape
    reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, time_steps, N_FEATURES)
    labels = np.asarray(labels)

    return reshaped_segments, labels

x_train, y_train = create_segments_and_labels(train, TIME_PERIODS, STEP_DISTANCE)
x_test, y_test = create_segments_and_labels(test, TIME_PERIODS, STEP_DISTANCE)

print('x_train shape: ', x_train.shape)
print(x_train.shape[0], 'training samples')
print('y_train shape: ', y_train.shape)

print('x_test shape: ', x_test.shape)
print(x_test.shape[0], 'testing samples')
print('y_test shape: ', y_test.shape)

num_time_periods, num_sensors = x_train.shape[1], x_train.shape[2]
num_classes = 2


input_shape = (num_time_periods*num_sensors)
x_train = x_train.reshape(x_train.shape[0], input_shape)
x_test = x_test.reshape(x_test.shape[0], input_shape)
print('x_train reshaped:', x_train.shape)
print('x_test reshaped:', x_test.shape)
print('input_shape:', input_shape)

y_train_hot = np_utils.to_categorical(y_train, num_classes)
y_test_hot = np_utils.to_categorical(y_test, num_classes)
print('New y_train shape for model fitting: ', y_train_hot.shape)
print('New y_test shape: ', y_test_hot.shape)

model_m = Sequential()
# Remark: since coreml cannot accept vector shapes of complex shape like
# [80,3] this workaround is used in order to reshape the vector internally
# prior feeding it into the network
model_m.add(Reshape((TIME_PERIODS, N_FEATURES), input_shape=(input_shape,)))
model_m.add(Dense(100, activation='relu'))
model_m.add(Dense(100, activation='relu'))
model_m.add(Dense(100, activation='relu'))
model_m.add(Flatten())
model_m.add(Dense(num_classes, activation='softmax'))
print(model_m.summary())


callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
        monitor='val_loss', save_best_only=True),
    keras.callbacks.EarlyStopping(monitor='acc', patience=1)
]

model_m.compile(loss='categorical_crossentropy',
                optimizer='adam', metrics=['accuracy'])

# Hyper-parameters
BATCH_SIZE = 400
EPOCHS = 50

# Enable validation to use ModelCheckpoint and EarlyStopping callbacks.
history = model_m.fit(x_train,
                      y_train_hot,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      callbacks=callbacks_list,
                      validation_split=0.2,
                      verbose=1)

if verbose:
    plt.figure(figsize=(6, 4))
    plt.plot(history.history['acc'], 'r', label='Accuracy of training data')
    plt.plot(history.history['val_acc'], 'b', label='Accuracy of validation data')
    plt.plot(history.history['loss'], 'r--', label='Loss of training data')
    plt.plot(history.history['val_loss'], 'b--', label='Loss of validation data')
    plt.title('Model Accuracy and Loss')
    plt.ylabel('Accuracy and Loss')
    plt.xlabel('Training Epoch')
    plt.ylim(0)
    plt.legend()
    plt.show()

# Print confusion matrix for training data
y_pred_train = model_m.predict(x_train)
# Take the class with the highest probability from the train predictions
max_y_pred_train = np.argmax(y_pred_train, axis=1)
print(classification_report(y_train, max_y_pred_train))


def show_confusion_matrix(validations, predictions):

    matrix = metrics.confusion_matrix(validations, predictions)
    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix,
                cmap='coolwarm',
                linecolor='white',
                linewidths=1,
                annot=True,
                fmt='d')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

# import pdb; pdb.set_trace()

y_pred_test = model_m.predict(x_test)
# Take the class with the highest probability from the test predictions
max_y_pred_test = np.argmax(y_pred_test, axis=1)
max_y_test = np.argmax(y_test_hot, axis=1)

if verbose:
    show_confusion_matrix(max_y_test, max_y_pred_test)

print(classification_report(max_y_test, max_y_pred_test))

coreml_model = coremltools.converters.keras.convert(model_m,
                                                    input_names=['15ThreadConductivityReadings'],
                                                    output_names=['output'],
                                                    class_labels=LABELS)



print(coreml_model)
coreml_model.author = 'Aayush Kumar'
coreml_model.license = 'N/A'
coreml_model.short_description = 'Levis Jacquard New Gesture: Force Touch Recognition'
coreml_model.output_description['output'] = 'Probability of each activity'
coreml_model.output_description['classLabel'] = 'Labels of activity'

coreml_model.save('NewGestureClassifier.mlmodel')

# print('\nPrediction from Keras:')

test_record = x_test[1].reshape(1,input_shape)
# keras_prediction = np.argmax(model_m.predict(test_record), axis=1)
# print(LABELS[keras_prediction[0]])
# print('\nPrediction from Coreml:')
# coreml_prediction = coreml_model.predict({'15ThreadConductivityReadings': test_record.reshape(input_shape)})
# print(coreml_prediction["classLabel"])

start = time.time()
n = 100
for i in range(n):
    coreml_model.predict({'15ThreadConductivityReadings': test_record.reshape(input_shape)})
end = time.time()
elapsed = end - start
print(elapsed, elapsed / n)