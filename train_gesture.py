# tutorial p1: https://towardsdatascience.com/human-activity-recognition-har-tutorial-with-keras-and-core-ml-part-1-8c05e365dfa0
# tutorial p2: https://towardsdatascience.com/human-activity-recognition-har-tutorial-with-keras-and-core-ml-part-2-857104583d94

# from __future__ import print_function
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
import os

verbose = False

N_FEATURES = 15
M_CLASSES = 2
LABELS = ['Undetected', 'ForceTouch ']

# ~~~~ SEGMENTATION PARAMETERS ~~~~
# The number of consecutive data entries within one segment
SEGMENT_LENGTH = 45

# The number of entries to advance forward from segment T to segment t+1
SEGMENT_INTERVAL = 15

# [                            SEGMENT T                            ]
#                        [                            SEGMENT T+1                          ]
# +-- SEGMENT_INTERVAL --+



# ~~~~ MODEL HYPER PARAMETERS ~~~~
BATCH_SIZE = 40
EPOCHS = 50


print('keras version ', keras.__version__)


def create_segments_and_labels(data):

    # x, y, z acceleration as features

    segments = []
    labels = []
    for i in range(0, data.shape[0] - SEGMENT_LENGTH, SEGMENT_INTERVAL):
        # sliding window of 45 rows, each excluding the last column
        x_segment = data[i:i + SEGMENT_LENGTH, :-1]

        # Retrieve the most often used label in this segment (mode of last column across 45 rows)
        y_label = stats.mode(data[i:i + SEGMENT_LENGTH, -1])[0][0]
        segments.append(x_segment)
        labels.append(y_label)

    # Bring the segments into a better shape, in our case:
    # (# of segments) x (# data entries per segment) x (# features per data entry)
    reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, SEGMENT_LENGTH, N_FEATURES)
    labels = np.asarray(labels)

    return reshaped_segments, labels

def ingest_annotated_data(file):
    data = np.genfromtxt('./data/annotated/{0}'.format(file), delimiter=",")

    rows, cols = data.shape
    test_data_size = int(rows / 5)
    print('\t\tDataset comprises of {0} rows, of which {1} are used for testing'.format(rows, test_data_size))

    # TODO: randomly sample data for train/test
    train = data[test_data_size:,:]
    test = data[:test_data_size, :]

    x_train, y_train = create_segments_and_labels(train)
    x_test, y_test = create_segments_and_labels(test)

    return x_train, x_test, y_train, y_test

def preprocess_annotated_data():

    usable_data = sys.argv[1:] if len(sys.argv) > 1 else [os.fsdecode(f) for f in os.listdir(os.fsencode('./data/annotated/'))]

    x_train = []
    x_test = []
    y_train = []
    y_test = []
    if verbose: print("ingesting and preprocessing...")
    for filename in usable_data:
        if verbose: print("\t... {0}".format(filename))
        batch_x_train, batch_x_test, batch_y_train, batch_y_test = ingest_annotated_data(filename)
        x_train.append(batch_x_train)
        x_test.append(batch_x_test)
        y_train.append(batch_y_train)
        y_test.append(batch_y_test)
        if verbose: print('\t\tSegmented {0} rows of training data and {1} rows of testing data'.format(batch_x_train.shape[0], batch_x_test.shape[0]))

    x_train = np.vstack(x_train)
    x_test = np.vstack(x_test)
    y_train = np.concatenate(y_train)
    y_test = np.concatenate(y_test)


    print('\nTraining/Testing Data Summary')
    positive_train_instances = sum(y_train)
    positive_test_instances = sum(y_train)
    print('\t{0} total training samples'.format(x_train.shape[0]))
    print('\t\tpositive examples', positive_train_instances)
    print('\t\tnegative examples', y_train.shape[0] - positive_train_instances)

    print('\t{0} total testing samples'.format(x_test.shape[0]))
    print('\t\tpositive examples', positive_test_instances)
    print('\t\tnegative examples', y_test.shape[0] - positive_test_instances, "\n")

    input_shape = (SEGMENT_LENGTH * N_FEATURES)
    x_train = x_train.reshape(x_train.shape[0], input_shape)
    x_test = x_test.reshape(x_test.shape[0], input_shape)
    y_train_hot = np_utils.to_categorical(y_train, M_CLASSES)
    y_test_hot = np_utils.to_categorical(y_test, M_CLASSES)
    if verbose:
        print('\tx_train reshaped:', x_train.shape)
        print('\tx_test reshaped:', x_test.shape)
        print('\tdata instance input shape:', input_shape)
        print('\tNew y_train one-hot vector shape for model fitting:', y_train_hot.shape)
        print('\tNew y_test one-hot vector shape: ', y_test_hot.shape)

    return x_train, x_test, y_train, y_train_hot, y_test_hot, input_shape


x_train, x_test, y_train, y_train_hot, y_test_hot, input_shape = preprocess_annotated_data()
model_m = Sequential()
# Remark: since coreml cannot accept vector shapes of complex shape like
# [80,3] this workaround is used in order to reshape the vector internally
# prior feeding it into the network
model_m.add(Reshape((SEGMENT_LENGTH, N_FEATURES), input_shape=(input_shape,)))
model_m.add(Dense(100, activation='relu'))
model_m.add(Dense(100, activation='relu'))
model_m.add(Dense(100, activation='relu'))
model_m.add(Flatten())
model_m.add(Dense(M_CLASSES, activation='softmax'))
if verbose: print(model_m.summary())


callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath='./model/checkpoints/best_model.{epoch:02d}-{val_loss:.2f}.h5',
        monitor='val_loss', save_best_only=True),
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=4)
]

model_m.compile(loss='categorical_crossentropy',
                optimizer='adam', metrics=['accuracy'])



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



coreml_model.author = 'Aayush Kumar'
coreml_model.license = 'N/A'
coreml_model.short_description = 'Levis Jacquard New Gesture: Force Touch Recognition'
coreml_model.output_description['output'] = 'Probability of each activity'
coreml_model.output_description['classLabel'] = 'Labels of activity'

print(coreml_model)

print("\n\n================================ SAVE AS: =================================")
print("====================== (Or Ctrl-D to not save/exit) =======================\n")
coreml_model.save('./model/coreml/NewGestureClassifier_{0}.mlmodel'.format(input()))

# print('\nPrediction from Keras:')

test_record = x_test[1].reshape(1,input_shape)
# keras_prediction = np.argmax(model_m.predict(test_record), axis=1)
# print(LABELS[keras_prediction[0]])
# print('\nPrediction from Coreml:')
# coreml_prediction = coreml_model.predict({'15ThreadConductivityReadings': test_record.reshape(input_shape)})
# print(coreml_prediction["classLabel"])
n = 500
print("Testing prediction speed on {0} samples".format(n))
start = time.time()
for i in range(n):
    coreml_model.predict({'15ThreadConductivityReadings': test_record.reshape(input_shape)})
end = time.time()
elapsed = end - start
print("Cumulative Prediction Time: {0:.4} sec, Average Prediction Time: {1:.4} sec".format(elapsed, elapsed / n))