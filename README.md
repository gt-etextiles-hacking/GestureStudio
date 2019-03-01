# JacquardUtils

## Components

1. Utilities for converting, visualizing, and annotating raw hexstring Jacquard Data (length = 15)

2. Script for training a new gesture using annotated data

## Requirements
* python 3
* numpy
* scipy
* keras
* seaborn
* matplotlib 2.1.1 (version 3 seems to break the visualization, PRs are more than appreciated)!

## Docs

### convert_threads.py
*Script to convert raw hex readings of Jacquard Data (stored in ./data/raw/<name> as a csv) to numpy compatible csv (outputed into ./data/converted/<name>)- run with `python convert_threads.py <name>`*

Example: `python convert_threads.py forcetouch_data.csv`

### visualize_threads.py
*Script to visualize converted CSV Jacquard Data (stored in ./data/converted/<name> as a csv) as a function of time- run with `python visualize_threads.py forcetouch_data.csv`*

Example: `python visualize_threads.py forcetouch_data.csv`

Note: Inspiration for visualizing the time series thread data came from the infamous arcade game Dance Dance Revolution, or DDR. Just as timing is indicated by when a DDR step is about to scroll beyond the visible window, the timing of each thread reading is indicated by when it reaches the last visible row before disappearing offscreen.

### annotate_threads.py
*Script to visualize csv thread readings- run with `python annotate_threads.py forcetouch_data.csv`*

Example: `python annotate_threads.py forcetouch_data.csv`

Note: This script displays the same visualization as the aforementioned `visualize_threads.py` does, except with a title indicating whether the script is recording a Positive Annotation (recording 1's indicating that yes, this is the gesture we are trying to detect) or a Negative Annotation (recording 0's indicating that no, this is something we should NOT identify as our new gesture). Clicking anywhere on the plot will toggle between positive and negative annotations, and basically makes this like a game of DDR where you want to time your toggling clicks so that you get positive annotations for the duration of your gesture and then get negative annotations for anything else.

The script will initially prompt you with the following message:
> Often times the VERY first row of data is the beginning of the gesture you're trying to recognize. In that case, would you like to start annotating with 1's? (y/n)

This simply refers to how your dataset might start off with a positive or negative instance of the gesture you are annotating for. Typing `y` will start the annotations with positive labels and `n` will start the annotations with negative labels.

As a heads up, you need to click the visualization window an extra time in the VERY beginning to make the visualization your active window. Only with an active window will your clicks be registered a toggling between positive and negative annotations.


### train_gesture.py
*Script to train a model to recognize gestures based on the annotation(s) stored in  ./data/annoations/. While storing checkpoints of the model in ./model/checkpoints, it also converts/exports an iOS compatible coreML model into ./model/coreml*

Example 1: `python train_gesture.py` to use ALL annotated data in ./data/annotated/* to train the model

Example 2: `python train_gesture.py test.csv hex_data.csv` to use only specified files, in this case ./data/annotated/test.csv and ./data/annotated/hex_data.csv, to train the model
