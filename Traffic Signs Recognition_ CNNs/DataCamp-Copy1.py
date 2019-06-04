import tensorflow as tf

config=tf.ConfigProto(log_device_placement=True)

config=tf.ConfigProto(allow_soft_placement=True)

import os
import skimage
from skimage import transform 
from skimage import data

def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory) 
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f) 
                      for f in os.listdir(label_directory) 
                      if f.endswith(".ppm")]
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    return images, labels

        
ROOT_PATH = "E:/Deep Learning Course/Panneaux de signalisation"
train_data_directory = os.path.join(ROOT_PATH, "TrafficSigns/Training")
test_data_directory = os.path.join(ROOT_PATH, "TrafficSigns/Testing")

images, labels = load_data(train_data_directory)

# Import the `pyplot` module
import matplotlib.pyplot as plt 
from __future__ import division
from scipy import *
from pylab import *

# Make a histogram with 62 bins of the `labels` data
plt.hist(labels, 62)
plt.title("Distribution of Traffic Sign Labels")
plt.plot()

# Show the plot
plt.show()

#####################################################################################3

# Import the `pyplot` module of `matplotlib`
import matplotlib.pyplot as plt

# Determine the (random) indexes of the images that you want to see 
traffic_signs = [230, 879, 300, 1200]

# Fill out the subplots with the random images that you defined 
for i in range(len(traffic_signs)):
    plt.subplot(1, 7, i+1)
    plt.axis('off')
    plt.imshow(images[traffic_signs[i]])
    plt.subplots_adjust(wspace=0.5)

plt.show()

#######################################################################################33

# Import `matplotlib`
import matplotlib.pyplot as plt

# Determine the (random) indexes of the images
traffic_signs = [300, 2250, 3650, 4574]

# Fill out the subplots with the random images and add shape, min and max values
for i in range(len(traffic_signs)):
    plt.subplot(1, 4, i+1)
    plt.axis('off')
    plt.imshow(images[traffic_signs[i]])
    plt.subplots_adjust(wspace=0.5)
    plt.show()
    print("shape: {0}, min: {1}, max: {2}".format(images[traffic_signs[i]].shape, 
                                                  images[traffic_signs[i]].min(), 
                                                  images[traffic_signs[i]].max()))

# Get the unique labels 
unique_labels = set(labels)

# Initialize the figure
plt.figure(figsize=(20, 20))

# Set a counter
i = 1

# For each unique label,
for label in unique_labels:
    # You pick the first image for each label
    image = images[labels.index(label)]
    # Define 64 subplots 
    plt.subplot(8, 8, i)
    # Don't include axes
    plt.axis('off')
    # Add a title to each subplot 
    plt.title("Label {0} ({1})".format(label, labels.count(label)))
    # Add 1 to the counter
    i += 1  
    # And you plot this first image 
    plt.imshow(image)
    
# Show the plot
plt.show()

##########################################################################################33
import numpy as np
# Import the `transform` module from `skimage`
# Rescale the images in the `images` array
images32 = [skimage.transform.resize(image, (32, 32)) for image in images]

# Import `rgb2gray` from `skimage.color`
from skimage.color import rgb2gray
# Convert `images32` to an array
images32 = np.array(images32)
# Convert `images28` to grayscale
images32 = rgb2gray(images32)   
 
###############################
# Determine the (random) indexes of the images
traffic_signs = [300, 2250, 3650, 4000]

# Fill out the subplots with the random images and add shape, min and max values
for i in range(len(traffic_signs)):
    plt.subplot(1, 4, i+1)
    plt.axis('off')
    plt.imshow(images32[traffic_signs[i]], cmap="gray")
    plt.subplots_adjust(wspace=0.5)
plt.show()
#######################################################################33

# DEEP LEARNING WITH TENSORFLOW

#Modelling the Neural Network

""" First, you define placeholders for inputs and labels because you won’t put in the “real” data yet. Remember that placeholders are values that are unassigned and that will be initialized by the session when you run it. So when you finally run the session, these placeholders will get the values of your dataset that you pass in the run() function! """

import tensorflow as tf

# Initialize placeholders 
x = tf.placeholder(dtype = tf.float32, shape = [None, 32, 32])
y = tf.placeholder(dtype = tf.int32, shape = [None])

# Flatten the input data
images_flat = tf.contrib.layers.flatten(x)

# Fully connected layer 
# Generates logits of size [None, 62]
logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)


# Define a loss function
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, 
                                                                    logits = logits))
# Define an optimizer 
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

# Convert logits to label indexes (probabilities)
correct_pred = tf.argmax(logits, 1)

# Define an accuracy metric
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

print("images_flat: ", images_flat)
print("logits: ", logits)
print("loss: ", loss)
print("predicted_labels: ", correct_pred)

#Running the neural network

tf.set_random_seed(1234)
sess = tf.Session()

sess.run(tf.global_variables_initializer())

#Feed data to the model

for i in range(201):
    print("EPOCH", i)
    _, loss_val = sess.run([train_op, loss], feed_dict={x: images32, y: labels})
    _, accuracy_val = sess.run([train_op, accuracy], feed_dict={x: images32, y: labels})

    if i % 10 == 0:
        print("Loss: ", loss_val)
        print("Accuracy: ", accuracy_val)

######Evaluating Your Neural Network###############################33
    
# Import `matplotlib`
import matplotlib.pyplot as plt
import random

# Pick 10 random images
sample_indexes = random.sample(range(len(images)), 10)
sample_images = [images32[i] for i in sample_indexes]
sample_labels = [labels[i] for i in sample_indexes]

# Run the "correct_pred" operation
predicted = sess.run([correct_pred], feed_dict={x: sample_images})[0]
                        
# Print the real and predicted labels
print(sample_labels)
print(predicted)

# Display the predictions and the ground truth visually.
fig = plt.figure(figsize=(10, 10))
for i in range(len(sample_images)):
    truth = sample_labels[i]
    prediction = predicted[i]
    plt.subplot(5, 2,1+i)
    plt.axis('off')
    color='blue' if truth == prediction else 'black'
    plt.text(30, 20, "Truth:        {0}\nPrediction: {1}".format(truth, prediction), 
             fontsize=20, color=color)
    plt.imshow(sample_images[i],  cmap="gray")

plt.show()

#############################Load in the test data######################
import numpy as np
# Import `skimage`
# Load the test data
test_images, test_labels = load_data(test_data_directory)

# Transform the images to 32 by 32 pixels
test_images32 = [skimage.transform.resize(image, 32, 32) for image in test_images]

# Convert to grayscale
from skimage.color import rgb2gray

test_images32 = rgb2gray(np.array(test_images32))

# Run predictions against the full test set.
predicted = sess.run([correct_pred], feed_dict={x: test_images32})[0]

# Calculate correct matches 
match_count = sum([int(y == y_) for y, y_ in zip(test_labels, predicted)])

# Calculate the accuracy
accuracy = match_count / len(test_labels)

print("Accuracy: {:.3f}".format(accuracy))
#########################################################################################
