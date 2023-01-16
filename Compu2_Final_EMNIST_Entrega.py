#!/usr/bin/env python
# coding: utf-8

# In[201]:


from emnist import list_datasets, extract_training_samples, extract_test_samples
import h5py
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPool2D, LeakyReLU, BatchNormalization
from tensorflow.keras import activations
import matplotlib.pyplot as plt
from pandas import DataFrame
import seaborn as sn
from sklearn.metrics import confusion_matrix
import numpy as np
from PIL import Image
import os
from sklearn.model_selection import KFold
from numpy import mean, std
from tensorflow.keras.preprocessing.image import load_img, img_to_array


# In[2]:


def load_and_preprocess_dataset():
    '''
    This function loads the EMNIST byclass dataset that contains 62 unbalanced classes and 814,255 characters.
    Assigns the corresponding training and test samples and then preprocess them to fit the requisites to pass them
    into a CNN by reshaping and normalizing them. 
    '''
    #Extract training and test samples from EMNIST byclass dataset and assigning to corresponding variables
    x_train, y_train = extract_training_samples('byclass')
    x_test, y_test = extract_test_samples('byclass')
    # Reshape the array to 4 dimensions adding a single color channel parameter for it to work with the Keras API.
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1) #x_train.shape[0] = 
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    # Convert values to float for division to normalize pixels 
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # Normalizing the RGB codes by dividing by the max RGB value to rescale our sampling sets between 0 and 1
    x_train /= 255
    x_test /= 255
    return x_train, x_test, y_train, y_test


# In[3]:


def define_model():
    '''
    Construct a CNN model with hyperparameters flexible to be tuned and modified. 
    '''
    filters = 32 #Number of feature maps
    kernel_size = (3,3) #Size of the convolution filter
    input_shape = (28, 28, 1) # Specific and requisite for the CNN
    units1 = 128 #Outputs of that layer
    units2 = 62 #Number of labels, since it is the output layer.
    act_function = tf.nn.relu #Relu necessary for convolutional layers, implemented for baseline model
    #act_function = LeakyReLU(alpha=0.2) # Non-zero gradient ReLU, candidate for improvement
    #act_function1 = tf.nn.relu #Maintain consistency in activation functions across hidden layers
    #act_function1 = LeakyReLU(alpha=0.2)
    act_function2 = tf.nn.softmax #Scaling 'arbitrary' numbers to mutually exclussive probabilities since it is a multiclass problem
    rate = 0.2 #20% of neurons dropped to avoid overfitting
    model = Sequential(
        [ 
            Conv2D(filters, kernel_size, input_shape=input_shape),
            BatchNormalization(),
            layers.Activation(activations.relu),
            MaxPool2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None),
            Conv2D(64, kernel_size, activation = act_function),
            Conv2D(64, kernel_size, activation = act_function),
            MaxPool2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None),
            Flatten(data_format=None),
            Dense(units1),
            BatchNormalization(),
            layers.Activation(activations.relu),
            Dropout(rate, noise_shape=None, seed=None),
            Dense(units2, activation=act_function2)
        ])
    optimizer1 = 'adam' #Hyperparameter that could change to non-adaptive optimizers such as SGD
    #Specific loss function used to reduce memory usage. The other option was categorical crossentropy, but it required for us to one-hot encoded
    #our target arrays, having many zeros in them since this problem has a big amount of classes. 
    loss1 = 'sparse_categorical_crossentropy'
    metrics1 = ['accuracy']
    model.compile(optimizer = optimizer1, 
                  loss = loss1, 
                  metrics = ['accuracy'])
    return model


# In[4]:


def pre_evaluation(data_X, data_Y, k_folds = 6):
    '''
    Pre-evaluating the model using 6-folds cross-validation (to divide exactly the 697,932 data)
    '''
    scores, histories = list(), list() #Initializing result lists
    # Call cross validation function, shuffling outside the loop to maintain consistency when testing k-fold and seeding
    kfold = KFold(k_folds, shuffle=True, random_state=1)
    # Initialize loop by alternating and evaluating on k-folds
    for train_ix, test_ix in kfold.split(data_X):
        # Call model
        model = define_model()
        # Slicing the data according to how the split of the k-fold was made
        train_X, train_Y, test_X, test_Y = data_X[train_ix], data_Y[train_ix], data_X[test_ix], data_Y[test_ix]
        # Fit model (will be discarded after this preevaluation, since the real fitting will be afterwards)
        #Hyperparameters of 10 epochs subject to change, batch size of 36 so each loop of each epoch has the same number of samples
        history = model.fit(train_X, train_Y, epochs=10, batch_size=36, validation_data=(test_X, test_Y), verbose=0)
        # Obtain loss and accuracy of the model
        loss, accuracy = model.evaluate(test_X, test_Y, verbose=0)
        print('> %.3f' % (accuracy * 100.0))
        # Store information on scores and histories arrays
        scores.append(accuracy)
        histories.append(history)
    return scores, histories


# In[5]:


def diagnosis(histories):
    '''
    Plot learning curves to evaluate how the model performed per fold of the 6-folds cross-validation
    '''
    for i in range(len(histories)):
        # Evaluate graphically the loss
        plt.subplot(2, 1, 1)
        plt.title('Sparse Cross Entropy Loss')
        plt.plot(histories[i].history['loss'], color='green', label='Train')
        plt.plot(histories[i].history['val_loss'], color='red', label='Test')
        # Evaluate graphically the accuracy
#         plt.subplot(2, 1, 2)
#         plt.title('Multi-Classification Accuracy')
#         plt.plot(histories[i].history['accuracy'], color='green', label='Train')
#         plt.plot(histories[i].history['val_accuracy'], color='red', label='Test')
    plt.show()


# In[6]:


def performance(scores):
    '''
    Quantitatively evaluate how the model performed by using mean, standard deviation and k-folds results.
    Plotting the accuracies to graphically evaluate the model's performance.
    '''
    # Quantitatively determine how well the model performed considering mean of accuracies, standard deviation and k-folds
    print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))
    # Graphically evaluate how well the model performed by using a boxplot to catch the max, min and average accuracies
    plt.boxplot(scores)
    plt.show()


# In[7]:


def run_pre_test():
    '''
    Run the whole pre-test by calling this function. 
    '''
    # Load and preprocess the dataset
    train_X, test_X, train_Y, test_Y = load_and_preprocess_dataset()
    # Evaluate model
    scores, histories = pre_evaluation(train_X, train_Y)
    # Plot performance of model learning
    diagnosis(histories)
    # Quantify how the model performed
    performance(scores)


# In[8]:


def save_model():
    '''
    Run the test with the hold out test set by calling this function. 
    '''
    # Load and preprocess the dataset
    train_X, test_X, train_Y, test_Y = load_and_preprocess_dataset()
    # Define model
    model = define_model()
    # Fit model
    model.fit(train_X, train_Y, epochs=10, batch_size=36, verbose=0)
    # save model
    model.save('final_EMNISTmodel.h5')


# In[84]:


save_model()


# In[174]:


train_X, test_X, train_Y, test_Y = load_and_preprocess_dataset()
len(train_Y)


# In[205]:


def confusion(y_test, y_pred):
    matrix = metrics.confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    return matrix


# In[9]:


def test():
    # Load and preprocess the dataset
    train_X, test_X, train_Y, test_Y = load_and_preprocess_dataset()
    # Load model
    model = tf.keras.models.load_model('final_EMNISTmodel.h5')
    # Evaluate model on test dataset
    loss, acc = model.evaluate(test_X, test_Y, verbose=0)
    y_pred = model.predict(test_X)
    confm = confusion_matrix(test_Y, y_pred.argmax(axis=1))
    df_cm = DataFrame(confm)
    #df_cm.to_csv(r'D:\Downloads\export_dataframe.csv', index = False, header=True)
    fig, ax = plt.subplots(figsize=(75,75))
    ax = sn.heatmap(df_cm, cmap='Oranges', annot=True)
    print('> %.3f' % (acc * 100.0))


# In[126]:


def load_image(filename):
    # Load the image
    img = load_img(filename, color_mode = "grayscale", target_size=(28, 28))
    # Convert to array
    img = img_to_array(img)
    # Reshape with 1 channel
    img = img.reshape(1, 28, 28, 1)
    # Preprocess array by converting type to float and normalizing
    img = img.astype('float32')
    img = 255-img
    img = img / 255
    return img


# In[208]:


def run_example():
    # Load the image
    img = load_image('char_y_minus.png')
    # Load model
    model = tf.keras.models.load_model('final_EMNISTmodel.h5')
    # Predict the class
    prediction = model.predict_classes(img)
    img_fig = np.resize(img, (28,28,1))
    plt.imshow(img_fig)
    d_num_to_real = {0:'0', 1:'1', 2: '2', 3: '3', 4:'4', 5:'5', 6:'6', 7:'7', 8: '8', 9: '9', 10: 'A', 11: 'B', 12: 'C',
                    13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N',
                    24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y',
                    35: 'Z', 36: 'a', 37: 'b', 38: 'c', 39: 'd', 40: 'e', 41: 'f', 42: 'g', 43: 'h', 44: 'i', 45: 'j',
                    46: 'k', 47: 'l', 48: 'm', 49: 'n', 50: 'o', 51: 'p', 52: 'q', 53: 'r', 54: 's', 55: 't', 56: 'u',
                    57: 'v', 58: 'w', 59: 'x', 60: 'y', 61: 'z'
                    }
    print(d_num_to_real[prediction[0]])
    #print(prediction[0])


# In[209]:


run_example()


# In[ ]:





# In[ ]:




