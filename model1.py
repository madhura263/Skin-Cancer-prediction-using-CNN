# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 18:32:56 2020

@author: Madhura
"""

#Skin cancer prediction using CNN 
#%%
#Importing the required libraries
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D , Flatten , Dense, InputLayer, BatchNormalization, Dropout
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2
from matplotlib import pyplot


#%%
#Loading the dataset using Keras ImageDataGenerator and splitting it into train and test set by 20%
#Adding data augmentation for avoiding overfitting
imagegen = ImageDataGenerator(validation_split = 0.2, rotation_range=10, shear_range=0.5, horizontal_flip = True)
train = imagegen.flow_from_directory('D:/Academic/Main/TY/Sem-5/DLFL/IA1/training/', subset = 'training', target_size=(128,128))
val = imagegen.flow_from_directory('D:/Academic/Main/TY/Sem-5/DLFL/IA1/training/', subset = 'validation', target_size=(128,128))

#%%
#CNN Model Building 
model = Sequential()

#Input layer
model.add(InputLayer(input_shape = (128 , 128, 3)))
model.add(BatchNormalization())

#1st Conv block
model.add(Conv2D(25 , (5,5), activation = 'relu', strides = (1,1), padding = 'same',  kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
model.add(Conv2D(25 , (5,5), activation = 'relu', strides = (1,1), padding = 'same',  kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
model.add(MaxPool2D(pool_size = (2,2), padding ='same'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

#2nd Conv block
model.add(Conv2D(50 , (5,5), activation = 'relu', strides = (2,2), padding = 'same',  kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
model.add(Conv2D(50 , (5,5), activation = 'relu', strides = (2,2), padding = 'same',  kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
model.add(MaxPool2D(pool_size = (2,2), padding ='same'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

#3rd Conv block
model.add(Conv2D(70 , (3,3), activation = 'relu', strides = (2,2), padding = 'same',  kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
model.add(Conv2D(70 , (3,3), activation = 'relu', strides = (2,2), padding = 'same',  kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
model.add(MaxPool2D(pool_size = (2,2), padding ='same'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

#dense block
model.add(Flatten())
model.add(Dense(units=256, activation = 'relu',  kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))

total_cat =  2  #Enter the total number of categories here

#Output layer
model.add(Dense(units = total_cat , activation = 'softmax'))

#Compiling the model
model.compile(loss = 'binary_crossentropy', optimizer = 'adam' , metrics =['accuracy'])

#Fitting the model on the data
history=model.fit_generator(train, epochs=100 ,validation_data = val)
#%%
#Visualization of the model performance
def summarize_model(model, history,train , val):
	# evaluate the model
	_, train_acc = model.evaluate(train, verbose=1)
	_, test_acc = model.evaluate(val, verbose=1)
	print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
	# plot loss during training
	pyplot.subplot(211)
	pyplot.title('Loss')
	pyplot.plot(history.history['loss'], label='train')
	pyplot.plot(history.history['val_loss'], label='test')
	pyplot.legend()
	# plot accuracy during training
	pyplot.subplot(212)
	pyplot.title('Accuracy')
	pyplot.plot(history.history['accuracy'], label='train')
	pyplot.plot(history.history['val_accuracy'], label='test')
	pyplot.legend()
	pyplot.show()
    
    
#%%
#Evaluate model behavior
summarize_model(model, history,train , val)

#%%
#Making predictions for test set
pred=model.predict_generator(val)

#%%
#Getting the true classes for the test set
true_classes = val.classes
class_labels = list(val.class_indices.keys())  
#%%
#Getting the class value for the made predictions
import numpy
predicted_classes = numpy.argmax(pred, axis=1) 
#%%
#Analysis of the model using precision and recall
import sklearn.metrics as metrics
report = metrics.classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report) 

#%%
#Building the confusion matrix
confusion_matrix = metrics.confusion_matrix(y_true=true_classes, y_pred=predicted_classes)

#%%
#Visualizing the confusion matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

array = confusion_matrix

df_cm = pd.DataFrame(array, range(2), range(2))
# plt.figure(figsize=(10,7))
sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size

plt.show()  
#%%
#Saving the model
model.save('model1.h5')