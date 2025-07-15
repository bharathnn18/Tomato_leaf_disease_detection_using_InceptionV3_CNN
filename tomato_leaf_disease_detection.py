from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# import the libraries as shown below
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
from glob import glob

# re-size all the images to this
IMAGE_SIZE = [224, 224]
train_path = '/content/drive/MyDrive/database/training'
valid_path = '/content/drive/MyDrive/database/validation'

# Here we will be using imagenet weights
inception = InceptionV3(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
for layer in inception.layers[:-70]:
    layer.trainable = False
for layer in inception.layers[-70:]:
    layer.trainable = True

 # Get the number of classes by counting subdirectories within the training path
import os
num_classes = len(os.listdir('/content/drive/MyDrive/database/training'))
num_classes

x = GlobalAveragePooling2D()(inception.output)

from tensorflow.keras.layers import Dropout
x = GlobalAveragePooling2D()(inception.output)
x = Dropout(0.5)(x)  # 50% dropout
prediction = Dense(num_classes, activation='softmax')(x)

# Update the Dense layer to have the correct number of output units
prediction = Dense(num_classes, activation='softmax')(x)
# create a model object
model = Model(inputs=inception.input, outputs=prediction)
# view the structure of the model
model.summary()

# tell the model what cost and optimization method to use
model.compile(
  loss='categorical_crossentropy',
  optimizer=Adam(learning_rate=1e-5),
  metrics=['accuracy']
)

# Use the Image Data Generator to import the images from the dataset
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.8,1.2]
)
test_datagen = ImageDataGenerator(rescale = 1./255)

# Make sure you provide the same target size as initialied for the image size
training_set = train_datagen.flow_from_directory('/content/drive/MyDrive/database/training',
                                                 target_size = (224, 224),
                                                 batch_size = 16,
                                                 class_mode = 'categorical')
test_set = test_datagen.flow_from_directory('/content/drive/MyDrive/database/validation',
                                            target_size = (224, 224),
                                            batch_size = 16,
                                            class_mode = 'categorical',
                                            shuffle=False)

# Stop training if val_loss doesn't improve after 3 epochs
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Save the best model based on validation loss
checkpoint = ModelCheckpoint('best_model_inception.h5', monitor='val_loss', save_best_only=True)
callbacks = [early_stop, checkpoint]

# fit the model
# Run the cell. It will take some time to execute
r = model.fit(
  training_set,
  validation_data=test_set,
  epochs=25,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set),
  callbacks=callbacks
)

# plot the loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.savefig('LossVal_loss.png')
plt.show()

# plot the accuracy
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.savefig('AccVal_acc')
plt.show()

# save it as a h5 file
model.save('model_inception.h5')

y_pred = model.predict(test_set)
y_pred
import numpy as np
y_pred = np.argmax(y_pred, axis=1)

#predict the label of an random image
img=image.load_img('/content/drive/MyDrive/database/validation/Leaf_Mold/02a29ab9-8cba-47a0-bc2f-e7af7dbae149___Crnl_L.Mold 7165.JPG',target_size=(224,224)
x=image.img_to_array(img)
x.shape
x=np.expand_dims(x,axis=0)
img_data=preprocess_input(x)
img_data.shape
model.predict(img_data)
a=np.argmax(model.predict(img_data), axis=1)
print("Predicted Class Index:", a)

class_indices = training_set.class_indices
labels = dict((v,k) for k,v in class_indices.items())
print("Predicted Label:", labels[a[0]])
# Predict the labels for validation data
Y_pred = model.predict(test_set, verbose=1)
y_pred = np.argmax(Y_pred, axis=1)
# True labels
y_true = test_set.classes
# Build confusion matrix
cm = confusion_matrix(y_true, y_pred)
# Plot it nicely
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_set.class_indices.keys(), yticklabels=test_set.class_indices.keys())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
# Classification report
report = classification_report(y_true, y_pred, target_names=list(test_set.class_indices.keys()))
print('Classification Report:\n', report)