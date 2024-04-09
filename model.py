import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO messages and above

# The rest of your script goes here

import numpy as np
import cv2
import pandas as pd
from tqdm.notebook import tqdm 
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import os


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# **********************
# original_fake_paths = []

# for dirname, _, filenames in tqdm(os.walk('./1-million-fake-faces')):
#     for filename in filenames:
#         original_fake_paths.append([os.path.join(dirname, filename), filename])

# Define the directory to walk
directory = './1-million-fake-faces'

# List to store paths and filenames
original_fake_paths = []

# Walk through the directory
for dirname, _, filenames in os.walk(directory):
    for filename in filenames:
        original_fake_paths.append([os.path.join(dirname, filename), filename])

# # First downsize all the images
# If not created a folder 'fake' then create one in the same directory 
save_dir = 'fake'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

fake_paths = [save_dir + filename for _, filename in original_fake_paths]

for path, filename in original_fake_paths:
    img = cv2.imread(path)
    img = cv2.resize(img, (224, 224))
    cv2.imwrite(os.path.join(save_dir, filename), img)

train_fake_paths, test_fake_paths = train_test_split(fake_paths, test_size=20000, random_state=2019)

fake_train_df = pd.DataFrame(train_fake_paths, columns=['filename'])
fake_train_df['class'] = 'FAKE'

fake_test_df = pd.DataFrame(test_fake_paths, columns=['filename'])
fake_test_df['class'] = 'FAKE'

# # Create real file paths dataframe
real_dir = './img_align_celeba'  #dataset folder 
eval_partition = pd.read_csv('./list_eval_partition.csv')

eval_partition['filename'] = eval_partition.image_id.apply(lambda st: real_dir + st)
eval_partition['class'] = 'REAL'

real_train_df = eval_partition.query('partition in [0, 1]')[['filename', 'class']]
real_test_df = eval_partition.query('partition == 2')[['filename', 'class']]

# # Combine both real and fake for dataframe
train_df = pd.concat([real_train_df, fake_train_df])
test_df = pd.concat([real_test_df, fake_test_df])

# # Generator
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_dataframe(
    train_df,
    target_size=(224, 224),
    batch_size=64,
    class_mode='binary',
    subset='training'
)

val_gen = datagen.flow_from_dataframe(
    train_df,
    target_size=(224, 224),
    batch_size=64,
    class_mode='binary',
    subset='validation'
)

datagen = ImageDataGenerator(rescale=1./255).flow_from_dataframe(
    test_df,
    target_size=(224, 224),
    batch_size=64,
    class_mode='binary'
)

# # Modelling
# # Load and freeze DenseNet

densenet = DenseNet121(
    weights='./DenseNet-BC-121-32-no-top.h5',
    include_top=False,
    input_shape=(224,224,3)
)

for layer in densenet.layers:
    layer.trainable = False

# # Build Model
    
def build_model(densenet):
    model = Sequential()
    model.add(densenet)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    
    model.add(layers.Dense(1, activation='sigmoid'))
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=0.0005),
        metrics=['accuracy']
    )
    
    return model

model = build_model(densenet)
model.summary()

# Training Phase 1 - Only train top layers

checkpoint = ModelCheckpoint('model.h5', save_best_only=True)

train_history_step1 = model.fit_generator(
    train_gen,
    validation_data=val_gen,
    steps_per_epoch=len(train_gen),
    validation_steps=len(val_gen),
    callbacks=[checkpoint],
    epochs=7
)

# Training Phase 2 - Unfreeze and train all
model.load_weights('model.h5')
for layer in model.layers:
    layer.trainable = True

train_history_step2 = model.fit_generator(
    train_gen,
    validation_data=val_gen,
    steps_per_epoch=len(train_gen),
    validation_steps=len(val_gen),
    callbacks=[checkpoint],
    epochs=3
)

# Eval
pd.DataFrame(train_history_step1.history).to_csv('history1.csv')
pd.DataFrame(train_history_step2.history).to_csv('history2.csv')
print("*********DONE***********")