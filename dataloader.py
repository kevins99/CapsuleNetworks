import pandas as pd
from PIL import Image, ImageOps
# from keras.preprocessing import image
import os
import numpy as np
import pickle
from skimage import io


trainLabels = pd.read_csv('trainLabels_cropped.csv')

img_rows, img_cols = 128, 128

path = os.getcwd()
path = path+'/resized_train_cropped/resized_train_cropped'
listing = os.listdir(path)
print(listing)
listing = listing[:16000]

immatrix = []
imlabel = []
count = 0

for file in listing:
    base = os.path.basename(path+'/'+file)
    print(base)
    fileName = os.path.splitext(base)[0]
    df_search = trainLabels.loc[trainLabels['image']==fileName]
    assert not df_search.empty 
    imlabel.append(df_search['level'].iloc[0])
    print(path+'/'+file)
    im = Image.open(path+'/'+file)
    img = np.array(im.resize((img_rows, img_cols)))
    print(img.shape)
    immatrix.append(img)
    # convert to green channel only
    img[:,:,[0,2]] = 0
    immatrix.append(img)
    count += 1

im = Image.fromarray(immatrix[1])
print("level:", imlabel[1])
print(im)

from sklearn.utils import shuffle

data, label = shuffle(immatrix, imlabel, random_state=42)
train_data = [data, label]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(train_data[0], train_data[1], test_size = 0.1, random_state = 42)

print(np.array(x_train).shape)
print(np.array(y_train).shape)

from keras.utils import np_utils

y_train = np_utils.to_categorical(np.array(y_train), 5)
y_test = np_utils.to_categorical(np.array(y_test), 5)

X_train = np.array(x_train).astype("float32")/255.
X_test = np.array(x_test).astype("float32")/255.

print(np.array(y_train).shape)


with open('pickles/X_train.pkl', 'wb') as f:
    pickle.dump(X_train, f)

with open('pickles/X_test.pkl', 'wb') as f:
    pickle.dump(X_test, f)

with open('pickles/y_train.pkl', 'wb') as f:
    pickle.dump(y_train, f)

with open('pickles/y_test.pkl', 'wb') as f:
    pickle.dump(y_test, f)