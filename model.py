import numpy as np
from keras import layers, models, optimizers
from keras import backend as K
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from utils import combine_images
from PIL import Image
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
import pandas as pd
import pickle
from keras_preprocessing.image import ImageDataGenerator
from keras import callbacks
from keras.callbacks import Callback,ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
import json

K.set_image_data_format('channels_last')


def CapsNet(input_shape, n_class, routings):
    x = layers.Input(shape=input_shape)
    conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(x)
    primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings,
                             name='digitcaps')(primarycaps)
    
    out_caps = Length(name='capsnet')(digitcaps)

    # y = layers.Input(shape=(n_class, ))
    # masked_by_y = Mask()([digitcaps, y])
    # masked = Mask()(digitcaps)

    # decoder = models.Sequential(name='decoder')
    # decoder.add(layers.Dense(512, activation='relu', input_dim=16*n_class))
    # decoder.add(layers.Dense(1024, activation='relu'))
    # decoder.add(layers.Dense(np.prod(input_shape), activation='relu'))

    train_model = models.Model(x, out_caps)
    # eval_model = models.Model(x, [out_caps, decoder(masked)])

    # noise = layers.Input(shape=(n_class, 16))
    # noised_digitcaps = layers.Add()([digitcaps, noise])
    # masked_noised_y = Mask()([noised_digitcaps, y])
    # manipulate_model = models.Model([x, y, noise], decoder(masked_noised_y))
    # return train_model, eval_model, manipulate_model
    return train_model

def margin_loss(y_true, y_pred):
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))

def train(model, args):
    
    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                               batch_size=args.batch_size, histogram_freq=int(args.debug))
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', monitor='val_capsnet_acc',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))

    
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=margin_loss,
                  loss_weights=[1.],
                  metrics={'capsnet': 'accuracy'})

    trainLabels = pd.read_csv('trainLabels_cropped.csv')
    trainLabels['image'] = trainLabels['image'].apply(lambda x:x+".jpeg")
    trainLabels['level'] = trainLabels['level'].astype(str)
    
    nb_classes = 5
    lbls = list(map(str, range(nb_classes)))
    batch_size = 32
    img_size = 128
    nb_epochs = 30

    train_datagen=ImageDataGenerator(
        rescale=1./255,
        featurewise_center=True,
        featurewise_std_normalization=True,
        zca_whitening=True,
        rotation_range=45,
        width_shift_range=0.2, 
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=0.1,   
        zoom_range = 0.3
        )

    path = os.getcwd()
    path = path+'/resized_train_cropped/resized_train_cropped'

    path, dirs, files = next(os.walk(path))
    file_count = len(files)
    print

    train_generator=train_datagen.flow_from_dataframe(
        dataframe=trainLabels,
        directory=path,
        x_col="image",
        y_col="level",
        batch_size=batch_size,
        shuffle=True,
        class_mode="categorical",
        classes=lbls,
        target_size=(img_size,img_size),
        color_mode="grayscale",
        subset='training')


    valid_generator=train_datagen.flow_from_dataframe(
        dataframe=trainLabels,
        directory=path,
        x_col="image",
        y_col="level",
        batch_size=batch_size,
        shuffle=True,
        class_mode="categorical",
        classes=lbls,
        target_size=(img_size,img_size),
        subset='training')


    checkpoint = ModelCheckpoint(
        'capsnet.h5', 
        monitor='val_loss', 
        verbose=0, 
        save_best_only=True, 
        save_weights_only=False,
        mode='auto'
    )

    history = model.fit_generator(
        generator=train_generator,
        steps_per_epoch=int(file_count / batch_size),
        epochs=nb_epochs,
        validation_data=valid_generator,
        validation_steps = 30,
        callbacks=[log, tb, checkpoint, lr_decay]
    )

    with open('history.json', 'w') as f:
        json.dump(history.history, f)

    history_df = pd.DataFrame(history.history)
    history_df[['loss', 'val_loss']].plot()
    history_df[['acc', 'val_acc']].plot()



if __name__ == "__main__":
    import os
    import argparse
    parser = argparse.ArgumentParser(description="Capsule Network on MNIST.")
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.9, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('--lam_recon', default=0.392, type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('--shift_fraction', default=0.1, type=float,
                        help="Fraction of pixels to shift at most in each direction.")
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")
    parser.add_argument('--digit', default=5, type=int,
                        help="Digit to manipulate")
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    args = parser.parse_args()
    print(args)
    
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # model, eval_model, manipulate_model = CapsNet(input_shape=(128, 128, 3),
                                                #   n_class=5,routings=3)
    
    model = CapsNet(input_shape=(128, 128, 1), n_class=5, routings=3)

    model.summary()

    if args.weights is not None: 
        model.load_weights(args.weights)

    if not args.testing:
        train(model=model, args=args)
    else:
        if args.weights is None:
            print('No weight provided')

                                  

    
    
