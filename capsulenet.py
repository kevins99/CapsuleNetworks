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
from keras.preprocessing.image import ImageDataGenerator

K.set_image_data_format('channels_last')


def CapsNet(input_shape, n_class, routings):
    x = layers.Input(shape=input_shape)
    conv1 = layers.Conv2D(filters=256, kernel_size=(9), strides=1, padding='valid', activation='relu', name='conv1')(x)
    primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings,
                             name='digitcaps')(primarycaps)
    
    out_caps = Length(name='capsnet')(digitcaps)

    y = layers.Input(shape=(n_class, ))
    masked_by_y = Mask()([digitcaps, y])
    masked = Mask()(digitcaps)

    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(512, activation='relu', input_dim=16*n_class))
    decoder.add(layers.Dense(1024, activation='relu'))
    decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))

    decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

    # Models for training and evaluation (prediction)
    train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
    eval_model = models.Model(x, [out_caps, decoder(masked)])

    # manipulate model
    noise = layers.Input(shape=(n_class, 16))
    noised_digitcaps = layers.Add()([digitcaps, noise])
    masked_noised_y = Mask()([noised_digitcaps, y])
    manipulate_model = models.Model([x, y, noise], decoder(masked_noised_y))
    return train_model, eval_model, manipulate_model


def margin_loss(y_true, y_pred):
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))

def train(model, args):
    trainLabels = pd.read_csv('trainLabels_cropped.csv')
    trainLabels['image'] = trainLabels['image'].apply(lambda x:x+".jpeg")
    trainLabels['level'] = trainLabels['level'].astype(str)

    nb_classes = 5
    lbls = list(map(str, range(nb_classes)))
    batch_size = 32
    img_size = 224
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
        zoom_range = 0.3,
    )
    print('break')

    valid_generator=train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory="../input/aptos2019-blindness-detection/train_images",
        x_col="id_code",
        y_col="diagnosis",
        batch_size=batch_size,
        shuffle=True,
        class_mode="categorical", 
        classes=lbls,
        target_size=(img_size,img_size),
        subset='validation')


    # callbacks
    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                               batch_size=args.batch_size, histogram_freq=int(args.debug))
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', monitor='val_capsnet_acc',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))

    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., args.lam_recon],
                  metrics={'capsnet': 'accuracy'})

    model.fit([x_train, y_train], [y_train, x_train], batch_size=args.batch_size, epochs=args.epochs,
                validation_data=[[x_test, y_test], [y_test, x_test]], callbacks=[log, tb, checkpoint, lr_decay])

    model.save_weights(args.save_dir + '/trained_model.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

    return model


def test(model, data, args):
    x_test, y_test = data
    y_pred, x_recon = model.predict(x_test, batch_size=100)
    print('-'*30 + 'Begin: test' + '-'*30)
    print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0])

    img = combine_images(np.concatenate([x_test[:50],x_recon[:50]]))
    image = img * 255
    Image.fromarray(image.astype(np.uint8)).save(args.save_dir + "/real_and_recon.png")
    print()
    print('Reconstructed images are saved to %s/real_and_recon.png' % args.save_dir)
    print('-' * 30 + 'End: test' + '-' * 30)
    plt.imshow(plt.imread(args.save_dir + "/real_and_recon.png"))
    plt.show()

def load_data():
    with open('pickles/X_train.pkl', 'rb') as f:
        X_train = pickle.load(f)

    with open('pickles/y_train.pkl', 'rb') as f:
        y_train = pickle.load(f)

    with open('pickles/X_test.pkl', 'rb') as f:
        X_test = pickle.load(f)

    with open('pickles/y_test.pkl', 'rb') as f:
        y_test = pickle.load(f)

    return (X_train, y_train), (X_test, y_test)

if __name__ == "__main__":
    import os
    import argparse
    from keras import callbacks
    import pandas as pd

    parser = argparse.ArgumentParser(description='CapsNet for DR')

    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=2)
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
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # (X_train, y_train), (X_test, y_test) = load_data()
    # X_train.reshape((-1, 128, 128, 1))
    # X_test.reshape((-1, 128, 128, 1))
    # print(X_train.shape)
    # print(y_train.shape)


    

    model, eval_model, manipulate_model = CapsNet(input_shape=X_train.shape[1:],
                                                  n_class=len(np.unique(np.argmax(y_train, 1))),
                                                  routings=args.routings)

    model.summary()

    if args.weights is not None:  # init the model weights with provided one
        model.load_weights(args.weights)
    if not args.testing:
        train(model=model, data=((X_train, y_train), (X_test, y_test)), args=args)
    # else:  # as long as weights are given, will run testing
    #     if args.weights is None:
    #         print('No weights are provided. Will test using random initialized weights.')
    #     manipulate_latent(manipulate_model, (x_test, y_test), args)
    #     test(model=eval_model, data=(x_test, y_test), args=args)