import os
import sys
import numpy as np
from glob import glob
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

def get_dataset_path():
    return '../datasets/'

def conv_im_to_numpy(src_im):
    np_im = np.array(Image.open(src_im))
    np_im = np_im[np.newaxis, :, :]
    return np_im

def mk_train_ds():
    fl = glob(get_dataset_path() + '*')
    train_labels = np.array(np.arange(len(fl)))
    for f in fl:
        step = fl.index(f)
        if step == 0:
            train_images = conv_im_to_numpy(f)
            if 'cup' in f:
                train_labels[step] = 0
            elif 'cat' in f:
                train_labels[step] = 1
        elif 'cup' in f:
            cup_im = conv_im_to_numpy(f)
            train_images = np.append(train_images, cup_im, axis = 0)
            train_labels[step] = 0
        elif 'cat' in f:
            cat_im = conv_im_to_numpy(f)
            train_images = np.append(train_images, cat_im, axis = 0)
            train_labels[step] = 1

    return train_images, train_labels

def def_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape = (28, 28)),
        tf.keras.layers.Dense(128, activation = 'relu'),
        tf.keras.layers.Dense(2)
    ])
    return model

def compile_model(model):
    model.compile(optimizer = 'adam',
                    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
                    metrics = ['accuracy'])
    return model

def fit_model(model, train_images, train_labels, epochs):
    model.fit(train_images, train_labels, epochs = epochs)

def test_model(model, test_images, test_labels):
    test_loss, test_acc = model.evalute(test_images, test_labels, verbose = 2)
    print(f'Test accuracy : ' + {test_acc})

def probability_model(model, img):
    probability_model = tf.keras.Sequential([model,
                                            tf.keras.layers.Softmax()])
    predictions = probability_model.predict(img)
    predictions_res = np.argmax(predictions[0])

    print('===========================')
    if predictions_res == 0:
        print('This is a CUP.')
    elif predictions_res == 1:
        print('This is a CAT.')
    print('===========================')

def main():
    train_images, train_labels = mk_train_ds()
    model = def_model()
    model = compile_model(model)
    fit_model(model, train_images, train_labels, 20)

    img_for_predictions = sys.argv[1]
    img_for_predictions = conv_im_to_numpy(img_for_predictions)
    probability_model(model, img_for_predictions)

if __name__ == '__main__':
    main()
