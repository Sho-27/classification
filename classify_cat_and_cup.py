import sys
import numpy as np
from glob import glob
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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
        tf.keras.layers.Flatten(input_shape = (200, 200)),
        tf.keras.layers.Dense(128, activation = 'relu'),
        tf.keras.layers.Dense(2)
    ])
    return model

def compile_model(model):
    model.compile(optimizer = 'adam',
                    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
                    metrics = ['accuracy'])

def fit_model(model, train_images, train_labels, epochs):
    history = model.fit(train_images, train_labels, epochs = epochs, validation_split = 0.2)
    return history.history

def probability_model(model, img):
    probability_model = tf.keras.Sequential([model,
                                            tf.keras.layers.Softmax()])
    predictions = probability_model.predict(img)
    predictions_res = np.argmax(predictions[0])
    return predictions_res

def inference_res(predictions_res):
    print('===========================')
    if predictions_res == 0:
        print('This is a CUP.')
    elif predictions_res == 1:
        print('This is a CAT.')
    print('===========================')

def check_fit_history(history):
    acc = history['accuracy']
    loss = history['loss']
    val_acc = history['val_accuracy']
    val_loss = history['val_loss']
    epoch = range(1, len(acc) + 1)
    Figure = plt.figure(figsize = (12, 7))
    Figure.subplots_adjust(hspace = 0.4, wspace = 0.2)
    acc_graph = Figure.add_subplot(2, 2, 1, title = 'Accuracy_graph', xlabel = 'Epoch', ylabel = 'Accuracy')
    loss_graph = Figure.add_subplot(2, 2, 2, title = 'Loss_graph', xlabel = 'Epoch', ylabel = 'Loss')
    val_acc_graph = Figure.add_subplot(2, 2, 3, title = 'Val_accuracy_graph', xlabel = 'Epoch', ylabel = 'Val_accuracy')
    val_loss_graph = Figure.add_subplot(2, 2, 4, title = 'Val_loss_graph', xlabel = 'Epoch', ylabel = 'Val_loss')    
    acc_graph.plot(epoch, acc)
    loss_graph.plot(epoch, loss)
    val_acc_graph.plot(epoch, val_acc)
    val_loss_graph.plot(epoch, val_loss)
#    acc_graph.xaxis.set_major_locator(ticker.MultipleLocator(1))
#    loss_graph.xaxis.set_major_locator(ticker.MultipleLocator(1))
#    val_acc_graph.xaxis.set_major_locator(ticker.MultipleLocator(1))
#    val_loss_graph.xaxis.set_major_locator(ticker.MultipleLocator(1))
    loss_graph.yaxis.set_major_formatter(ticker.FuncFormatter(lambda epoch, loss : '{:,}'.format(int(epoch))))
    plt.show()

def main():
    train_images, train_labels = mk_train_ds()
    model = def_model()
    compile_model(model)
    history = fit_model(model, train_images, train_labels, 100)

    img_for_predictions = sys.argv[1]
    img_for_predictions = conv_im_to_numpy(img_for_predictions)
    predictions_res = probability_model(model, img_for_predictions)
    inference_res(predictions_res)

    check_fit_history(history)

if __name__ == '__main__':
    main()
