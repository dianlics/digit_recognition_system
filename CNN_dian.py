# import library
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import to_categorical
import matplotlib.pyplot as plt


def build_network(x_train, y_train, num_classes, model_name='keras_mnist.h5'):
    # Build the model
    model = Sequential()
    model.add(Convolution2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(Convolution2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=200, verbose=1)
    print("The model has successfully trained.")

    # Save the model
    model.save(model_name)
    print("The model has successfully saved.")

    return model, history


def plot_performance(model, history, x_test, y_test, output_name="NNperformance.png"):
    '''
    Retrieve accuracies from the history object and save them to a figure.

    **Parameters**

        model:
            A class object that holds the trained model.
        history:
            A class object that holds the history of the model.
        output_name: *str, optional*
            The filename of the output image.

    **Returns**

        None
    '''

    # Plot accuracy
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='lower right')
    # Plot loss
    plt.subplot(2, 1, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.tight_layout()
    # Calculate loss and accuracy on the testing data
    loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
    print("Test Loss:", loss)
    print("Test Accuracy:", accuracy)
    # Save figure as .png file
    fig = plt.gcf()
    fig.savefig(output_name)


if __name__ == '__main__':
    # load dataset directly from keras lib
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # reshape format: [# of samples][width][height][channels]
    # channels = 1 means grey scale, while 3 means RGB.
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')

    # converts a class vector (integers) to binary class matrix
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # Normalize the data to help with training
    x_train /= 255
    x_test /= 255

    # define a CNN model
    num_classes = 10

    # Build network
    model, history = build_network(x_train, y_train, num_classes, 'CNN_digit.h5')

    # Observe the performance of the network
    plot_performance(model, history, x_test, y_test, "myCNNPerformance.png")
