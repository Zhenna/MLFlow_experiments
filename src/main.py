from data import prepare_data
from model import shallow_CNN

NUM_CLASSES = 10
INPUT_SHAPE = (28, 28, 1)

def run_deep_learning():
    # load data
    (x_train, y_train), (x_test, y_test) = prepare_data(NUM_CLASSES)
    # build model
    model = shallow_CNN(INPUT_SHAPE, NUM_CLASSES)
    # fit model to data
    model.fit(x_train, y_train, batch_size=128, epochs=15, validation_split=0.1)
    # evaluate model performance
    score = model.evaluate(x_test, y_test, verbose=0)
    print(x_test[0].shape)
    return score[0], score[1], model

if __name__ == "__main__":
    run_deep_learning()
