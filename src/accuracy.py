from collections import Counter
import numpy as np


def predict_class_audio(MFCCs, model):

    MFCCs = MFCCs.reshape(MFCCs.shape[0], MFCCs.shape[1], MFCCs.shape[2], 1)
    y_predicted = np.argmax(model.predict(MFCCs), axis=-1)
    return(Counter(list(y_predicted)).most_common(1)[0][0])


def predict_prob_class_audio(MFCCs, model):

    MFCCs = MFCCs.reshape(MFCCs.shape[0], MFCCs.shape[1], MFCCs.shape[2], 1)
    y_predicted = model.predict_proba(MFCCs, verbose=0)
    print("->>>", y_predicted)
    return(np.argmax(np.sum(y_predicted, axis=0)))


def predict_class_all(X_train, model):

    predictions = []
    for mfcc in X_train:
        predictions.append(predict_class_audio(mfcc, model))

    print(predictions)
    return predictions


def confusion_matrix(y_predicted, y_test):

    confusion_matrix = np.zeros((len(y_test[0]), len(y_test[0])), dtype=int)
    for index, predicted in enumerate(y_predicted):
        confusion_matrix[np.argmax(y_test[index])][predicted] += 1
    return(confusion_matrix)


def get_accuracy(y_predicted, y_test):

    c_matrix = confusion_matrix(y_predicted, y_test)
    return(np.sum(c_matrix.diagonal()) / float(np.sum(c_matrix)))


if __name__ == '__main__':
    pass
