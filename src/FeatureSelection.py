import numpy as np
def BackwardsFeatureSelection(classifier, X_train, y_train, X_test, y_test):

    if(X_train.ndim < 2):
        return -1

    X_train_tmp = X_train
    X_test_tmp = X_test
    
    classifier.fit(X_train, y_train)
    prediction = classifier.predict(X_test)

    accuracy = [np.sum(np.asarray(np.round(prediction), dtype=np.int) == y_test)/float(prediction.size)]
    features = [range(0, X_train.shape[1])]

    while X_train_tmp.shape[1] > 1:

        m = findWorstFeature(classifier, X_train_tmp, y_train, X_test_tmp, y_test)

        X_train_tmp = np.delete(X_train_tmp, m, axis=1)
        X_test_tmp = np.delete(X_test_tmp, m, axis=1)

        classifier.fit(X_train_tmp, y_train)
        prediction = classifier.predict(X_test_tmp)

        accuracy.append(np.sum(np.asarray(np.round(prediction), dtype=np.int) == y_test)/float(prediction.size))
        features.append(np.delete(features[-1], m, axis=0))

    print accuracy

    return features[np.argmax(accuracy)]



def findWorstFeature(classifier, X_train, y_train, X_test, y_test):

    accuracy = []

    for i in range(0, X_train.shape[1]):
        
        X_train_tmp = X_train
        X_train_tmp = np.delete(X_train_tmp, i, axis=1)

        X_test_tmp = X_test
        X_test_tmp = np.delete(X_test_tmp, i, axis=1)

        classifier.fit(X_train_tmp, y_train)
        prediction = classifier.predict(X_test_tmp)

        accuracy.append(np.sum(prediction == y_test)/float(prediction.size))

    return np.argmax(accuracy)