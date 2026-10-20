import numpy as np

from matplotlib import pyplot as plt
from pyod.utils.data import generate_data
from pyod.models import knn
from sklearn.metrics import confusion_matrix, balanced_accuracy_score,roc_curve

def main():
    # set = np.array()
    X_train, X_test, y_train, y_test = generate_data(n_train=400,n_test=100,contamination=0.1)

    # print(set[0])

    plt.scatter(X_train[:,0],X_train[:,1], c=y_train, cmap='bwr')
    plt.savefig("./plots/1.pdf")

    plt.close()

    model = knn.KNN()

    model = model.fit(X_train)

    y_hat = model.predict(X_test)
    y_scores = model.decision_function(X_test)


    print(confusion_matrix(y_true=y_test,y_pred=y_hat))

    print(balanced_accuracy_score(y_true=y_test,y_pred=y_hat))

    fpr, tpr, thresholds = roc_curve(y_true=y_test,y_score=y_scores)


    plt.plot(fpr,tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig("./plots/2.pdf")


if __name__ == "__main__":
    main()