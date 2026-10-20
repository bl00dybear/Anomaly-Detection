import numpy as np

from matplotlib import pyplot as plt
from pyod.utils.data import generate_data
from pyod.models import knn
from sklearn.metrics import confusion_matrix, balanced_accuracy_score,roc_curve
 
def task_1_2():
    X_train, X_test, y_train, y_test = generate_data(n_train=400,n_test=100,contamination=0.1)

    # print(set[0])

    plt.scatter(X_train[:,0],X_train[:,1], c=y_train, cmap='bwr')
    plt.savefig("./plots/1.pdf")

    plt.close()

    model = knn.KNN()

    model = model.fit(X_train)

    y_hat = model.predict(X_test)
    y_scores = model.decision_function(X_test)


    tn,fp,fn,tp=confusion_matrix(y_true=y_test,y_pred=y_hat).ravel()

    tpr=tp/(tp+fn)
    tnr=tn/(tn+fp)

    ba = (tpr+tnr)/2

    print(f"Balanced acc: {ba}")


    fpr, tpr, thresholds = roc_curve(y_true=y_test,y_score=y_scores)


    plt.plot(fpr,tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig("./plots/2.pdf")
    plt.close()


def task_3():
    points, _, labels, _= generate_data(n_train=1000,n_test=0,contamination=0.1,n_features=1)
    
    # print(np.shape(points))
    
    mu = np.mean(points)
    sigma=np.std(points)
        
    z_scores = np.array([np.abs((points[i]-mu)/sigma) for i in range (0,np.shape(points)[0])]).reshape(np.shape(points)[0])
    
    q90 = np.quantile(z_scores, q=0.9)
    
    tp,tn,fp,fn = 0,0,0,0
    
    for i, point in enumerate(points):
        if z_scores[i]<=q90:
            if labels[i] == 0:
                tn+=1
            else:
                fn+=1
        else:
            if labels[i] == 1:
                tp+=1
            else:
                fp+=1
    
    tpr=tp/(tp+fn)
    tnr=tn/(tn+fp)
    
    ba = (tpr+tnr)/2
    print(f"Balanced acc: {ba}")
    
    
def task_4():
    n=1000
    
    mu = np.array([2,3,15])
    sigma = np.array([
        [25, 23, 22],
        [23, 30, 13],
        [22, 13, 40]
    ])
    
    try:
        L = np.linalg.cholesky(sigma)
        
        X = np.random.standard_normal((n,3))
    
        y = np.array([np.dot(L,X[i])+mu for i in range (0,np.shape(X)[0])])
    
        print(np.shape(y))
        
        rng = np.random.default_rng(0)
        labels = np.zeros(n, dtype=int)
        idx = rng.choice(n, size=int(0.1 * n), replace=False)
        y[idx] += 8.0
        labels[idx] = 1

        z = (y - y.mean(axis=0)) / y.std(axis=0)
        z_scores = np.max(np.abs(z), axis=1)

        print(np.shape(z_scores))

        q90 = np.quantile(z_scores, 0.9)        
        
        print(q90)
        
        tp,tn,fp,fn = 0,0,0,0
    
        for i in range(np.shape(z_scores)[0]):
            if z_scores[i]<=q90:
                if labels[i] == 0:
                    tn+=1
                else:
                    fn+=1
            else:
                if labels[i] == 1:
                    tp+=1
                else:
                    fp+=1
        
        tpr=tp/(tp+fn)
        tnr=tn/(tn+fp)
        
        ba = (tpr+tnr)/2
        print(f"Balanced acc: {ba}")
            
        
        
        
    except np.linalg.LinAlgError:
        print("Sigma must be symetric and positive defined")
    

    

def main():
    task_1_2()
    
    task_3()
    
    task_4()


if __name__ == "__main__":
    main()