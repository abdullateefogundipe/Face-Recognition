from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split



def train_func(data):
    scaler=StandardScaler()
    scaler.fit(data)
    scaled_data=pd.DataFrame(scaler.transform(data))
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.10, random_state=1)
    accuracy=[]
    for i in range(1,40):
        knn=KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        pred=knn.predict(X_test)
        accuracy.append(np.mean(pred==y_test))
    plt.figure()
    plt.plot(range(1,40),accuracy, marker='o', markerfacecolor='red', markersize=9)
    plt.xlabel('N')
    plt.ylabel('accuracy')    
    print(confusion_matrix( y_test,pred))
    print(classification_report( y_test,pred))
    return knn
    
    
def predict(model,thumbnail ):    
    return model.predict(thumbnail)
