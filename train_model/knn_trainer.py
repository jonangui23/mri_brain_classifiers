from sklearn.neighbors import KNeighborsClassifier

def train_knn(X, y):
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X, y)
    return model
