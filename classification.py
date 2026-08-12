from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()

x = iris.data
y = iris.target

x_train , x_test , y_train , y_test = train_test_split(
    x, y, 
    test_size = 0.2,
    random_state = 42
)
model = KNeighborsClassifier()

model.fit(x_train, y_train)

predictions = model.predict(x_test)
print("Predictions: ")
print(predictions)

print("\nActual Labels: ")
print(y_test)

accuracy = accuracy_score (y_test, predictions)
print("\nAccuracy: ", accuracy)

new_flower = [[2.1, 0.2, 3.5, 1.5]]
prediction = model.predict(new_flower)

print("Prediction: ", prediction)
print("Flower Name: ", iris.target_names[prediction[0]])