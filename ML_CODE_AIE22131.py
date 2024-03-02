import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import RandomizedSearchCV
import librosa


class Main:
    def __init__(self):
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self, base_folder_path):
        X = []
        y = []
        for subfolder_name in os.listdir(base_folder_path):
            label = subfolder_name
            subfolder_path = os.path.join(base_folder_path, subfolder_name)

            for file_name in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, file_name)
                if os.path.isfile(file_path):
                    features = self.extract_features(file_path)
                    X.append(features)
                    y.append(label)
        self.X = np.array(X)
        self.y = np.array(y)

    def extract_features(self, file_path):
        y, sr = librosa.load(file_path)
        mfccs = librosa.feature.mfcc(y=y, sr=sr)
        spectral_flux = librosa.feature.spectral_flux(y=y, sr=sr)
        features = np.concatenate([mfccs, spectral_flux])
        return features

    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42
        )

    def task_a1(self):
        # A1. Evaluation of Confusion Matrix and Performance Metrics:
        knn_classifier = KNeighborsClassifier(n_neighbors=3)
        knn_classifier.fit(self.X_train, self.y_train)
        y_pred = knn_classifier.predict(self.X_test)
        return confusion_matrix(self.y_test, y_pred), classification_report(self.y_test, y_pred)

    def task_a2(self):
        # A2. Calculation and Analysis of Regression Metrics:
        knn_regressor = KNeighborsRegressor(n_neighbors=3)
        knn_regressor.fit(self.X_train, self.y_train)
        y_pred_reg = knn_regressor.predict(self.X_test)
        return mean_squared_error(self.y_test, y_pred_reg), np.sqrt(mean_squared_error(self.y_test, y_pred_reg)), mean_absolute_percentage_error(self.y_test, y_pred_reg), r2_score(self.y_test, y_pred_reg)

    def task_a3(self):
        # A3. Generating Training Data and Plotting Scatter Plot:
        colors = ['blue' if label == 0 else 'red' for label in self.y_train]
        return self.X_train, colors

    def task_a4(self):
        # A4. Generating Test Data and Classifying with kNN:
        knn_classifier = KNeighborsClassifier(n_neighbors=3)
        knn_classifier.fit(self.X_train, self.y_train)
        x_values = np.arange(0, 10.1, 0.1)
        y_values = np.arange(0, 10.1, 0.1)
        test_data = np.array(np.meshgrid(x_values, y_values)).T.reshape(-1, 2)
        predicted_labels = knn_classifier.predict(test_data)
        test_colors = ['blue' if label == 0 else 'red' for label in predicted_labels]
        return test_data, test_colors

    def task_a7(self):
        # A7. Use RandomizedSearchCV() to find the ideal ‘k’ value for your kNN classifier.
        param_grid = {'n_neighbors': range(1, 21)}
        knn_classifier = KNeighborsClassifier()
        random_search = RandomizedSearchCV(knn_classifier, param_distributions=param_grid, n_iter=10, cv=5, scoring='accuracy')
        random_search.fit(self.X_train, self.y_train)
        return random_search.best_params_['n_neighbors']

if __name__ == "__main__":
    main = Main()
    folder_path = input("Enter the path to your dataset folder: ")
    main.load_data(folder_path)
    main.split_data()
    # A1
    conf_matrix, classification_rep = main.task_a1()
    print("Confusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(classification_rep)
    # A2
    mse, rmse, mape, r2 = main.task_a2()
    print("\nRegression Metrics:")
    print("Mean Squared Error:", mse)
    print("Root Mean Squared Error:", rmse)
    print("Mean Absolute Percentage Error:", mape)
    print("R-squared Score:", r2)
    # A3
    X_train, colors = main.task_a3()
    plt.scatter(X_train[:, 0], X_train[:, 1], c=colors)
    plt.xlabel('Feature X')
    plt.ylabel('Feature Y')
    plt.title('Training Data Scatter Plot')
    plt.show()
    # A4
    test_data, test_colors = main.task_a4()
    plt.scatter(test_data[:, 0], test_data[:, 1], c=test_colors, alpha=0.1)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=colors)
    plt.xlabel('Feature X')
    plt.ylabel('Feature Y')
    plt.title('Test Data Scatter Plot with Predicted Classes')
    plt.show()
    # A7
    best_k = main.task_a7()
    print("Best k value:", best_k)