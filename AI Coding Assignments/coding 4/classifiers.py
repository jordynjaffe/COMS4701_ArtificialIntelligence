from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from matplotlib import colormaps
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_dataset(file_path):
    data = pd.read_csv(file_path)
    X1 = data.iloc[:, 0]
    X2 = data.iloc[:, 1]
    labels = data.iloc[:, -1]
    plt.figure(figsize=(8, 6))
    for label in labels.unique():
        plt.scatter(X1[labels == label], X2[labels == label], label=f"Class {label}")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Scatter Plot of Dataset")
    plt.legend()
    plt.grid(True)
    #plt.show()
    plt.savefig('plot_dataset.png')

    plt.show(block=False)

class Classifiers():
    def __init__(self,data):
        ''' 
        TODO: Write code to convert the given pandas dataframe into training and testing data 
        # all the data should be nxd arrays where n is the number of samples and d is the dimension of the data
        # all the labels should be nx1 vectors with binary labels in each entry 
        '''

        #my code starts:

        #self.plot_dataset(file_path)
        #plot_dataset('input.csv')
        #X = data.iloc[:, :-1].values
        self.features = data.iloc[:, :-1].values
        self.labels = data.iloc[:, -1].values
        #Y = data.iloc[:, -1].values

        self.training_data, self.testing_data, self.training_labels, self.testing_labels = train_test_split(
            self.features, self.labels, test_size=0.4, random_state=42)

        
        # self.training_data = None
        # self.training_labels = None
        # self.testing_data = None
        # self.testing_labels = None
        self.outputs = []
    
    def test_clf(self, clf, param, classifier_name=''):
        # TODO: Fit the classifier and extrach the best score, training score and parameters
        #pass
        # Use the following line to plot the results
        grid_search = GridSearchCV(clf, param, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(self.training_data, self.training_labels.ravel())
        best_model = grid_search.best_estimator_
        testing_score = best_model.score(self.testing_data, self.testing_labels)

        # Record results: ORIGINAL DIRECTLY BELOW
        self.outputs.append(f"{classifier_name}, {grid_search.best_score_:.5f}, {testing_score:.5f}")
        #self.outputs.append(
            #f"{classifier_name} {grid_search.best_params_}, Best Training Score: {grid_search.best_score_:.5f}, Test Score: {testing_score:.5f}")

        # Plot results
        self.plot(self.testing_data, best_model.predict(self.testing_data), model=best_model,
                  classifier_name=classifier_name)
        #self.plot(self.testing_data, clf.predict(self.testing_data),model=clf,classifier_name=name)

    # def plot_dataset(file_path):
    #     data = pd.read_csv(file_path)
    #     X1 = data.iloc[:, 0]
    #     X2 = data.iloc[:, 1]
    #     labels = data.iloc[:, -1]
    #     plt.figure(figsize=(8, 6))
    #     for label in labels.unique():
    #         plt.scatter(X1[labels == label], X2[labels == label], label=f"Class {label}")
    #     plt.xlabel("Feature 1")
    #     plt.ylabel("Feature 2")
    #     plt.title("Scatter Plot of Dataset")
    #     plt.legend()
    #     plt.grid(True)
    #     plt.show()

    def classifyNearestNeighbors(self):
        # TODO: Write code to run a Nearest Neighbors classifier
        param = {
            'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
            'leaf_size': [5, 10, 15, 20, 25, 30]
        }
        k_near_neighbor = KNeighborsClassifier()
        self.test_clf(k_near_neighbor, param, "K-Nearest Neighbors")
        #grid_search =GridSearchCV(k_near_neighbor, param)
        # grid_search = GridSearchCV(k_near_neighbor, param, cv=5, scoring='accuracy', n_jobs=-1)
        # grid_search.fit(self.training_data, self.training_labels.ravel())
        # best_model = grid_search.best_estimator_
        # testing_score = best_model.score(self.testing_data, self.testing_labels)
        #
        # self.outputs.append(f"K-Nearest Neighbors, {grid_search.best_score_:.3f}, {testing_score:.3f}")
        # self.plot(self.testing_data, best_model.predict(self.testing_data), model=best_model,
        #           classifier_name="K-Nearest Neighbors")
        #pass
        
    def classifyLogisticRegression(self):
        # TODO: Write code to run a Logistic Regression classifier
        param = {
        'C': [0.1, 0.5, 1, 5, 10, 50, 100]
        }
        logistic_regression = LogisticRegression(max_iter=1000)
        self.test_clf(logistic_regression, param, "Logistic Regression")
        # grid_search = GridSearchCV(logistic_regression, param, cv=5, scoring='accuracy', n_jobs=-1)
        # grid_search.fit(self.training_data, self.training_labels.ravel())
        # best_model = grid_search.best_estimator_
        # testing_score = best_model.score(self.testing_data, self.testing_labels)
        # self.outputs.append(f"Logistic Regression, {grid_search.best_score_:.3f}, {testing_score:.3f}")
        # self.plot(self.testing_data, best_model.predict(self.testing_data), model=best_model,
        #           classifier_name="Logistic Regression")
        #pass
    
    def classifyDecisionTree(self):
        # TODO: Write code to run a Logistic Regression classifier
        param = {
            'max_depth': list(range(1, 51)),  # Max depth values from 1 to 50
            'min_samples_split': list(range(2, 11))  # Min samples split from 2 to 10
        }
        decision_tree = DecisionTreeClassifier(random_state=42)
        self.test_clf(decision_tree, param, "Decision Tree")
        #https://scikit-learn.org/stable/glossary.html#term-random_state ; suggested 42 as popular integer value for random_state
        # grid_search = GridSearchCV(decision_tree, param, cv=5, scoring='accuracy', n_jobs=-1)
        # grid_search.fit(self.training_data, self.training_labels)
        # best_model = grid_search.best_estimator_
        # testing_score = best_model.score(self.testing_data, self.testing_labels)
        # self.outputs.append(f"Decision Tree, {grid_search.best_score_:.3f}, {testing_score:.3f}")
        # self.plot(self.testing_data, best_model.predict(self.testing_data), model=best_model,
        #           classifier_name="Decision Tree")
        #pass

    def classifyRandomForest(self):
        # TODO: Write code to run a Random Forest classifier
        param = {
            'max_depth': [1, 2, 3, 4, 5],
            'min_samples_split': list(range(2, 11))
        }
        random_forest = RandomForestClassifier(random_state=42)
        self.test_clf(random_forest, param, "Random Forest")
        # grid_search = GridSearchCV(random_forest, param, cv=5, scoring='accuracy', n_jobs=-1)
        # grid_search.fit(self.training_data, self.training_labels.ravel())
        # best_model = grid_search.best_estimator_
        # testing_score = best_model.score(self.testing_data, self.testing_labels)
        #
        # self.outputs.append(f"Random Forest, {grid_search.best_score_:.3f}, {testing_score:.3f}")
        # self.plot(self.testing_data, best_model.predict(self.testing_data), model=best_model,
        #           classifier_name="Random Forest")
        #pass

    def classifyAdaBoost(self):
        # TODO: Write code to run a AdaBoost classifier
        param = {
            'n_estimators': list(range(10, 71, 10))
        }
        ada_boost = AdaBoostClassifier(random_state=42)
        self.test_clf(ada_boost, param, "AdaBoost")
        # grid_search = GridSearchCV(ada_boost, param, cv=5, scoring='accuracy', n_jobs=-1)
        # grid_search.fit(self.training_data, self.training_labels.ravel())
        # best_model = grid_search.best_estimator_
        # testing_score = best_model.score(self.testing_data, self.testing_labels)
        #
        # self.outputs.append(f"AdaBoost, {grid_search.best_score_:.3f}, {testing_score:.3f}")
        # self.plot(self.testing_data, best_model.predict(self.testing_data), model=best_model,
        #           classifier_name="AdaBoost")

        #pass

    def plot(self, X, Y, model,classifier_name = ''):
        X1 = X[:, 0]
        X2 = X[:, 1]

        X1_min, X1_max = min(X1) - 0.5, max(X1) + 0.5
        X2_min, X2_max = min(X2) - 0.5, max(X2) + 0.5

        X1_inc = (X1_max - X1_min) / 200.
        X2_inc = (X2_max - X2_min) / 200.

        X1_surf = np.arange(X1_min, X1_max, X1_inc)
        X2_surf = np.arange(X2_min, X2_max, X2_inc)
        X1_surf, X2_surf = np.meshgrid(X1_surf, X2_surf)

        L_surf = model.predict(np.c_[X1_surf.ravel(), X2_surf.ravel()])
        L_surf = L_surf.reshape(X1_surf.shape)

        plt.title(classifier_name)
        plt.contourf(X1_surf, X2_surf, L_surf, cmap = plt.cm.coolwarm, zorder = 1)
        plt.scatter(X1, X2, s = 38, c = Y)

        plt.margins(0.0)
        # uncomment the following line to save images
        plt.savefig(f'{classifier_name}.png')

        plt.show(block=False)  # Prevent hanging
        plt.pause(0.1)
        #plt.show()

    
if __name__ == "__main__":
    plot_dataset('input.csv')
    df = pd.read_csv('input.csv')
    models = Classifiers(df)
    print('Classifying with NN...')
    models.classifyNearestNeighbors()
    print('Classifying with Logistic Regression...')
    models.classifyLogisticRegression()
    print('Classifying with Decision Tree...')
    models.classifyDecisionTree()
    print('Classifying with Random Forest...')
    models.classifyRandomForest()
    print('Classifying with AdaBoost...')
    models.classifyAdaBoost()

    with open("output.csv", "w") as f:
        print('Name, Best Training Score, Testing Score',file=f)
        for line in models.outputs:
            print(line, file=f)