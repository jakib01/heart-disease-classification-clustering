import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm

import warnings

warnings.filterwarnings("ignore")


import random
from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV

from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


# **Reading the data and finding the data information and plots**

# Get the current script's directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# Construct the file path
file_path = os.path.join(current_directory, 'heart.csv.txt')

# Load the CSV file
heart_data = pd.read_csv(file_path)

# print(heart_data.head())  # Display the first few rows to verify


# In[ ]:


heart_data


# In[ ]:


heart_data.describe()


# In[ ]:


heart_data.info()


# In[1]:


for column in heart_data.columns:
    print('-'*50 + ' ' + column + ' ' + '-'*50)
    sns.boxplot(data=heart_data[column])
    plt.show()


# In[ ]:


correlation = heart_data.corr()
sns.heatmap(correlation, cmap='crest')


# In[ ]:


columns_list = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
fig, axs = plt.subplots(2, 4, figsize=(20, 10))

for index, column in enumerate(columns_list):
  row = index // 4
  col = index % 4
  value_counts = heart_data[column].value_counts()
  axs[row, col].pie(value_counts, labels=value_counts.index, autopct='%1.1f%%')
  axs[row, col].set_title(column)

plt.show()


# In[ ]:


columns_con = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']


fig, axs = plt.subplots(2, 3, figsize=(20, 10))

# Loop over columns
for index, column in enumerate(columns_con):
    # Bin the data into 10 intervals
    bins = pd.cut(heart_data[column], bins=4)
    # Get the value counts of each interval
    value_counts = bins.value_counts()
    # Plot the pie chart
    row = index // 3
    col = index % 3
    axs[row, col].pie(value_counts, labels=value_counts.index, autopct='%1.1f%%')
    axs[row, col].set_title(column)

plt.show()


# In[ ]:


fig, axs = plt.subplots(5, 5, figsize=(40, 35))

for i, column1 in enumerate(columns_con):
    for j, column2 in enumerate(columns_con):
        # Bin the data into 10 intervals
        bins = pd.cut(heart_data[column1], bins=4)
        # Get the value counts of each interval
        value_counts = bins.value_counts()
        # Plot the 2D histogram
        sns.histplot(heart_data, x=column1, y=column2, hue="target", ax=axs[i, j])
        axs[i, j].set_title(f"{column1} vs {column2}")

plt.show()


# In[ ]:


fig, axs = plt.subplots(2, 4, figsize=(20, 10))

for index, column in enumerate(columns_list):
    # Calculate value counts
    value_counts = heart_data[column].value_counts()
    # Plot histogram
    row = index // 4
    col = index % 4
    sns.histplot(data=heart_data, x=column, hue=column, ax=axs[row, col])
    axs[row, col].set_title(column)

plt.show()


# In[ ]:


# Set the figure size
plt.figure(figsize=(40, 35))

# Loop over the continuous columns
for i, column1 in enumerate(columns_con):
    for j, column2 in enumerate(columns_con):
        # Check if the current pair of columns are different
        if column1 != column2:
            # Create a scatter plot using sns.lmplot()
            sns.lmplot(x=column1, y=column2, data=heart_data)
            # Set the title of the subplot
            plt.title(column1 + " vs " + column2)

# Adjust the spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()


# <p align="right">نمودار همبستگی به شکل دیگر و با استفاده از رگرسیون خطی در اینجا نمایش داده شده است

# **Generating missing values**

# In[ ]:


generate_missing_columns = {"age": 0, "fbs": 0, "exang": 0, "slope": 0, "trestbps": 0}

for column in generate_missing_columns:
  random_number_of_missing = np.random.randint(0, int(len(heart_data)/10))
  generate_missing_columns[column] = random_number_of_missing

for columns, random_missing in generate_missing_columns.items():
    for array in range(random_missing):
      missing_index = np.random.randint(0, len(heart_data))
      heart_data[columns][missing_index] = np.nan


# In[ ]:


heart_data.info()


# In[ ]:


imp_mean = SimpleImputer(strategy='mean')

for columns in generate_missing_columns:
  heart_data[columns] = imp_mean.fit_transform(heart_data[columns].values.reshape(-1,1))


# In[ ]:


heart_data.info()


# # **Building models**

# **Data splitting**

# In[ ]:


X = heart_data.iloc[:, :-1]
y = heart_data.iloc[:, -1].values


# In[ ]:


X


# In[ ]:


y


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


# In[ ]:


sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


models = {'lg' : LogisticRegression(),
    'knn' : KNeighborsClassifier(),
    'svm' : SVC(),
    'gnb' : GaussianNB(),
    'dt' : DecisionTreeClassifier(),
    'rf' : RandomForestClassifier(),
    'boosting' : AdaBoostClassifier(),
    'bagging' : BaggingClassifier()}

param_grids = {
    'lg': {
        'penalty': ['l2', 'none'],
        'C': [0.001, 0.01, 0.1, 1, 10],
        'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        'max_iter': [100, 200, 300]
    },
    'knn': {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree'],
        'leaf_size': [30, 40, 50]
    },
    'svm': {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'gamma': ['scale', 'auto'],
        'degree': [2, 3, 4]
    },
    'gnb': {},
    'dt': {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'rf': {
        'n_estimators': [10, 50, 100],
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'boosting': {
        'n_estimators': [10, 50, 100],
        'learning_rate': [0.1, 0.5, 1.0],
        'base_estimator': [DecisionTreeClassifier(max_depth=1), DecisionTreeClassifier(max_depth=2)],
        'algorithm': ['SAMME', 'SAMME.R']
    },
    'bagging': {
        'n_estimators': [10, 50, 100],
        'max_samples': [0.5, 1.0],
        'base_estimator': [None, DecisionTreeClassifier(max_depth=1), DecisionTreeClassifier(max_depth=2)],
        'bootstrap': [True, False],
        'bootstrap_features': [True, False]
    }
}


class ModelPipeline:
    def __init__(self, models, X_train, X_test, y_train, y_test):
        self.models = models
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.param_grids = param_grids
        self.predictions = {}
        self.pred_proba = {}
        self.cm = {}
        self.best_models = {}

    @staticmethod
    def __get_model_name(model_pred):
      model_name = model_pred.replace("y_pred_", "")
      return type(models[model_name]).__name__

    def fit_models(self):
        for model in self.models.values():
            model.fit(self.X_train, self.y_train)

    def prediction(self):
        for model_name, model in self.models.items():
            pred_str = 'y_pred_{}'.format(model_name)
            pred_model = model.predict(self.X_test)
            self.predictions[pred_str] = pred_model

    def metrics(self):
        self.prediction()

        for model_pred, y_pred in self.predictions.items():
            model_name = self.__get_model_name(model_pred)
            cm_str = 'cm_{}'.format(model_name)
            cm_model = confusion_matrix(self.y_test, y_pred)
            self.cm[cm_str] = cm_model

            print(50 * '-' + ' {} '.format(model_name) + 50 * '-')
            sns.heatmap(cm_model, annot=True)
            plt.show()
            print(classification_report(self.y_test, y_pred))



    def curve(self):
        plt.figure(figsize=(20, 10))
        auc_score_list = []
        colors = cm.rainbow(np.linspace(0, 1, len(self.models)))
        color_counter = 0

        for model_name, model in self.models.items():
            pred_str = 'y_pred_proba_{}'.format(model_name)

            if hasattr(model, 'predict_proba'):
                pred_model = model.predict_proba(self.X_test)
                self.pred_proba[pred_str] = pred_model

                fpr, tpr, thresh = roc_curve(self.y_test, self.pred_proba[pred_str][:, 1], pos_label=1)

                # roc curve for tpr = fpr
                random_probs = [0 for _ in range(len(self.y_test))]
                p_fpr, p_tpr, _ = roc_curve(self.y_test, random_probs, pos_label=1)

                auc_score = roc_auc_score(self.y_test, self.pred_proba[pred_str][:, 1])
                auc_score_list.append(auc_score)

                plt.style.use('seaborn')
                # plot roc curve with a different color for each model
                plt.plot(fpr, tpr, linestyle='--', color=colors[color_counter], label=model_name)
                plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
                # update color counter
                color_counter += 1

        # title
        plt.title('ROC curve')
        # x label
        plt.xlabel('False Positive Rate')
        # y label
        plt.ylabel('True Positive Rate')
        plt.legend(loc='best')
        plt.savefig('ROC', dpi=300)
        plt.show()

    def plot_auc_curve(self):
        plt.figure(figsize=(20, 10))
        for model_name, model in self.models.items():
            pred_str = 'y_pred_proba_{}'.format(model_name)
            if hasattr(model, 'predict_proba'):
                pred_model = model.predict_proba(self.X_test)
                self.pred_proba[pred_str] = pred_model
                auc_score = roc_auc_score(self.y_test, self.pred_proba[pred_str][:, 1])
                plt.bar(model_name, auc_score)
        plt.xlabel('Model')
        plt.ylabel('AUC Score')
        plt.title('Area Under the ROC Curve (AUC) Scores')
        plt.show()


    def gridsearch(self):
        for model_name, model in self.models.items():
            param_grid = self.param_grids[model_name]
            grid_search = GridSearchCV(model, param_grid, scoring='accuracy', cv=5)
            grid_search.fit(self.X_train, self.y_train)
            best_params = grid_search.best_params_
            best_model = grid_search.best_estimator_

            self.best_models[model_name] = best_model

            model_name = self.__get_model_name(model_name)
            print('\n')
            print(50 * '-' + ' {} '.format(model_name) + 50 * '-')
            print('Grid Search Results for {}:'.format(model_name))
            print('Best Parameters:', best_params)
            print('Best Model:', best_model)

        return self.best_models

    def get_classification_report(self):
        for model_name, model in self.best_models.items():
            model_name = self.__get_model_name(model_name)
            print('\n')
            print(50 * '-' + ' {} '.format(model_name) + 50 * '-')
            y_pred = model.predict(self.X_test)
            print('Classification Report for {}:'.format(model_name))
            print(classification_report(self.y_test, y_pred))


# In[ ]:


mp = ModelPipeline(models, X_train, X_test, y_train, y_test)


# In[ ]:


mp.fit_models()


# In[ ]:


mp.prediction()


# In[ ]:


mp.metrics()


# <p align="right">طبق متریک دقت، مدل کا نزدیک ترین همسایه و نایو بیز از بقیه بهتر هستند
#
#

# In[ ]:


mp.curve()


# In[ ]:


mp.plot_auc_curve()


# In[ ]:


best_models = mp.gridsearch()

# Get the classification report for the best models
mp.get_classification_report()


# # ***Phase 2: Clustering***

# In[ ]:


df = heart_data.iloc[:, :-1]
df = sc.fit_transform(df)

df


# In[ ]:


class Cluster:

    def __init__(self, data):
        self.data = data

    def number_of_clusters(self):
        inertia = []
        k_values = range(1, 11)

        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(self.data)
            inertia.append(kmeans.inertia_)

        # Plot the Elbow curve
        plt.plot(k_values, inertia, 'bo-')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia')
        plt.title('Elbow Method')
        plt.show()

        plt.figure(figsize=(10, 5))
        dendrogram(linkage(self.data, method='ward'))
        plt.title('Dendrogram')
        plt.xlabel('Sample Index')
        plt.ylabel('Distance')
        plt.show()

    def kmeans(self, k):
      kmeans = KMeans(n_clusters=k, random_state=42)
      kmeans.fit(df)
      labels = kmeans.labels_

      self.labels = labels

    def score(self):
      silhouette_avg = silhouette_score(self.data, self.labels)
      print('Silhouette Score:', silhouette_avg)

      # Calculate pairwise distances
      pairwise_dist = pairwise_distances(self.data)
      print('Pairwise Distances:')
      print(pairwise_dist)

      # Calculate Calinski-Harabasz score
      calinski_harabasz = calinski_harabasz_score(self.data, self.labels)
      print('Calinski-Harabasz Score:', calinski_harabasz)

      # Calculate Davies-Bouldin score
      davies_bouldin = davies_bouldin_score(self.data, self.labels)
      print('Davies-Bouldin Score:', davies_bouldin)


# In[ ]:


cl = Cluster(df)

cl.number_of_clusters()


# In[ ]:


cl.kmeans(2)
cl.score()


# In[ ]:


cl.kmeans(3)
cl.score()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




