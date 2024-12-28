import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (confusion_matrix, classification_report, roc_curve, roc_auc_score, silhouette_score,
                             pairwise_distances, calinski_harabasz_score, davies_bouldin_score)
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings("ignore")

# Load the data
file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'heart.csv.txt')
heart_data = pd.read_csv(file_path)

# Data Exploration and Visualization
sns.heatmap(heart_data.corr(), cmap='crest')
plt.show()

# Boxplot for all columns
for column in heart_data.columns:
    sns.boxplot(data=heart_data[column])
    plt.title(f"Boxplot of {column}")
    plt.show()

# Pie charts for categorical columns
categorical_columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
fig, axs = plt.subplots(2, 4, figsize=(20, 10))
for idx, column in enumerate(categorical_columns):
    value_counts = heart_data[column].value_counts()
    axs[idx // 4, idx % 4].pie(value_counts, labels=value_counts.index, autopct='%1.1f%%')
    axs[idx // 4, idx % 4].set_title(column)
plt.show()

# Handle missing values
missing_columns = ["age", "fbs", "exang", "slope", "trestbps"]
for col in missing_columns:
    heart_data[col] = heart_data[col].apply(lambda x: np.nan if np.random.rand() < 0.1 else x)

imputer = SimpleImputer(strategy='mean')
heart_data[missing_columns] = imputer.fit_transform(heart_data[missing_columns])

# Prepare data for modeling
X = heart_data.iloc[:, :-1]
y = heart_data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Training Pipeline
models = {
    'Logistic Regression': LogisticRegression(),
    'KNN': KNeighborsClassifier(),
    'SVM': SVC(probability=True),
    'Naive Bayes': GaussianNB(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'Bagging': BaggingClassifier()
}

class ModelPipeline:
    def __init__(self, models, X_train, X_test, y_train, y_test):
        self.models = models
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.predictions = {}

    def train(self):
        for name, model in self.models.items():
            model.fit(self.X_train, self.y_train)

    def evaluate(self):
        for name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            cm = confusion_matrix(self.y_test, y_pred)
            print(f"\n{name} - Classification Report:\n{classification_report(self.y_test, y_pred)}")
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f"Confusion Matrix - {name}")
            plt.show()

    def plot_roc_curve(self):
        plt.figure(figsize=(10, 6))
        for name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(self.X_test)[:, 1]
                fpr, tpr, _ = roc_curve(self.y_test, y_proba)
                auc_score = roc_auc_score(self.y_test, y_proba)
                plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.2f})")
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.show()

pipeline = ModelPipeline(models, X_train, X_test, y_train, y_test)
pipeline.train()
pipeline.evaluate()
pipeline.plot_roc_curve()

# Clustering
class Clustering:
    def __init__(self, data):
        self.data = data

    def elbow_method(self):
        inertia = []
        for k in range(1, 11):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(self.data)
            inertia.append(kmeans.inertia_)
        plt.plot(range(1, 11), inertia, marker='o')
        plt.title('Elbow Method')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Inertia')
        plt.show()

    def hierarchical_clustering(self):
        dendrogram(linkage(self.data, method='ward'))
        plt.title('Dendrogram')
        plt.xlabel('Samples')
        plt.ylabel('Distance')
        plt.show()

    def kmeans_clustering(self, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.labels = kmeans.fit_predict(self.data)
        print(f"Cluster Labels: {np.unique(self.labels)}")

    def evaluate_clustering(self):
        silhouette_avg = silhouette_score(self.data, self.labels)
        print(f"Silhouette Score: {silhouette_avg:.2f}")

cluster_data = scaler.fit_transform(heart_data.iloc[:, :-1])
clustering = Clustering(cluster_data)
clustering.elbow_method()
clustering.hierarchical_clustering()
clustering.kmeans_clustering(n_clusters=3)
clustering.evaluate_clustering()
