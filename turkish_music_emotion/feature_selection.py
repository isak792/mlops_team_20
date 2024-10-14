import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

class FeatureSelector:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.processed_data_path = f"{data_dir}/processed/Acoustic_Features_Processed.csv"
        self.feature_selected_data_path = f"{data_dir}/processed/feature_selected_data.csv"
        self.df = pd.read_csv(self.processed_data_path)
        self.X = self.df.drop('Class', axis=1)
        self.y = self.df['Class']
        self.le = LabelEncoder()
        self.y_encoded = self.le.fit_transform(self.y)

    def variance_threshold_selection(self, threshold=0.01):
        selector = VarianceThreshold(threshold)
        selector.fit(self.X)
        return self.X.columns[selector.get_support()].tolist()

    def correlation_analysis(self, X, threshold=0.8):
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        return [col for col in X.columns if col not in to_drop]

    def mutual_information_selection(self, X, n_features=20):
        mi_scores = mutual_info_classif(X, self.y_encoded)
        mi_scores = pd.Series(mi_scores, index=X.columns)
        mi_scores = mi_scores.sort_values(ascending=False)
        return mi_scores.head(n_features).index.tolist()

    def random_forest_importance(self, X, n_features=20):
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, self.y_encoded)
        importances = pd.Series(rf.feature_importances_, index=X.columns)
        importances = importances.sort_values(ascending=False)
        return importances.head(n_features).index.tolist()

    def plot_feature_importances(self, importances, title):
        plt.figure(figsize=(10, 6))
        importances.plot(kind='bar')
        plt.title(title)
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.tight_layout()
        plt.show()

    def select_features(self):
        print("Original number of features:", self.X.shape[1])

        # 1. Variance Threshold
        var_selected = self.variance_threshold_selection()
        print("Features after variance threshold:", len(var_selected))

        # 2. Correlation Analysis
        corr_selected = self.correlation_analysis(self.X[var_selected])
        print("Features after correlation analysis:", len(corr_selected))

        # 3. Mutual Information
        mi_selected = self.mutual_information_selection(self.X[corr_selected])
        print("Top 20 features by mutual information:", mi_selected)

        # Plot mutual information scores
        mi_scores = mutual_info_classif(self.X[corr_selected], self.y_encoded)
        mi_scores = pd.Series(mi_scores, index=self.X[corr_selected].columns).sort_values(ascending=False)
        self.plot_feature_importances(mi_scores, 'Mutual Information Scores')

        # 4. Random Forest Feature Importance
        rf_selected = self.random_forest_importance(self.X[corr_selected])
        print("Top 20 features by random forest importance:", rf_selected)

        # Plot random forest feature importances
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(self.X[corr_selected], self.y_encoded)
        importances = pd.Series(rf.feature_importances_, index=self.X[corr_selected].columns).sort_values(ascending=False)
        self.plot_feature_importances(importances, 'Random Forest Feature Importances')

        # Combine selected features
        combined_features = list(set(mi_selected + rf_selected))
        print("Number of features after combining MI and RF:", len(combined_features))

        return combined_features

    def save_selected_features(self, selected_features):
        final_df = self.df[selected_features + ['Class']]
        final_df.to_csv(self.feature_selected_data_path, index=False)
        print(f"Feature selection completed. Selected features saved to '{self.feature_selected_data_path}'")
        print("\nFinal selected features:")
        print(selected_features)
