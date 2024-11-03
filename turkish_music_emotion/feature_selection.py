"""
Módulo de Selección de Características para Clasificación de Emociones en Música Turca.

Este módulo proporciona un pipeline completo de selección de características que combina
múltiples métodos para identificar las características más relevantes para la clasificación
de emociones en música turca. Los métodos incluyen:
- Filtrado por umbral de varianza
- Análisis de correlación
- Selección por información mutua
- Selección por importancia usando Random Forest

El módulo tiene como objetivo reducir la dimensionalidad mientras preserva las características
más informativas para la tarea de clasificación.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

class FeatureSelector:
    """
    Clase para realizar selección de características usando múltiples métodos.

    Esta clase implementa un pipeline completo de selección de características que
    combina varios métodos para seleccionar las características más relevantes
    para la clasificación.

    Attributes:
        data_dir (str): Directorio que contiene los archivos de datos.
        processed_data_path (str): Ruta al archivo de características acústicas procesadas.
        feature_selected_data_path (str): Ruta donde se guardarán las características seleccionadas.
        df (pd.DataFrame): DataFrame que contiene el conjunto de datos completo.
        X (pd.DataFrame): Matriz de características.
        y (pd.Series): Variable objetivo.
        le (LabelEncoder): Codificador de etiquetas para la variable objetivo.
        y_encoded (np.array): Variable objetivo codificada.
    """

    def __init__(self, data_dir):
        """
        Inicializa el FeatureSelector con el directorio de datos.

        Args:
            data_dir (str): Ruta al directorio que contiene los archivos de datos.
        """
        self.data_dir = data_dir
        self.processed_data_path = f"{data_dir}/processed/Acoustic_Features_Processed.csv"
        self.feature_selected_data_path = f"{data_dir}/processed/feature_selected_data.csv"
        self.df = pd.read_csv(self.processed_data_path)
        self.X = self.df.drop('Class', axis=1)
        self.y = self.df['Class']
        self.le = LabelEncoder()
        self.y_encoded = self.le.fit_transform(self.y)

    def variance_threshold_selection(self, threshold=0.01):
        """
        Selecciona características basadas en su varianza.

        Elimina las características con varianza por debajo del umbral especificado.

        Args:
            threshold (float, optional): Umbral de varianza para la selección de características.
                Por defecto es 0.01.

        Returns:
            list: Lista de nombres de características que superan el umbral de varianza.
        """
        selector = VarianceThreshold(threshold)
        selector.fit(self.X)
        return self.X.columns[selector.get_support()].tolist()

    def correlation_analysis(self, X, threshold=0.8):
        """
        Elimina características altamente correlacionadas.

        Identifica y elimina características que tienen un coeficiente de correlación
        mayor que el umbral especificado.

        Args:
            X (pd.DataFrame): Matriz de características para analizar.
            threshold (float, optional): Umbral de correlación. Por defecto es 0.8.

        Returns:
            list: Lista de características después de eliminar las altamente correlacionadas.
        """
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        return [col for col in X.columns if col not in to_drop]

    def mutual_information_selection(self, X, n_features=20):
        """
        Selecciona características basadas en información mutua con la variable objetivo.

        Args:
            X (pd.DataFrame): Matriz de características para analizar.
            n_features (int, optional): Número de características principales a seleccionar. 
                Por defecto es 20.

        Returns:
            list: Lista de características principales basadas en puntuaciones de información mutua.
        """
        mi_scores = mutual_info_classif(X, self.y_encoded)
        mi_scores = pd.Series(mi_scores, index=X.columns)
        mi_scores = mi_scores.sort_values(ascending=False)
        return mi_scores.head(n_features).index.tolist()

    def random_forest_importance(self, X, n_features=20):
        """
        Selecciona características basadas en la importancia determinada por Random Forest.

        Args:
            X (pd.DataFrame): Matriz de características para analizar.
            n_features (int, optional): Número de características principales a seleccionar. 
                Por defecto es 20.

        Returns:
            list: Lista de características principales basadas en la importancia de Random Forest.
        """
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, self.y_encoded)
        importances = pd.Series(rf.feature_importances_, index=X.columns)
        importances = importances.sort_values(ascending=False)
        return importances.head(n_features).index.tolist()

    def plot_feature_importances(self, importances, title):
        """
        Genera una gráfica de barras para visualizar la importancia de las características.

        Args:
            importances (pd.Series): Series con las puntuaciones de importancia.
            title (str): Título para la gráfica.
        """
        plt.figure(figsize=(10, 6))
        importances.plot(kind='bar')
        plt.title(title)
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.tight_layout()
        plt.show()

    def select_features(self):
        """
        Ejecuta el pipeline completo de selección de características.

        Este método combina múltiples técnicas de selección:
        1. Filtrado por umbral de varianza
        2. Análisis de correlación
        3. Selección por información mutua
        4. Selección por importancia de Random Forest

        Returns:
            list: Lista final de nombres de características seleccionadas.

        Example:
            >>> selector = FeatureSelector("data_directory")
            >>> selected_features = selector.select_features()
            >>> print(f"Se seleccionaron {len(selected_features)} características")
        """
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

        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(self.X[corr_selected], self.y_encoded)
        importances = pd.Series(rf.feature_importances_, index=self.X[corr_selected].columns).sort_values(ascending=False)
        self.plot_feature_importances(importances, 'Random Forest Feature Importance')

        combined_features = list(set(mi_selected + rf_selected))
        print("Number of features after combining MI and RF:", len(combined_features))

        return combined_features

    def save_selected_features(self, selected_features):
        """
        Guarda las características seleccionadas en un archivo CSV.

        Args:
            selected_features (list): Lista de nombres de características a guardar.
        """
        final_df = self.df[selected_features + ['Class']]
        final_df.to_csv(self.feature_selected_data_path, index=False)
        print(f"Feature selection completed. Selected features saved to '{self.feature_selected_data_path}'")
        print("\nFinal selected features:")
        print(selected_features)
