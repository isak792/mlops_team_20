"""
Módulo de preprocesamiento y análisis de datos para características acústicas.

Este módulo proporciona clases para el preprocesamiento, análisis y visualización
de datos acústicos, incluyendo:
- Transformación de características sesgadas
- Manejo de outliers por clase
- Análisis exploratorio de datos
- Visualización mediante PCA
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class DataPreprocessor:
    """
    Clase para el preprocesamiento de características acústicas.

    Implementa transformaciones para normalizar la distribución de características
    sesgadas y maneja outliers específicos por clase.

    Attributes:
        data_dir (str): Directorio que contiene los datos.
        raw_data_path (str): Ruta al archivo de datos crudos.
        processed_data_path (str): Ruta donde se guardarán los datos procesados.
        df (pd.DataFrame): DataFrame con los datos crudos.
        df_transformed (pd.DataFrame): DataFrame con los datos transformados.
        pt (PowerTransformer): Transformador para normalización.
        skewed_features (list): Lista de características identificadas como sesgadas.
    """

    def __init__(self, data_dir):
        """
        Inicializa el preprocesador con el directorio de datos.

        Args:
            data_dir (str): Ruta al directorio que contiene los datos.
        """
        self.data_dir = data_dir
        self.raw_data_path = os.path.join(data_dir, 'raw', 'Acoustic Features.csv')
        self.processed_data_path = os.path.join(data_dir, 'processed', 'Acoustic_Features_Processed.csv')
        self.df = pd.read_csv(self.raw_data_path)
        self.df_transformed = None
        self.pt = PowerTransformer(method='yeo-johnson', standardize=True)
        self.skewed_features = [
            '_Fluctuation_Mean',
            '_Roughness_Mean',
            '_AttackTime_Mean',
            '_Pulseclarity_Mean',
            '_Spectralskewness_Mean',
            '_Spectralkurtosis_Mean',
            '_Spectralflatness_Mean',
            '_Chromagram_Mean_2',
            '_Chromagram_Mean_4',
            '_Chromagram_Mean_6',
            '_Chromagram_Mean_7',
            '_HarmonicChangeDetectionFunction_PeriodEntropy'
        ]

    def plot_transformations(self, feature, original, transformed):
        """
        Visualiza las distribuciones original y transformada de una característica.

        Args:
            feature (str): Nombre de la característica.
            original (pd.Series): Datos originales.
            transformed (pd.Series): Datos transformados.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        sns.histplot(original, kde=True, ax=ax1)
        ax1.set_title(f'Original {feature}')
        ax1.set_xlabel('')
        
        sns.histplot(transformed, kde=True, ax=ax2)
        ax2.set_title(f'Yeo-Johnson Transformed {feature}')
        ax2.set_xlabel('')
        
        plt.tight_layout()
        plt.show()
        
        print(f"Original {feature}:")
        print(f"Skewness: {stats.skew(original):.3f}")
        print(f"Kurtosis: {stats.kurtosis(original):.3f}")
        
        print(f"\nTransformed {feature}:")
        print(f"Skewness: {stats.skew(transformed):.3f}")
        print(f"Kurtosis: {stats.kurtosis(transformed):.3f}")

    def handle_outliers_by_class(self, feature, class_column, lower_percentile=1, upper_percentile=99):
        """
        Maneja outliers por clase utilizando límites de percentiles.

        Args:
            feature (str): Nombre de la característica a procesar.
            class_column (str): Nombre de la columna de clase.
            lower_percentile (int): Percentil inferior para recorte. Por defecto 1.
            upper_percentile (int): Percentil superior para recorte. Por defecto 99.
        """
        self.df_transformed[f'{feature}_capped'] = self.df_transformed[feature]
        
        for class_value in self.df_transformed[class_column].unique():
            class_data = self.df_transformed[self.df_transformed[class_column] == class_value]
            lower_limit = np.percentile(class_data[feature], lower_percentile)
            upper_limit = np.percentile(class_data[feature], upper_percentile)
            
            self.df_transformed.loc[self.df_transformed[class_column] == class_value, f'{feature}_capped'] = \
                self.df_transformed.loc[self.df_transformed[class_column] == class_value, feature].clip(lower_limit, upper_limit)

    def plot_class_specific_handling(self, original_feature, capped_feature, class_column):
        """
        Visualiza la distribución de una característica antes y después del manejo de outliers.

        Args:
            original_feature (str): Nombre de la característica original.
            capped_feature (str): Nombre de la característica con outliers manejados.
            class_column (str): Nombre de la columna de clase.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        sns.boxplot(x=class_column, y=original_feature, data=self.df_transformed, ax=ax1)
        ax1.set_title(f'Before: {original_feature}')
        
        sns.boxplot(x=class_column, y=capped_feature, data=self.df_transformed, ax=ax2)
        ax2.set_title(f'After: {capped_feature}')
        
        plt.tight_layout()
        plt.show()

    def preprocess_data(self):
        """
        Ejecuta el pipeline completo de preprocesamiento.

        Aplica transformación Yeo-Johnson y manejo de outliers a todas las
        características sesgadas identificadas.
        """
        self.df_transformed = self.df.copy()
        self.df_transformed[self.skewed_features] = self.pt.fit_transform(self.df[self.skewed_features])

        for feature in self.skewed_features:
            self.plot_transformations(feature, self.df[feature], self.df_transformed[feature])
            self.handle_outliers_by_class(feature, 'Class')
            self.plot_class_specific_handling(feature, f'{feature}_capped', 'Class')
            self.df_transformed[feature] = self.df_transformed[f'{feature}_capped']
            self.df_transformed = self.df_transformed.drop(f'{feature}_capped', axis=1)

    def save_processed_data(self):
        """
        Guarda los datos procesados en un archivo CSV.
        """
        self.df_transformed.to_csv(self.processed_data_path, index=False)
        print(f"Preprocessing completed. Fully processed data saved to '{self.processed_data_path}'")

    def print_summary_statistics(self):
        """
        Imprime estadísticas descriptivas de los datos procesados.
        """
        print("\nSummary statistics of processed data:")
        print(self.df_transformed.describe())

    def check_remaining_skewness_kurtosis(self):
        """
        Verifica características que aún mantienen alto sesgo o curtosis
        después del preprocesamiento.
        """
        skewness = self.df_transformed.drop('Class', axis=1).skew()
        kurtosis = self.df_transformed.drop('Class', axis=1).kurtosis()

        print("\nFeatures with remaining high skewness (>1 or <-1):")
        print(skewness[abs(skewness) > 1])

        print("\nFeatures with remaining high kurtosis (>7):")
        print(kurtosis[kurtosis > 7])

class DataFrameAnalyzer:
    """
    Clase para análisis exploratorio básico de DataFrames.

    Attributes:
        df (pd.DataFrame): DataFrame a analizar.
    """

    def __init__(self, df):
        """
        Args:
            df (pd.DataFrame): DataFrame para análisis.
        """
        self.df = df

    def print_info(self):
        """
        Imprime información general del DataFrame.
        """
        print("DataFrame Information:")
        self.df.info()

    def print_class_distribution(self, class_column):
        """
        Muestra la distribución de clases en el dataset.

        Args:
            class_column (str): Nombre de la columna de clase.
        """
        if class_column in self.df.columns:
            print("\nClass distribution:")
            print(self.df[class_column].value_counts())
        else:
            print(f"Error: La columna '{class_column}' no existe en el DataFrame.")

class DataStatistics:
    """
    Clase para análisis estadístico de datos.

    Attributes:
        df (pd.DataFrame): DataFrame para análisis estadístico.
    """

    def __init__(self, df):
        """
        Args:
            df (pd.DataFrame): DataFrame para análisis.
        """
        self.df = df

    def print_summary_statistics(self):
        """
        Imprime estadísticas descriptivas del DataFrame.
        """
        print("\nSummary statistics:")
        print(self.df.describe())

class PCAVisualizer:
    """
    Clase para visualización de datos mediante PCA.

    Realiza reducción de dimensionalidad y visualización de datos mediante
    Análisis de Componentes Principales.

    Attributes:
        df (pd.DataFrame): DataFrame con los datos.
        target_column (str): Nombre de la columna objetivo.
        n_components (int): Número de componentes principales a calcular.
        X (pd.DataFrame): Matriz de características.
        y (pd.Series): Variable objetivo.
        pca_model (PCA): Modelo PCA ajustado.
        X_scaled (np.array): Datos escalados.
        X_pca (np.array): Datos transformados por PCA.
        pca_df (pd.DataFrame): DataFrame con resultados de PCA.
    """

    def __init__(self, df, target_column, n_components=2):
        """
        Args:
            df (pd.DataFrame): DataFrame con los datos.
            target_column (str): Nombre de la columna objetivo.
            n_components (int, optional): Número de componentes principales. Por defecto 2.
        """
        self.df = df
        self.target_column = target_column
        self.n_components = n_components
        self.X = df.drop(target_column, axis=1)
        self.y = df[target_column]
        self.pca_model = None
        self.X_scaled = None
        self.X_pca = None
        self.pca_df = None

    def scale_features(self):
        """
        Escala las características usando StandardScaler.
        """
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(self.X)

    def apply_pca(self):
        """
        Aplica PCA a los datos escalados.
        """
        self.pca_model = PCA(n_components=self.n_components)
        self.X_pca = self.pca_model.fit_transform(self.X_scaled)
        self.pca_df = pd.DataFrame(data=self.X_pca, columns=[f'PC{i+1}' for i in range(self.n_components)])
        self.pca_df[self.target_column] = self.y

    def plot_pca(self):
        """
        Visualiza los resultados del PCA en un gráfico de dispersión.
        """
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x='PC1', y='PC2', hue=self.target_column, data=self.pca_df)
        plt.title(f'PCA of {self.target_column} Features')
        plt.show()

    def print_explained_variance(self):
        """
        Imprime el ratio de varianza explicada por cada componente principal.
        """
        print("\nExplained variance ratio:")
        print(self.pca_model.explained_variance_ratio_)

    def run(self):
        """
        Ejecuta el pipeline completo de PCA: escalado, transformación y visualización.
        """
        self.scale_features()
        self.apply_pca()
        self.plot_pca()
        self.print_explained_variance()
