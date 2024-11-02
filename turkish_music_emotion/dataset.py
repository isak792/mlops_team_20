from pathlib import Path
from loguru import logger
import pandas as pd
import yaml
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from turkish_music_emotion.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, INTERIM_DATA_DIR

class DataHandler:
    """
    Clase para manejar operaciones de carga, procesamiento y guardado de datos.

    Esta clase proporciona funcionalidades para el manejo completo del ciclo de vida
    de datos, incluyendo carga desde CSV, procesamiento, escalado y guardado.

    Attributes:
        data (pd.DataFrame): DataFrame que contiene los datos cargados.
        le (LabelEncoder): Instancia de LabelEncoder para codificación de etiquetas.
    """

    def __init__(self):
        """
        Inicializa una nueva instancia de DataHandler.

        El DataFrame data se inicializa como None y se creará al cargar los datos.
        """
        self.data = None
        self.le = LabelEncoder()

    def load_data(self, input_path: Path, input_filename: str) -> pd.DataFrame:
        """
        Carga datos desde un archivo CSV.

        Args:
            input_path (Path): Ruta del directorio que contiene el archivo.
            input_filename (str): Nombre del archivo CSV a cargar.

        Returns:
            pd.DataFrame: DataFrame con los datos cargados.

        Examples:
            >>> handler = DataHandler()
            >>> data = handler.load_data(Path("data/raw"), "dataset.csv")
        """
        full_input_path = input_path / input_filename
        logger.info(f"Loading data from {full_input_path}")
        
        self.data = pd.read_csv(full_input_path)
        logger.success(f"Data loaded successfully from {full_input_path}")
        return self.data

    def save_data(self, output_dir: Path, filename: str) -> None:
        """
        Guarda el DataFrame actual en un archivo CSV.

        Args:
            output_dir (Path): Directorio donde se guardará el archivo.
            filename (str): Nombre del archivo de salida.

        Returns:
            None

        Raises:
            Warning: Si no hay datos cargados en el DataFrame.

        Examples:
            >>> handler = DataHandler()
            >>> handler.save_data(Path("data/processed"), "processed_data.csv")
        """
        if self.data is None:
            logger.warning("No data to save. Load the data first.")
            return
        
        output_path = output_dir / filename
        logger.info(f"Saving data to {output_path}")
        
        self.data.to_csv(output_path, index=False)
        logger.success(f"Data saved successfully to {output_path}")

    def process_data(self) -> pd.DataFrame:
        """
        Procesa los datos eliminando valores faltantes y realiza análisis de missing values.

        Returns:
            pd.DataFrame: DataFrame procesado sin valores faltantes.

        Raises:
            Warning: Si no hay datos cargados para procesar.

        Examples:
            >>> handler = DataHandler()
            >>> processed_data = handler.process_data()
        """
        if self.data is None:
            logger.warning("No data loaded. Load data before processing.")
            return
        
        analyzer = MissingValueAnalyzer(self.data)
        missing_columns = analyzer.get_missing_columns()
        analyzer.display_missing_columns()
        
        self.data.dropna(inplace=True)
        logger.success("Data processed successfully.")
        return self.data
    
    def scale_data(self) -> pd.DataFrame:
        """
        Escala las columnas numéricas del DataFrame usando StandardScaler.

        Returns:
            pd.DataFrame: DataFrame con las columnas numéricas escaladas.

        Raises:
            Warning: Si no hay datos cargados para escalar.

        Examples:
            >>> handler = DataHandler()
            >>> scaled_data = handler.scale_data()
        """
        if self.data is None:
            logger.warning("No data loaded. Load data before scaling.")
            return
        scaler = StandardScaler()
        numeric_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
        self.data[numeric_cols] = scaler.fit_transform(self.data[numeric_cols])
        logger.success("Data scaled successfully.")
        return self.data

    def run(self, input_path: Path, input_filename: str, output_filename: str) -> 'DataHandler':
        """
        Ejecuta el pipeline completo de procesamiento de datos.

        Este método ejecuta secuencialmente la carga, procesamiento, escalado y
        guardado de datos.

        Args:
            input_path (Path): Ruta del directorio de entrada.
            input_filename (str): Nombre del archivo de entrada.
            output_filename (str): Nombre del archivo de salida.

        Returns:
            DataHandler: Instancia actual del DataHandler.

        Examples:
            >>> handler = DataHandler()
            >>> handler.run(Path("data/raw"), "input.csv", "output.csv")
        """
        self.load_data(input_path, input_filename)
        self.process_data()
        self.scale_data()
        self.save_data(PROCESSED_DATA_DIR, output_filename)
        return self


class MissingValueAnalyzer:
    """
    Clase para analizar valores faltantes en un DataFrame.

    Esta clase proporciona métodos para identificar y mostrar información sobre
    columnas que contienen valores faltantes.

    Attributes:
        data (pd.DataFrame): DataFrame a analizar.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Inicializa el analizador de valores faltantes.

        Args:
            data (pd.DataFrame): DataFrame a analizar.
        """
        self.data = data
    
    def get_missing_columns(self) -> pd.Series:
        """
        Identifica las columnas que contienen valores faltantes.

        Returns:
            pd.Series: Series con el conteo de valores faltantes por columna.

        Examples:
            >>> analyzer = MissingValueAnalyzer(df)
            >>> missing_cols = analyzer.get_missing_columns()
        """
        missing_values = self.data.isnull().sum()
        columns_with_missing = missing_values[missing_values > 0]
        return columns_with_missing
    
    def display_missing_columns(self) -> None:
        """
        Muestra en consola las columnas con valores faltantes.

        Si no hay valores faltantes, muestra un mensaje indicándolo.

        Examples:
            >>> analyzer = MissingValueAnalyzer(df)
            >>> analyzer.display_missing_columns()
        """
        columns_with_missing = self.get_missing_columns()
        if columns_with_missing.empty:
            print("No hay columnas con valores faltantes.")
        else:
            print("Columnas con valores faltantes:")
            print(columns_with_missing)


class LabelEncoderWrapper:
    """
    Wrapper para LabelEncoder que facilita la codificación de múltiples columnas.

    Esta clase proporciona métodos para codificar y decodificar columnas categóricas,
    manteniendo un registro de los encoders utilizados.

    Attributes:
        data (pd.DataFrame): DataFrame con los datos a codificar.
        label_encoders (dict): Diccionario que mapea nombres de columnas a sus LabelEncoders.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Inicializa el wrapper de LabelEncoder.

        Args:
            data (pd.DataFrame): DataFrame con los datos a codificar.
        """
        self.data = data
        self.label_encoders = {}

    def encode_column(self, column: str) -> None:
        """
        Codifica una columna categórica específica.

        Args:
            column (str): Nombre de la columna a codificar.

        Raises:
            ValueError: Si la columna especificada no existe en el DataFrame.

        Examples:
            >>> encoder = LabelEncoderWrapper(df)
            >>> encoder.encode_column('categoria')
        """
        if column not in self.data.columns:
            raise ValueError(f"La columna '{column}' no existe en el DataFrame.")
        
        label_encoder = LabelEncoder()
        self.data[column] = label_encoder.fit_transform(self.data[column])
        self.label_encoders[column] = label_encoder
        
    def get_encoded_data(self) -> pd.DataFrame:
        """
        Obtiene el DataFrame con las columnas codificadas.

        Returns:
            pd.DataFrame: DataFrame con las columnas codificadas.

        Examples:
            >>> encoder = LabelEncoderWrapper(df)
            >>> encoded_df = encoder.get_encoded_data()
        """
        return self.data
    
    def decode_column(self, column: str) -> None:
        """
        Decodifica una columna previamente codificada.

        Args:
            column (str): Nombre de la columna a decodificar.

        Raises:
            ValueError: Si la columna no fue codificada previamente.

        Examples:
            >>> encoder = LabelEncoderWrapper(df)
            >>> encoder.decode_column('categoria')
        """
        if column not in self.label_encoders:
            raise ValueError(f"La columna '{column}' no fue codificada o no se ha guardado el codificador.")
        
        label_encoder = self.label_encoders[column]
        self.data[column] = label_encoder.inverse_transform(self.data[column])


if __name__ == "__main__":
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)

    handler = DataHandler()
    handler.run(RAW_DATA_DIR, params['data']['input_filename'], params['data']['output_filename'])
