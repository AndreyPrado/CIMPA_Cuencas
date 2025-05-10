import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

class ModeloRandomForest_Cuencas:
    def __init__(self, url, target_column, col_ignore):
        """
        Constructor de la clase ModeloRandomForest.
        
        :param url: Ruta del archivo de datos (Excel o CSV).
        :param target_column: Nombre de la columna objetivo.
        """
        self.datos = pd.read_csv(url).fillna(method = "bfill")
        self.target_column = target_column
        self.col_ignore = col_ignore
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None

    def preparar_datos(self):
        """
        Método para dividir los datos en conjuntos de entrenamiento y prueba.
        """
        X = self.datos.drop(columns=[self.target_column, self.col_ignore])
        y = self.datos[self.target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    def ajustar_modelo(self):
        """
        Método para ajustar el modelo Random Forest.
        """
        self.model.fit(self.X_train, self.y_train)

    def evaluar_modelo(self):
        """
        Método para evaluar el modelo ajustado.
        """
        r2_train = self.model.score(self.X_train, self.y_train)  # R² en datos de entrenamiento
        r2_test = self.model.score(self.X_test, self.y_test)    # R² en datos de prueba
        
        print(f"R² (train): {r2_train:.4f}")
        print(f"R² (test): {r2_test:.4f}")

        # Validación cruzada
        scores = cross_val_score(self.model, self.X_train, self.y_train, cv=5, scoring='r2')
        print(f"R² promedio (CV): {scores.mean():.4f}")

        # Predicciones
        self.y_pred = self.model.predict(self.X_test)

        # Calcular métricas de error
        mae = mean_absolute_error(self.y_test, self.y_pred)
        mse = mean_squared_error(self.y_test, self.y_pred)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(self.y_test, self.y_pred) * 100  # En porcentaje

        print(f"""
        Métricas de Error:
        - MAE (Error Absoluto Medio): {mae:.2f}
        - MSE (Error Cuadrático Medio): {mse:.2f}
        - RMSE (Raíz del Error Cuadrático Medio): {rmse:.2f}
        - MAPE (Error Porcentual Absoluto Medio): {mape:.2f}%
        """)

    def visualizar_resultados(self):
        """
        Método para visualizar los resultados de las predicciones.
        """
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=self.y_test, y=self.y_pred, alpha=0.6)
        plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--')  # Línea de perfecta predicción
        plt.title('Valores Reales vs Predicciones')
        plt.xlabel('Valores Reales')
        plt.ylabel('Predicciones')
        plt.grid(True)
        plt.show()