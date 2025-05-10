import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class ModeloARIMA:
    def __init__(self, url):
        """
        Constructor de la clase ModeloARIMA.
        
        :param url: Ruta del archivo de datos (Excel o CSV).
        """
        self.data = pd.read_csv(url)  # Cargar datos desde un archivo Excel
        self.model = None
        self.results = None
        self.forecast = None

    def visualizar_datos(self):
        """
        Método para visualizar las columnas relevantes del conjunto de datos.
        """
        visual = self.data[["caudal", "temperatura", "datetime", "precipitacion", "viento", "nino", "humedad"]]
        visual.plot(figsize=(12, 6))
        plt.title("Visualización de Datos")
        plt.show()

    def test_estacionaridad(self, column):
        """
        Método para realizar el test de Dickey-Fuller aumentado.
        
        :param column: Nombre de la columna a evaluar.
        :return: Resultados del test.
        """
        result = adfuller(self.data[column])
        print('ADF Statistic:', result[0])
        print('p-value:', result[1])
        print('Critical Values:', result[4])
        return result

    def diferenciar(self, column):
        """
        Método para diferenciar la serie temporal.
        
        :param column: Nombre de la columna a diferenciar.
        :return: Serie diferenciada.
        """
        serie_diff = self.data[column].diff().dropna()
        plt.figure(figsize=(12, 6))
        plt.plot(serie_diff)
        plt.title('Serie Temporal Diferenciada (Estacionaria)')
        plt.show()
        return serie_diff

    def ajustar_modelo(self, column, order=(6, 1, 1)):
        """
        Método para ajustar el modelo ARIMA.
        
        :param column: Nombre de la columna a modelar.
        :param order: Orden del modelo ARIMA (p, d, q).
        """
        self.model = ARIMA(self.data[column], order=order)
        self.results = self.model.fit()
        print(self.results.summary())

    def evaluar_modelo(self, y):
        """
        Método para evaluar el modelo ajustado.
        
        :param y: Valores reales para la evaluación.
        """
        start = 0
        end = len(y) - 1
        predictions = self.results.predict(start=start, end=end)
        
        rmse = np.sqrt(mean_squared_error(y[start:end + 1], predictions))
        mae = mean_absolute_error(y[start:end + 1], predictions)
        r2 = r2_score(y[start:end + 1], predictions)
        
        print(f"\nRMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"R²: {r2:.2f}")

    def pronosticar(self, steps=12):
        """
        Método para realizar pronósticos con el modelo ARIMA ajustado.
        
        :param steps: Número de pasos a pronosticar.
        """
        forecast = self.results.get_forecast(steps=steps)
        forecast_mean = forecast.predicted_mean
        conf_int = forecast.conf_int()

        # Graficar resultados
        plt.figure(figsize=(12, 6))
        plt.plot(self.data['caudal'], label='Datos reales')
        plt.plot(forecast_mean, label='Pronóstico', color='red')
        plt.fill_between(conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.3)
        plt.title('Pronóstico ARIMA')
        plt.legend()
        plt.show()