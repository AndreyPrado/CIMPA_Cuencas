import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import matplotlib.pyplot as plt
from BaseDatos_Cuencas import BaseDatos_Cuencas as BaseDatos
from Grafico_Cuencas import Grafico_Cuencas as Grafico

class Modelo_Cuencas(BaseDatos):
    
    #Constructor de la Clase
    def __init__(self, url):
        super().__init__(url)
        self.__prediccion = None
        
    #Getters
    @property
    def prediccion(self):
        return self.__prediccion
    
    #Setters
    @prediccion.setter
    def prediccion(self, new_value):
        self.__prediccion = new_value
    

    #Método String
    def __str__(self):
        return f"Objeto Tipo Modelo"
    
    #Método para cargar las columnas
    def cargar_datos(self, x: list, y:list):
        X = self.datos.drop(columns = x)
        Y = self.datos[y]
        
        X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size= 0.8, random_state = 42)
        
        return X_train, X_test, Y_train, Y_test
    
    #Método para hallar la correlación más alta
    def top_corr(self, percentil):
        matriz = self.datos.select_dtypes(include = "number").corr()
        valores = matriz.values
        columns = matriz.columns
        index = matriz.index
        
        np.fill_diagonal(valores, np.nan)
        threshold = np.nanpercentile(valores, percentil)
        high_corr = np.argwhere(valores >= threshold)
        
        high_corr = high_corr[high_corr[:,0]<high_corr[:,1]]
        
        if high_corr.size >0:

            sorted_indices = np.argsort(-valores[high_corr[:,0], high_corr[:,1]])
            high_corr = high_corr[sorted_indices]
        
        print(f"Correlaciones >= {threshold:.2f} (percentil {percentil}):")
        for row, col in high_corr:
            print(f"{columns[row]} vs {index[col]} -> {valores[row,col]:.2f}")
        
        
    #Método para laguear las variables
    def lag(self, cantidad: int, nombre: str):
        nombre_final = nombre+"_lag_"+str(cantidad)
        self.datos[nombre_final] = self.datos[nombre].shift(cantidad)
        
        return self.datos
        