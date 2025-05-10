import pandas as pd
from BaseDatos_Cuencas import BaseDatos_Cuencas as BaseDatos

class AnalisisDatos_Cuencas(BaseDatos):
    
    #Constructor de la Clase AnalisisDatos
    def __init__(self, url):
        super().__init__(url)
        self.__filas = self.tamano[0]
        self.__columnas = self.tamano[1]
        
    #Método String    
    def __str__(self):
        return f"Base de Datos en Formato Pandas para aplicar métodos de análisis de datos"
    
    #Getters
    @property
    def filas(self):
        return self.__filas
    
    @property
    def columnas(self):
        return self.__columnas
    
    #Método para detectar Valores Nulos
    def nulos(self):
        df = self.datos
        resumen = {}
        
        for col in df.columns:
            indices = df[df[col].isnull()].index.tolist()
        
            if indices:
                resumen[col] = {
                    "Índices": indices,
                    "Cantidad": len(indices)}
            
        return resumen
    
    #Método para Clasificar Columnas
    def est_basicas(self):
        cuantitativas = self.datos.select_dtypes(include=["number"]).columns.tolist()
        resumen = {}
        
        for col in cuantitativas:
            resumen[col] = {
                "min" :self.datos[col].describe()["min"],
                "q1" :self.datos[col].describe()["25%"],
                "q2" :self.datos[col].describe()["50%"],
                "q3" :self.datos[col].describe()["75%"],
                "max" :self.datos[col].describe()["max"]}
            
        return resumen
    
    #Método para Detectar Outliers
    
    def detectar_outliers(self):
        cuantitativas = self.datos.select_dtypes(include=["number"]).columns.tolist()
        est_bas = self.est_basicas()
        resumen = {}
        
        for col in cuantitativas:
            q1 = est_bas[col]["q1"]
            q3 = est_bas[col]["q2"]
            iqr = q3-q1
            liminf = q1-1.5*iqr
            limsup = q3+1.5*iqr
            val = self.datos[(self.datos[col]<liminf)|(self.datos[col]>limsup)]
            val = pd.DataFrame(val)
            
            if not val.empty:
                indices = val.index.to_list()
                resumen[col] = {
                    "Índices": indices,
                    "Cantidad": len(indices)}
        
        return resumen
        