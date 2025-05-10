import pandas as pd

class BaseDatos_Cuencas():
    
    #Constructor de la Clase
    #Se asume que la base de datos ya está limpia
    def __init__(self, url: str):
        self.__url = url
        self.__datos = pd.read_csv(self.__url).fillna(method = "bfill")
        self.__tamano = self.__datos.shape
    
    #Getters
    @property
    def datos(self):
        return self.__datos
    
    @property
    def tamano(self):
        return self.__tamano
    
    @property
    def url(self):
        return self.__url
        
    #Setters
    @datos.setter
    def url(self, new_str):
        self.__datos = pd.DataFrame(new_str)
            
    #Método String
    def __str__(self):
        return f"Base de Datos de dimensiones {self.tamano} \ny valores {self.datos}"
    
    #Método Para Descargar Pandas.DataFrame en CSV
    def descargar_csv(self, nombre: str):
        cadena = nombre+".csv"
        self.__datos.to_csv(cadena)
        return f"Archivo {nombre}.csv descargado con éxito"
        
    
