import matplotlib.pyplot as plt
import seaborn as sns
from BaseDatos_Cuencas import BaseDatos_Cuencas as BaseDatos

class Grafico_Cuencas(BaseDatos):
    
    def __init__(self, url):
        super().__init__(url)
        self.__grafico = None
        
    #Getter
    @property
    def grafico(self):
        return self.__grafico
    
    #Método String
    def __str__(self):
        return "Gráficos para la visualización de datos"
    

    #Método para graficar un boxplot    
    def boxplot(self, col = None):
        df = self.datos
        
        if col is None:
            col = df.select_dtypes(include=["number"]).columns.tolist()
            
        fig, ax = plt.subplots(figsize=(10,5))
        sns.boxplot(data = df[col], ax=ax)
        ax.set_title("Gráfico de Cajas")
        plt.tight_layout()
        
        self.__grafico = fig
        
        return fig
    
    #Método para hacer un gráfico de líneas
    def linea(self, col = None):
        df = self.datos
        
        if col is None:
            col = df.select_dtypes(include=["number"]).columns.tolist()
            
        fig, ax = plt.subplots(figsize=(10,5))
        sns.lineplot(data=df[col], ax=ax)
        ax.set_title("Gráfico de Líneas")
        plt.tight_layout()
        
        self.__grafico = fig
        
        return fig
    
    #Método para hacer un heatmap
    def heatmap(self):
       df = self.datos.select_dtypes(include=["number"])
       corr = df.corr()
       
       fig, ax = plt.subplots(figsize=(8,6))
       sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
       ax.set_title("Mapa de Calor de Correlaciones")
       plt.tight_layout()
       
       self.__grafico = fig
       
       return fig
   
    #Método para graficas las distribuciones
    def dist(self, col):
        df = self.datos
        
        if col not in df.columns:
            raise ValueError(f"La columna '{col}' no está en el DataFrame")
        else:
            fig, ax = plt.subplots(figsize=(8,5))
            sns.histplot(df[col], kde = True, ax=ax)
            ax.set_title(f"Distribución de {col}")
            plt.tight_layout()
            
            self.__grafico = fig
            
            return fig
        
    #Método para guardar los gráficos en png
    def guardar_en_png(self, nombre: str):
        if self.grafico is None:
            return "No se ha generado ningún gráfico"
        else:
            self.grafico.savefig(nombre+".png")
        
        return f"Gráfico guardado como {nombre}.png"