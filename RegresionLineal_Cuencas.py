import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from BaseDatos import BaseDatos
from Modelo import Modelo
from Grafico import Grafico

class RegresionLineal(Modelo, Grafico):
    
    # Constructor de la clase
    def __init__(self, url):
        super().__init__(url)
        self.__modelo_sklearn = None
        self.__modelo_statsmodels = None
        self.__X_train = None
        self.__X_test = None
        self.__y_train = None
        self.__y_test = None
        self.__X_train_scaled = None
        self.__X_test_scaled = None
        self.__predicciones = None
        self.__metricas = {}
        self.__scaler = StandardScaler()
        self.__coeficientes = None
        self.__p_values = None
        
    # Getters
    @property
    def modelo_sklearn(self):
        return self.__modelo_sklearn
    
    @property
    def modelo_statsmodels(self):
        return self.__modelo_statsmodels
    
    @property
    def metricas(self):
        return self.__metricas
    
    @property
    def coeficientes(self):
        return self.__coeficientes
    
    @property
    def p_values(self):
        return self.__p_values
    
    # Método String
    def __str__(self):
        return "Modelo de Regresión Lineal para predicción"
    
    # Método para dividir los datos
    def dividir_datos(self, variable_y, excluir_cols=None, test_size=0.2, random_state=42):
        """
        Divide los datos en conjuntos de entrenamiento y prueba
        
        Args:
            variable_y (str): Nombre de la columna de la variable dependiente
            excluir_cols (list): Lista de columnas a excluir (además de la variable_y)
            test_size (float): Proporción de datos para prueba
            random_state (int): Semilla para reproducibilidad
            
        Returns:
            Conjuntos X_train, X_test, y_train, y_test
        """
        # Verificar que la variable dependiente existe
        if variable_y not in self.datos.columns:
            raise ValueError(f"La columna '{variable_y}' no existe en los datos")
        
        # Preparar la lista de columnas a excluir
        if excluir_cols is None:
            excluir_cols = []
        
        # Añadir la variable dependiente a las columnas a excluir
        columnas_excluir = excluir_cols + [variable_y]
        
        # Asegurar que las columnas a excluir existen
        for col in columnas_excluir:
            if col not in self.datos.columns:
                raise ValueError(f"La columna '{col}' no existe en los datos")
        
        # Resetear índices para evitar problemas de alineación
        datos_reset = self.datos.reset_index(drop=True)
        
        # Crear conjuntos X e y
        X = datos_reset.drop(columns=columnas_excluir)
        y = datos_reset[variable_y]
        
        # Dividir datos
        self.__X_train, self.__X_test, self.__y_train, self.__y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"Datos divididos en conjuntos de entrenamiento ({len(self.__X_train)} registros) "
              f"y prueba ({len(self.__X_test)} registros)")
        
        return self.__X_train, self.__X_test, self.__y_train, self.__y_test
    
    # Método para escalar los datos
    def escalar_datos(self):
        """
        Escala los datos usando StandardScaler
        
        Returns:
            DataFrames escalados X_train y X_test
        """
        if self.__X_train is None or self.__X_test is None:
            raise ValueError("Primero debes dividir los datos usando el método dividir_datos()")
        
        # Escalar los datos
        self.__X_train_scaled = pd.DataFrame(
            self.__scaler.fit_transform(self.__X_train),
            columns=self.__X_train.columns,
            index=self.__X_train.index
        )
        
        self.__X_test_scaled = pd.DataFrame(
            self.__scaler.transform(self.__X_test),
            columns=self.__X_test.columns,
            index=self.__X_test.index
        )
        
        print("Datos escalados correctamente")
        
        return self.__X_train_scaled, self.__X_test_scaled
    
    # Método para entrenar modelo con sklearn
    def entrenar_sklearn(self, fit_intercept=True, n_jobs=None):
        """
        Entrena un modelo de regresión lineal usando sklearn
        
        Args:
            fit_intercept (bool): Si se debe calcular el intercepto
            n_jobs (int): Número de jobs para computación paralela
            
        Returns:
            El modelo entrenado
        """
        if self.__X_train_scaled is None:
            raise ValueError("Primero debes escalar los datos usando el método escalar_datos()")
        
        # Crear y entrenar el modelo
        self.__modelo_sklearn = LinearRegression(fit_intercept=fit_intercept, n_jobs=n_jobs)
        self.__modelo_sklearn.fit(self.__X_train_scaled, self.__y_train)
        
        # Guardar coeficientes
        self.__coeficientes = pd.DataFrame({
            'Variable': self.__X_train_scaled.columns,
            'Coeficiente': self.__modelo_sklearn.coef_
        })
        
        print("Modelo sklearn entrenado correctamente")
        
        return self.__modelo_sklearn
    
    # Método para entrenar modelo con statsmodels
    def entrenar_statsmodels(self, add_constant=True):
        """
        Entrena un modelo de regresión lineal usando statsmodels
        
        Args:
            add_constant (bool): Si se debe añadir una constante (intercepto)
            
        Returns:
            El modelo entrenado
        """
        if self.__X_train_scaled is None:
            raise ValueError("Primero debes escalar los datos usando el método escalar_datos()")
        
        # Añadir constante si es necesario
        X_train_sm = self.__X_train_scaled
        if add_constant:
            X_train_sm = sm.add_constant(X_train_sm)
        
        # Verificar alineación de índices
        if not all(X_train_sm.index == self.__y_train.index):
            raise ValueError("Los índices de X_train y y_train no están alineados")
        
        # Entrenar modelo
        self.__modelo_statsmodels = sm.OLS(self.__y_train, X_train_sm).fit()
        
        # Guardar coeficientes y p-values
        self.__coeficientes = pd.DataFrame({
            'Variable': self.__modelo_statsmodels.params.index,
            'Coeficiente': self.__modelo_statsmodels.params.values,
            'P-valor': self.__modelo_statsmodels.pvalues.values
        })
        
        self.__p_values = self.__modelo_statsmodels.pvalues
        
        print("Modelo statsmodels entrenado correctamente")
        
        return self.__modelo_statsmodels
    
    # Método para hacer predicciones
    def predecir(self, usar_statsmodels=False, nuevos_datos=None):
        """
        Realiza predicciones con el modelo entrenado
        
        Args:
            usar_statsmodels (bool): Si se deben usar el modelo de statsmodels
            nuevos_datos (DataFrame): Datos nuevos para predicción (si None, se usan los de prueba)
            
        Returns:
            Las predicciones
        """
        # Verificar que al menos un modelo está entrenado
        if self.__modelo_sklearn is None and self.__modelo_statsmodels is None:
            raise ValueError("Primero debes entrenar al menos un modelo")
        
        # Determinar datos para predicción
        if nuevos_datos is not None:
            # Escalar nuevos datos
            X_pred = pd.DataFrame(
                self.__scaler.transform(nuevos_datos),
                columns=nuevos_datos.columns
            )
        else:
            # Usar conjunto de prueba
            X_pred = self.__X_test_scaled
        
        # Realizar predicciones
        if usar_statsmodels and self.__modelo_statsmodels is not None:
            # Añadir constante si es necesario
            if 'const' in self.__modelo_statsmodels.params.index:
                X_pred_sm = sm.add_constant(X_pred)
            else:
                X_pred_sm = X_pred
            
            self.__predicciones = self.__modelo_statsmodels.predict(X_pred_sm)
        else:
            # Usar modelo sklearn
            if self.__modelo_sklearn is None:
                raise ValueError("Primero debes entrenar el modelo sklearn")
            
            self.__predicciones = self.__modelo_sklearn.predict(X_pred)
        
        # Guardar predicciones en la clase padre
        self.prediccion = self.__predicciones
        
        return self.__predicciones
    
    # Método para evaluar el modelo
    def evaluar_modelo(self, y_real=None):
        """
        Evalúa el rendimiento del modelo usando diferentes métricas
        
        Args:
            y_real (Series): Valores reales para comparar (si None, se usan los de prueba)
            
        Returns:
            Diccionario con las métricas calculadas
        """
        if self.__predicciones is None:
            raise ValueError("Primero debes hacer predicciones con el método predecir()")
        
        # Determinar valores reales
        if y_real is None:
            y_real = self.__y_test
        
        # Calcular métricas
        rmse = np.sqrt(mean_squared_error(y_real, self.__predicciones))
        mae = mean_absolute_error(y_real, self.__predicciones)
        r2 = r2_score(y_real, self.__predicciones)
        
        # Guardar métricas
        self.__metricas = {
            'R²': r2,
            'RMSE': rmse,
            'MAE': mae
        }
        
        # Imprimir métricas
        print(f"R²: {r2:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        
        return self.__metricas
    
    # Método para mostrar resumen del modelo statsmodels
    def mostrar_resumen(self):
        """
        Muestra el resumen completo del modelo statsmodels
        
        Returns:
            Resumen del modelo
        """
        if self.__modelo_statsmodels is None:
            raise ValueError("Primero debes entrenar el modelo statsmodels")
        
        return self.__modelo_statsmodels.summary()
    
    # Método para graficar coeficientes
    def graficar_coeficientes(self, top_n=None, excluir_constante=True):
        """
        Genera un gráfico de barras con los coeficientes del modelo
        
        Args:
            top_n (int): Número de coeficientes a mostrar (los más importantes)
            excluir_constante (bool): Si se debe excluir la constante del gráfico
            
        Returns:
            Figura con el gráfico
        """
        if self.__coeficientes is None:
            raise ValueError("Primero debes entrenar un modelo")
        
        # Crear copia de coeficientes para no modificar el original
        coefs = self.__coeficientes.copy()
        
        # Excluir constante si es necesario
        if excluir_constante and 'const' in coefs['Variable'].values:
            coefs = coefs[coefs['Variable'] != 'const']
        
        # Ordenar por valor absoluto
        coefs['Abs_Coef'] = np.abs(coefs['Coeficiente'])
        coefs = coefs.sort_values('Abs_Coef', ascending=False)
        
        # Limitar a top_n si es necesario
        if top_n is not None and top_n < len(coefs):
            coefs = coefs.head(top_n)
        
        # Crear gráfico
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Definir colores según signo
        colors = ['red' if x < 0 else 'blue' for x in coefs['Coeficiente']]
        
        # Graficar barras
        ax.barh(coefs['Variable'], coefs['Coeficiente'], color=colors)
        
        # Añadir línea vertical en cero
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Añadir etiquetas y título
        ax.set_xlabel('Coeficiente')
        ax.set_ylabel('Variable')
        ax.set_title('Coeficientes del Modelo de Regresión Lineal')
        
        # Añadir valores en las barras
        for i, v in enumerate(coefs['Coeficiente']):
            ax.text(v + (0.01 if v >= 0 else -0.01), 
                    i, 
                    f"{v:.4f}", 
                    va='center', 
                    ha='left' if v >= 0 else 'right',
                    fontweight='bold')
        
        # Ajustar layout
        plt.tight_layout()
        
        # Guardar referencia al gráfico
        self._Grafico__grafico = fig  # Acceso al atributo privado de la clase Grafico
        
        return fig
    
    # Método para graficar valores reales vs predicciones
    def graficar_predicciones(self, columna_fecha=None):
        """
        Genera un gráfico comparando valores reales vs predicciones
        
        Args:
            columna_fecha (str): Nombre de la columna con fechas (si None, se usa el índice)
            
        Returns:
            Figura con el gráfico
        """
        if self.__predicciones is None:
            raise ValueError("Primero debes hacer predicciones con el método predecir()")
        
        # Crear DataFrame con valores reales y predicciones
        resultados = pd.DataFrame({
            'Real': self.__y_test,
            'Predicción': self.__predicciones
        })
        
        # Añadir columna de fecha si está disponible
        if columna_fecha is not None and columna_fecha in self.datos.columns:
            # Obtener fechas correspondientes a los índices del conjunto de prueba
            fechas = self.datos.loc[self.__y_test.index, columna_fecha]
            resultados['Fecha'] = fechas
            
            # Ordenar por fecha
            resultados = resultados.sort_values('Fecha')
            x_axis = resultados['Fecha']
            x_label = 'Fecha'
        else:
            # Usar índice numérico
            resultados = resultados.reset_index(drop=True)
            x_axis = resultados.index
            x_label = 'Índice'
        
        # Crear gráfico
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Graficar valores reales y predicciones
        ax.plot(x_axis, resultados['Real'], label='Valores Reales', marker='o', linestyle='-', alpha=0.6)
        ax.plot(x_axis, resultados['Predicción'], label='Predicciones', marker='s', linestyle='--', color='red')
        
        # Añadir etiquetas y título
        ax.set_xlabel(x_label)
        ax.set_ylabel('Valor')
        ax.set_title('Comparación entre Valores Reales y Predicciones')
        
        # Añadir leyenda
        ax.legend()
        
        # Añadir grid
        ax.grid(True, alpha=0.3)
        
        # Si hay muchas fechas, rotar las etiquetas del eje x
        if columna_fecha is not None:
            plt.xticks(rotation=45)
        
        # Ajustar layout
        plt.tight_layout()
        
        # Guardar referencia al gráfico
        self._Grafico__grafico = fig  # Acceso al atributo privado de la clase Grafico
        
        return fig
    
    # Método para graficar dispersión de predicciones
    def graficar_dispersion(self):
        """
        Genera un gráfico de dispersión de valores reales vs predicciones
        
        Returns:
            Figura con el gráfico
        """
        if self.__predicciones is None:
            raise ValueError("Primero debes hacer predicciones con el método predecir()")
        
        # Crear gráfico
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Graficar dispersión
        ax.scatter(self.__y_test, self.__predicciones, alpha=0.6)
        
        # Añadir línea diagonal de referencia
        min_val = min(self.__y_test.min(), self.__predicciones.min())
        max_val = max(self.__y_test.max(), self.__predicciones.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        
        # Añadir etiquetas y título
        ax.set_xlabel('Valores Reales')
        ax.set_ylabel('Predicciones')
        ax.set_title('Gráfico de Dispersión: Valores Reales vs Predicciones')
        
        # Añadir texto con métricas
        if self.__metricas:
            text = f"R² = {self.__metricas['R²']:.4f}\nRMSE = {self.__metricas['RMSE']:.4f}"
            ax.text(0.05, 0.95, text, transform=ax.transAxes, 
                    verticalalignment='top', bbox={'boxstyle': 'round', 'alpha': 0.5})
        
        # Ajustar layout
        plt.tight_layout()
        
        # Guardar referencia al gráfico
        self._Grafico__grafico = fig  # Acceso al atributo privado de la clase Grafico
        
        return fig
    
    # Método para identificar variables más significativas
    def variables_significativas(self, umbral_p=0.05):
        """
        Identifica variables significativas basadas en el p-valor
        
        Args:
            umbral_p (float): Umbral de p-valor para considerar una variable significativa
            
        Returns:
            DataFrame con variables significativas
        """
        if self.__p_values is None:
            raise ValueError("Primero debes entrenar el modelo con statsmodels")
        
        # Filtrar por p-valor
        significativas = self.__coeficientes[self.__coeficientes['P-valor'] < umbral_p]
        
        return significativas.sort_values('P-valor')