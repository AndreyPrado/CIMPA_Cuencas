import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from Modelo_Cuencas import Modelo_Cuencas as Modelo
from Grafico_Cuencas import Grafico_Cuencas as Grafico


class RedesNeuronales_Cuencas(Modelo, Grafico):
    
    #Constructor de la Clase
    def __init__(self, url):
        super().__init__(url)
        self.__modelo = None
        self.__historial = None
        self.__predicciones = None
        self.__x_escalado = None
        self.__y_escalado = None
        self.__x_test = None
        self.__y_test = None
        self.__x_train = None
        self.__y_train = None
        self.__x_val = None
        self.__y_val = None
        self.__variables_x = None
        self.__variables_y = None
        self.__scaler_X = MinMaxScaler()
        self.__scaler_y = MinMaxScaler()
        self.__metricas = {}
        
    #Getters
    @property
    def modelo(self):
        return self.__modelo
    
    @property
    def historial(self):
        return self.__historial
    
    @property
    def predicciones(self):
        return self.__predicciones
    
    @property
    def metricas(self):
        return self.__metricas
    
    #Método String
    def __str__(self):
        return f"Modelo de Red Neuronal para predicción"
    
    #Método para preprocesar datos
    def preprocesar_datos(self, variables_x, variable_y):
        """
        Preprocesa los datos para el entrenamiento de la red neuronal
        
        Parámetros:
            variables_x (list): Lista de columnas para las variables independientes
            variable_y (str): Nombre de la columna para la variable dependiente
            
        Retorna: 
            x_escalado (list): Variables x escaladas
            y_escalado (list): Variables y escaladas
        """
        # Verificar que todas las columnas existen
        columnas = self.datos.columns
        for col in variables_x:
            if col not in columnas:
                raise ValueError(f"La columna '{col}' no existe en el conjunto de datos")
        
        if variable_y not in columnas:
            raise ValueError(f"La columna '{variable_y}' no existe en el conjunto de datos")
        
        self.__variables_x = variables_x
        self.__variables_y = variable_y
        
        # Extraer X e y
        X = self.datos[variables_x]
        y = self.datos[variable_y]
        
        # Escalar los datos
        self.__x_escalado = self.__scaler_X.fit_transform(X)
        self.__y_escalado = self.__scaler_y.fit_transform(y.values.reshape(-1, 1))
        
        test_size = int(len(self.__x_escalado)*0.15)
        self.__x_test = self.__x_escalado[-test_size:]
        self.__y_test = self.__y_escalado[-test_size:]
        
        x_trainval = self.__x_escalado[:-test_size]
        y_trainval = self.__y_escalado[:-test_size]
        
        tscv = TimeSeriesSplit(n_splits = 5)
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(x_trainval)):
            self.__x_train, self.__x_val = x_trainval[train_idx], x_trainval[val_idx]
            self.__y_train, self.__y_val = y_trainval[train_idx] , y_trainval[val_idx]
        
        return self.__x_train, self.__y_train, self.__x_test, self.__y_test, self.__x_val, self.__y_val
    
    #Método para crear la arquitectura de la red neuronal
    def crear_modelo(self, capas=None, learning_rate=0.1, funcion_perdida="mean_squared_error"):
        """
        Crea la arquitectura de la red neuronal
        
        Args:
            capas (list): Lista de tuplas con (num_neuronas, funcion_activacion)
            learning_rate (float): Tasa de aprendizaje
            funcion_perdida (str): Función de pérdida a utilizar
        
        Returns:
            El modelo creado
        """
        if capas is None:
            # Si no se especifican capas, usar una arquitectura por defecto
            # similar a la del notebook
            input_shape = self.__x_escalado.shape[1]
            modelo = tf.keras.Sequential([
                tf.keras.layers.Dense(units=input_shape + 1, input_shape=(input_shape,), activation='relu'),
                tf.keras.layers.Dense(units=input_shape + 1, activation='relu'),
                tf.keras.layers.Dense(units=input_shape + 1, activation='relu'),
                tf.keras.layers.Dense(units=1)
            ])
        else:
            # Construir modelo con las capas especificadas
            modelo = tf.keras.Sequential()
            input_shape = self.__x_escalado.shape[1]
            
            # Primera capa (debe especificar input_shape)
            neuronas, activacion = capas[0]
            modelo.add(tf.keras.layers.Dense(units=neuronas, input_shape=(input_shape,), 
                                            activation=activacion))
            
            # Resto de capas
            for neuronas, activacion in capas[1:]:
                modelo.add(tf.keras.layers.Dense(units=neuronas, activation=activacion))
        
        # Compilar el modelo
        modelo.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            loss=funcion_perdida
        )
        
        self.__modelo = modelo
        return modelo
    
    #Método para entrenar el modelo
    def entrenar(self, patience = 10, epochs=1000, verbose=1):

        if self.__modelo is None:
            raise ValueError("Primero debes crear el modelo con el método crear_modelo()")
        
        if self.__x_escalado is None or self.__y_escalado is None:
            raise ValueError("Primero debes preprocesar los datos con el método preprocesar_datos()")
            
            
        early_stop = tf.keras.callbacks.EarlyStopping(monitor = "val_loss", patience=patience, restore_best_weights = True)
        
        print("Entrenando el modelo...")
        self.__historial = self.__modelo.fit(
            self.__x_train, 
            self.__y_train, 
            validation_data = (self.__x_val, self.__y_val),
            epochs=epochs, 
            batch_size = 32,
            verbose=verbose,
            callbacks=[early_stop]
        )
        print("Modelo entrenado!")
        
        return self.__historial
    
    #Método para graficar el historial de entrenamiento
    def graficar_perdidas(self):
        """
        Gráfica las pérdidas durante el entrenamiento
        """
        if self.__historial is None:
            raise ValueError("Primero debes entrenar el modelo con el método entrenar()")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.__historial.history['loss'], label='Pérdida de entrenamiento')
        
        if 'val_loss' in self.__historial.history:
            ax.plot(self.__historial.history['val_loss'], label='Pérdida de validación')
            
        ax.set_xlabel("Épocas")
        ax.set_ylabel("Pérdida")
        ax.set_title("Evolución de la pérdida durante el entrenamiento")
        ax.legend()
        ax.grid(True)
        
        self._Grafico__grafico = fig  # Acceso al atributo privado de la clase Grafico
        return fig
    
    #Método para hacer predicciones
    def predecir(self):
        """
        Realiza predicciones con el modelo entrenado
        
        Returns:
            Las predicciones en la escala original
        """
        if self.__modelo is None:
            raise ValueError("Primero debes crear y entrenar el modelo")
        
        # Hacer predicciones
        predicciones_escaladas = self.__modelo.predict(self.__x_val)
        
        # Revertir el escalado
        self.__predicciones = self.__scaler_y.inverse_transform(predicciones_escaladas).flatten()
        
        # Guardar también en el atributo de la clase padre
        self.prediccion = self.__predicciones
        
        return self.__predicciones
    
    #Método para evaluar el modelo
    def evaluar_modelo(self, y_real=None):

        if self.__predicciones is None:
            raise ValueError("Primero debes hacer predicciones con el método predecir()")
        
        y_real = self.__modelo.predict(self.__x_test)
        
        y_real = self.__scaler_y.inverse_transform(y_real).flatten()
        
        # Calcular métricas
        rmse = np.sqrt(mean_squared_error(y_real, self.__predicciones))
        mae = mean_absolute_error(y_real, self.__predicciones)
        r2 = r2_score(y_real, self.__predicciones)
        mse = mean_squared_error(y_real, self.__predicciones)
        
        # Guardar métricas
        self.__metricas = {
            'R²': r2,
            'RMSE': rmse,
            'MAE': mae,
            'MSE': mse
        }
        
        # Imprimir métricas
        print(f"R²: {r2:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"MSE: {mse:.4f}")
        
        return self.__metricas
    
    
    def shap(self):
        
        import shap
        
        explainer = shap.DeepExplainer(self.__modelo, self.__x_train[:len(self.__x_train)//2])
        shap_values = explainer.shap_values(self.__x_test[:len(self.__x_train)//2])
        
        shap.summary_plot(shap_values, self.__x_test, plot_type = "bar")
        
        shap.summary_plot(shap_values, self.__x_test)
        
        
    def permutacion(self):
        
        from scikeras.wrappers import KerasRegressor
        from sklearn.inspection import permutation_importance
        from sklearn.metrics import mean_squared_error, make_scorer
        
        modelo_sk = KerasRegressor( model = self.crear_modelo(), epochs = 100, batch_size = 32, verbose = 0)
        
        modelo_sk.fit(self.__x_train, self.__y_train)
        
        scoring = make_scorer(mean_squared_error, greater_is_better=False)
        
        perm_result = permutation_importance(
            modelo_sk, self.__x_test, self.__y_test,
            scoring = scoring, n_repeats = 10, random_state = 42)
        
        sorted_idx = perm_result.importances_mean.argsort()
        
        plt.barh(
            range(self.__x_test.shape[1]),
            perm_result.importances_mean[sorted_idx],
            xerr = perm_result.importances_std[sorted_idx]
            )
        
        plt.yticks(ticks = range(self.__x_test.shape[1]),labels = self.__variables_x[sorted_idx])
        plt.title("Importancia de las variables bajo permutación")
        plt.show()
        
        
    def gradientes(self):
        
        input_data = tf.convert_to_tensor(self.__x_test[:1].astype("float32"))
        with tf.GradientTape() as tape:
            tape.watch(input_data)
            predicciones = self.__modelo(input_data)
            
        grads = tape.gradient(predicciones, input_data)
        
        
        feature_names = self.__variables_x
        plt.barh(range(self.__x_test.shape[1]), np.abs(grads.numpy()).mean(axis=0))
        plt.yticks(range(self.__x_test.shape[1], feature_names))
        plt.title("Importancia de las variables bajo Gradientes")
        plt.show()
    
    #Método para graficar resultados
    def graficar_resultados(self, columna_fecha, nombre_real, nombre_prediccion=None):
        """
        Gráfica comparativa entre valores reales y predicciones
        
        Args:
            columna_fecha: Nombre de la columna con las fechas
            nombre_real: Nombre de la columna con los valores reales
            nombre_prediccion: Nombre para las predicciones (por defecto 'Predicción')
            
        Returns:
            La figura generada
        """
        if self.__predicciones is None:
            raise ValueError("Primero debes hacer predicciones con el método predecir()")
        
        if nombre_prediccion is None:
            nombre_prediccion = 'Predicción'
        
        # Preparar los datos
        indices_test = self.__x_test.index if hasattr(self.__x_test, 'index') else self.datos.index[-len(self.__predicciones):]

        df_resultado = pd.DataFrame({
            'Fecha': self.datos.loc[indices_test, columna_fecha].values,
            'Real': self.datos.loc[indices_test, nombre_real].values,
            nombre_prediccion: self.__predicciones
        })
                
        # Agrupar por fecha si hay duplicados
        df_resultado = df_resultado.groupby('Fecha').mean().reset_index()
        
        # Convertir a datetime si no lo es
        if not pd.api.types.is_datetime64_any_dtype(df_resultado['Fecha']):
            df_resultado['Fecha'] = pd.to_datetime(df_resultado['Fecha'])
        
        # Crear gráfico
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df_resultado['Fecha'], df_resultado['Real'], label='Datos reales', color='blue')
        ax.plot(df_resultado['Fecha'], df_resultado[nombre_prediccion], 
                label=nombre_prediccion, color='orange', linestyle='dashed')
        
        # Configurar eje X para fechas
        if len(df_resultado) > 60:
            # Si hay muchas fechas, mostrar cada 30 días aprox.
            interval = max(1, len(df_resultado) // 12)
            plt.xticks(df_resultado['Fecha'][::interval], 
                       [d.strftime('%b %Y') for d in df_resultado['Fecha'][::interval]])
        
        ax.set_title(f'Comparación entre valores reales y {nombre_prediccion.lower()}')
        ax.set_xlabel('Fecha')
        ax.set_ylabel(nombre_real)
        ax.legend()
        ax.grid(True)
        
        self._Grafico__grafico = fig  # Acceso al atributo privado de la clase Grafico
        return fig
    
    #Método para guardar el modelo
    def guardar_modelo(self, nombre):
        """
        Guarda el modelo en un archivo
        
        Args:
            nombre: Nombre del archivo (sin extensión)
        """
        if self.__modelo is None:
            raise ValueError("No hay modelo para guardar")
        
        self.__modelo.save(f"{nombre}.h5")
        print(f"Modelo guardado como {nombre}.h5")
    
    #Método para cargar un modelo guardado
    def cargar_modelo(self, ruta):
        """
        Carga un modelo guardado previamente
        
        Args:
            ruta: Ruta al archivo del modelo
        """
        self.__modelo = tf.keras.models.load_model(ruta)
        print(f"Modelo cargado desde {ruta}")
        return self.__modelo