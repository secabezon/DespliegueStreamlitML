import streamlit as st
from PIL import Image
import pandas as pd
import time

#import torch
import joblib
import numpy as np
import matplotlib.pyplot as plt

from feature_engine.imputation import CategoricalImputer, MeanMedianImputer, AddMissingIndicator
from feature_engine.encoding import OrdinalEncoder, RareLabelEncoder
from sklearn.preprocessing import MinMaxScaler, Binarizer
from sklearn.pipeline import Pipeline

import Libs.preprocessors as pp

# Importar las variables del archivo config.py
from configuraciones import config

#def prediccion_o_inferencia(modelo, datos_de_test):
def prediccion_o_inferencia(pipeline_de_test, datos_de_test):
    #Dropeamos
    datos_de_test['running']=datos_de_test['running'].apply(lambda x: float(x.replace('km','')) if x[-2:]=='km' else float(x.replace('miles',''))*1.609344)
    datos_de_test.drop(['Id','wheel'], axis=1, inplace=True)
    predicciones = pipeline_de_test.predict(datos_de_test)
    predicciones_sin_escalar = np.exp(predicciones)
    return predicciones, predicciones_sin_escalar, datos_de_test

# Designing the interface
st.title("Proyecto DataPath - Sebastian Cabezon")

image = Image.open('src/images/datapath.png')
st.image(image, use_column_width=True)

st.sidebar.write("Suba el archivo CSV Correspondiente para realizar la predicción")

#-------------------------------------------------------------------------------------------------
# Cargar el archivo CSV desde la barra lateral
uploaded_file = st.sidebar.file_uploader(" ", type=['csv'])

if uploaded_file is not None:
    # Leer el archivo CSV
    df_de_los_datos_subidos = pd.read_csv(uploaded_file)
    
    # Mostrar el contenido del archivo CSV
    st.write('Contenido del archivo CSV:')
    st.dataframe(df_de_los_datos_subidos)
#-------------------------------------------------------------------------------------------------
#Cargar el modelo
#modelo = torch.load('./modelos/melhor_modelo.pt')

#Cargar Pipeline
pipeline_de_produccion = joblib.load('src/pipeline.joblib')
 
if st.sidebar.button("Click aqui para enviar el CSV al Pipeline"):
    if uploaded_file is None:
        st.sidebar.write("No se cargó correctamente el archivo, subalo de nuevo")
    else:
        with st.spinner('Pipeline y Modelo procesando...'):

            #prediction = prediccion_o_inferencia(modelo, df_de_los_datos_subidos)
            prediction,prediction_sin_escalar, datos_procesados = prediccion_o_inferencia(pipeline_de_produccion, df_de_los_datos_subidos)
            time.sleep(2)
            st.success('Listo!')

            # Mostrar los resultados de la predicción
            st.write('Resultados de la predicción:')
            st.write(prediction) #Dataframe - 1 sola Columna
            st.write(prediction_sin_escalar) #Dataframe - 1 sola Columna

            # Graficar los precios de venta predichos
            fig, ax = plt.subplots()
            pd.Series(prediction).hist(bins=50, ax=ax)
            ax.set_title('Histograma de los precios de venta predichos')
            ax.set_xlabel('Precio')
            ax.set_ylabel('Frecuencia')

            # Mostrar la gráfica en Streamlit
            st.pyplot(fig)

            #Proceso para descargar todo el archivo con las predicciones
            #----------------------------------------------------------------------------------
            # Concatenar predicciones con el archivo original subido
            df_resultado = datos_procesados.copy()
            df_resultado['Predicción Escalada'] = prediction
            df_resultado['Predicción Sin Escalar'] = prediction_sin_escalar
            
            # Mostrar el DataFrame concatenado
            st.write('Datos originales con predicciones:')
            st.dataframe(df_resultado)

            # Crear archivo CSV para descargar
            csv = df_resultado.to_csv(index=False).encode('utf-8')

            # Botón para descargar el CSV
            st.download_button(
                label="Descargar archivo CSV con predicciones",
                data=csv,
                file_name='predicciones_casas.csv',
                mime='text/csv',
            )
            #----------------------------------------------------------------------------------

#streamlit run basics_streamlit.py