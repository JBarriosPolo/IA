import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importar datos 
archivo = 'interfuerza.xlsx'

#leer datos
archivo_read = pd.read_excel(archivo, sheet_name='Listado_de_Productos_Completo (')
df = pd.DataFrame(archivo_read)

