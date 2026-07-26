"""
import pandas as pd
import os

# 1. Leer el archivo directamente desde raw_data/
ruta_archivo = os.path.join(os.path.dirname(__file__), '..', 'raw_data', 'nuevos_clientes.csv')
df = pd.read_csv(ruta_archivo)

# 2. Modificar variables clave para generar drift deliberado
df['ingreso_neto'] = df['ingreso_neto'] * 3.5  # Incremento drástico en ingresos
df['monto'] = df['monto'] * 5.0              # Incremento drástico en montos

# 3. Guardar el archivo con drift en la misma carpeta raw_data con el nombre que espera el monitor
ruta_salida = os.path.join(os.path.dirname(__file__), '..', 'raw_data', 'p15_extrac.csv')
df.to_csv(ruta_salida, index=False)

print("¡Archivo con drift generado exitosamente en raw_data/p14_extrac.csv!")

"""


import pandas as pd
import os
   
# 1. Leer el archivo original desde raw_data/
ruta_archivo = os.path.join(os.path.dirname(__file__), '..', 'raw_data', 'nuevos_clientes.csv')
df = pd.read_csv(ruta_archivo)
   
# 2. Modificar variables clave para generar drift deliberado
df['ingreso_neto'] = df['ingreso_neto'] * 3.5  
df['monto'] = df['monto'] * 5.0              

# 3. Guardarlo con el nombre exacto que espera el monitor (ej. p12_extrac.csv)
ruta_salida = os.path.join(os.path.dirname(__file__), '..', 'raw_data', 'p16_extrac.csv')
df.to_csv(ruta_salida, index=False)

print("¡Archivo con drift guardado como p16_extrac.csv en raw_data/!")  