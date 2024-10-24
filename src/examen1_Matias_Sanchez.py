import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Primeramente cargo el archivo CSV en un DataFrame
df = pd.read_csv( './data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

#Informacion general del Dataframe
print("\nInformación general del DataFrame:")
print(df.info())


# Inspección de los Datos
print("\nTipos de datos de cada columna:")
print(df.dtypes)
#Encontrar duplicados
cantidad_duplicados = df.duplicated().sum()
print(f"Cantidad de filas duplicadas: {cantidad_duplicados}")

# Diccionario de datos
diccionario_datos = {
    'customerID': 'object',
    'gender': 'object',
    'SeniorCitizen': 'int64',
    'Partner': 'object',
    'Dependents': 'object',
    'tenure': 'int64',
    'PhoneService': 'object',
    'MultipleLines': 'object',
    'InternetService': 'object',
    'OnlineSecurity': 'object',
    'OnlineBackup': 'object',
    'DeviceProtection': 'object',
    'TechSupport': 'object',
    'StreamingTV': 'object',
    'StreamingMovies': 'object',
    'Contract': 'object',
    'PaperlessBilling': 'object',
    'PaymentMethod': 'object',
    'MonthlyCharges': 'float64',
    'TotalCharges': 'object', 
    'Churn': 'object'
}
#Crear una copia ded df original
df_original = df.copy() 

# Asegurarse de que las columnas coincidan con los tipos de datos indicados en el diccionario
for columna, tipo_dato in diccionario_datos.items():
    try:
        if tipo_dato == 'datetime64[ns]':  # Si es una columna de fecha
            df[columna] = pd.to_datetime(df[columna], errors='coerce')
        elif tipo_dato == 'float64' and df[columna].dtype == 'object':
            df[columna] = pd.to_numeric(df[columna], errors='coerce')  # Convierte a numérico, NaN si falla
        else:
            df[columna] = df[columna].astype(tipo_dato)
    except ValueError as e:
        print(f"Error al convertir la columna '{columna}' al tipo {tipo_dato}: {e}")

# Verificar los tipos de datos después de la conversión
print("\nTipos de datos después de la conversión:")
print(df.dtypes)

# Identificar y solucionar inconsistencias en valores categóricos 
columnas_categoricas = df.select_dtypes(include='object').columns

# Convertir a minúsculas y eliminar espacios en blanco
for columna in columnas_categoricas:
    df[columna] = df[columna].str.lower().str.strip()
    
# Encontrar las inconsistencias: comparando el DataFrame original con el corregido
inconsistencias = (df_original[columnas_categoricas] != df[columnas_categoricas]).sum().sum()

print(f"\nCantidad total de valores inconsistentes encontrados: {inconsistencias}")

# Identificar valores faltantes antes de la interpolación
valores_faltantes_antes = df_original.isnull().sum()
print("\nCantidad de valores faltantes por columna antes de la interpolación:")
print(valores_faltantes_antes[valores_faltantes_antes > 0])

# Solucionar valores faltantes en columnas numéricas
columnas_numericas = df.select_dtypes(include='number').columns

for columna in columnas_numericas:
    if df[columna].isnull().sum() > 0:
        # Rellenar los valores faltantes con la mediana entre el valor anterior y el siguiente
        df[columna] = df[columna].interpolate(method='linear', limit_direction='both')

# Identificar valores faltantes después de la interpolación
valores_faltantes_despues = df.isnull().sum()

# Calcular cuántos valores se han rellenado
valores_rellenados = valores_faltantes_antes - valores_faltantes_despues
total_rellenados = valores_rellenados.sum()


# 1. Convertir negativos a positivos en las columnas numéricas FUENTE: https://joserzapata.github.io/courses/python-ciencia-datos/pandas/
df['tenure'] = df['tenure'].abs()  # Convertir a positivo
df['MonthlyCharges'] = df['MonthlyCharges'].abs()  # Convertir a positivo

# 2. Verificar si hay strings en columnas numéricas y convertirlos a la mediana
for columna in df.select_dtypes(include=['int64', 'float64']):
    # Encontrar índices donde hay valores que no pueden convertirse a numéricos
    indices_invalidos = df[~df[columna].apply(lambda x: isinstance(x, (int, float)))].index
    
    for indice in indices_invalidos:
        # Calcular la mediana del valor anterior y siguiente
        valor_anterior = df[columna].iloc[indice - 1] if indice > 0 else None
        valor_siguiente = df[columna].iloc[indice + 1] if indice < len(df) - 1 else None

        # Reemplazar por la mediana entre el anterior y el siguiente
        if valor_anterior is not None and valor_siguiente is not None:
            df.at[indice, columna] = (valor_anterior + valor_siguiente) / 2 
        
        elif valor_anterior is not None:  
            df.at[indice, columna] = valor_anterior
        
        elif valor_siguiente is not None:  
            df.at[indice, columna] = valor_siguiente
            
# Visualización 1: Histograma de 'MonthlyCharges' FUENTE: https://aprendeconalf.es/docencia/python/manual/matplotlib/
plt.figure(figsize=(10, 6))
sns.histplot(df['MonthlyCharges'], bins=30, kde=True)
plt.title('Distribución de Monthly Charges')
plt.xlabel('Monthly Charges')
plt.ylabel('Frecuencia')
plt.grid(axis='y', alpha=0.75)

# interpretación
plt.text(50, 120, "La mayoría de los clientes tienen cargos mensuales entre $20 y $80.", fontsize=12, color='blue')
plt.show()

# Visualización 2: Boxplot de 'MonthlyCharges' en función de 'Churn' FUENTE: https://www.datacamp.com/es/tutorial/python-boxplots
plt.figure(figsize=(10, 6))
sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
plt.title('Cargos Mensuales según Churn')
plt.xlabel('Churn')
plt.ylabel('Monthly Charges')
plt.grid(axis='y', alpha=0.75)

# interpretación
plt.text(0.1, 100, "Los clientes que se dan de baja tienden a tener\ncargos mensuales más altos en comparación\ncon los que no se dan de baja.", fontsize=12, color='blue')
plt.show()


# Visualización 1: Gráfico de dispersión de 'MonthlyCharges' vs. 'tenure' FUENTE: https://aprendeconalf.es/docencia/python/manual/matplotlib/
plt.figure(figsize=(10, 6))
sns.scatterplot(x='tenure', y='MonthlyCharges', hue='Churn', style='Churn', data=df, alpha=0.7)
plt.title('Relación entre Tenure y Cargos Mensuales')
plt.xlabel('Tenure (Meses)')
plt.ylabel('Cargos Mensuales')
plt.legend(title='Churn')
plt.grid()

# interpretación
plt.text(30, 100, "Los clientes con más tenure tienden a tener\ncargos mensuales más bajos, especialmente\nlos que no se dan de baja.", fontsize=12, color='blue')
plt.show()

# Visualización 2: Gráfico de violín de 'MonthlyCharges' por 'Churn' y 'InternetService' FUENTE: https://python-charts.com/es/distribucion/grafico-violin-matplotlib/
plt.figure(figsize=(12, 6))
sns.violinplot(x='Churn', y='MonthlyCharges', hue='InternetService', data=df, split=True)
plt.title('Distribución de Cargos Mensuales por Churn e Internet Service')
plt.xlabel('Churn')
plt.ylabel('Cargos Mensuales')
plt.grid()

# interpretación
plt.text(0.2, 100, "Los clientes que se dan de baja\ntienen una mayor dispersión en los cargos\nmensuales, especialmente en Internet Fiber.", fontsize=12, color='blue')
plt.show()




