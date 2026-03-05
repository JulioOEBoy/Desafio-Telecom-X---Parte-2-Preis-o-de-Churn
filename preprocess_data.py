import pandas as pd
import json

# Carregar o dataset JSON
with open('TelecomX_Data.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)

# Normalizar as colunas aninhadas
df_customer = pd.json_normalize(df['customer'])
df_phone = pd.json_normalize(df['phone'])
df_internet = pd.json_normalize(df['internet'])
df_account = pd.json_normalize(df['account'])

# Concatenar tudo
df_final = pd.concat([df[['customerID', 'Churn']], df_customer, df_phone, df_internet, df_account], axis=1)

# Limpeza básica (conforme Parte 1 do desafio)
# Remover IDs e colunas redundantes se houver
# Converter 'Total' para numérico (pode haver espaços vazios)
df_final['Charges.Total'] = pd.to_numeric(df_final['Charges.Total'], errors='coerce')
df_final = df_final.dropna(subset=['Charges.Total'])

# Remover linhas onde Churn está vazio (se houver)
df_final = df_final[df_final['Churn'] != '']

# Salvar dados tratados
df_final.to_csv('dados_tratados.csv', index=False)
print("Dados tratados salvos em dados_tratados.csv")
print(f"Formato do dataframe: {df_final.shape}")
print(df_final.head())
