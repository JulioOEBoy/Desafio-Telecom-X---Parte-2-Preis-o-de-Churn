import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar dados
df = pd.read_csv('dados_tratados.csv')

# 1. Distribuição de Churn
plt.figure(figsize=(6, 4))
sns.countplot(x='Churn', data=df, palette='viridis')
plt.title('Distribuição de Churn na Base de Dados')
plt.savefig('churn_distribution.png')

# 2. Churn por Tipo de Contrato
plt.figure(figsize=(10, 6))
sns.countplot(x='Contract', hue='Churn', data=df, palette='magma')
plt.title('Churn por Tipo de Contrato')
plt.savefig('churn_by_contract.png')

# 3. Churn por Faturamento Mensal (Charges.Monthly)
plt.figure(figsize=(10, 6))
sns.kdeplot(df[df['Churn'] == 'No']['Charges.Monthly'], label='Não Churn', fill=True)
sns.kdeplot(df[df['Churn'] == 'Yes']['Charges.Monthly'], label='Churn', fill=True)
plt.title('Distribuição de Faturamento Mensal por Churn')
plt.legend()
plt.savefig('churn_by_monthly_charges.png')

print("Visualizações geradas com sucesso.")
