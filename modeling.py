import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score

# 1. Carregar dados
df = pd.read_csv('dados_tratados.csv')

# 2. Preparação dos Dados
# Remover customerID pois não é preditivo
df = df.drop(columns=['customerID'])

# Identificar colunas categóricas e numéricas
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
num_cols = df.select_dtypes(exclude=['object']).columns.tolist()

# Encoding das variáveis categóricas
# Churn é o alvo
le = LabelEncoder()
df['Churn'] = le.fit_transform(df['Churn']) # No=0, Yes=1

# Para as outras categóricas, usar One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=[c for c in cat_cols if c != 'Churn'], drop_first=True)

# 3. Divisão Treino/Teste
X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Normalização
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Modelagem - Regressão Logística
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

# 6. Modelagem - Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train) # RF não precisa de scaling
y_pred_rf = rf.predict(X_test)

# 7. Resultados
print("--- Regressão Logística ---")
print(classification_report(y_test, y_pred_lr))
print(f"Acurácia: {accuracy_score(y_test, y_pred_lr):.4f}")

print("\n--- Random Forest ---")
print(classification_report(y_test, y_pred_rf))
print(f"Acurácia: {accuracy_score(y_test, y_pred_rf):.4f}")

# 8. Importância das Variáveis (Random Forest)
importances = rf.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10))
plt.title('Top 10 Variáveis mais Importantes (Random Forest)')
plt.tight_layout()
plt.savefig('feature_importance.png')

# Salvar métricas para o README
with open('metrics.txt', 'w') as f:
    f.write(f"LR Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}\n")
    f.write(f"RF Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}\n")
    f.write("\nFeature Importance:\n")
    f.write(feature_importance_df.head(10).to_string())

print("\nModelagem concluída. Gráfico salvo como feature_importance.png")
