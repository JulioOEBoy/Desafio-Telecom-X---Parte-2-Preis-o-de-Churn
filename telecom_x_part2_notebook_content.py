
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score

# Configurações para visualização
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# --- 1. Carregamento e Pré-processamento dos Dados ---
print("\n--- 1. Carregamento e Pré-processamento dos Dados ---")

# Carregar o dataset JSON (se não existir, o script preprocess_data.py deve ser executado primeiro)
# Ou carregar diretamente o CSV tratado se já existir
try:
    df = pd.read_csv("dados_tratados.csv")
    print("Dados tratados carregados com sucesso.")
except FileNotFoundError:
    print("Arquivo 'dados_tratados.csv' não encontrado. Executando pré-processamento...")
    # Este bloco seria para re-executar o preprocess_data.py se necessário
    # Para este desafio, assumimos que dados_tratados.csv já foi gerado.
    # Se fosse um notebook real, o código de preprocess_data.py estaria aqui.
    import json
    with open("TelecomX_Data.json", "r") as f:
        data = json.load(f)
    df_raw = pd.DataFrame(data)

    df_customer = pd.json_normalize(df_raw["customer"])
    df_phone = pd.json_normalize(df_raw["phone"])
    df_internet = pd.json_normalize(df_raw["internet"])
    df_account = pd.json_normalize(df_raw["account"])

    df = pd.concat([df_raw[["customerID", "Churn"]], df_customer, df_phone, df_internet, df_account], axis=1)

    df["Charges.Total"] = pd.to_numeric(df["Charges.Total"], errors="coerce")
    df = df.dropna(subset=["Charges.Total"])
    df = df[df["Churn"] != ""]
    df.to_csv("dados_tratados.csv", index=False)
    print("Pré-processamento concluído e 'dados_tratados.csv' gerado.")
    df = pd.read_csv("dados_tratados.csv")

print(f"Formato inicial do dataframe: {df.shape}")
print("Primeiras 5 linhas do dataframe:")
print(df.head())

# Remover customerID
df = df.drop(columns=["customerID"])

# Classificação das variáveis
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
num_cols = df.select_dtypes(exclude=["object"]).columns.tolist()

print("\nVariáveis Categóricas:", cat_cols)
print("Variáveis Numéricas:", num_cols)

# Encoding da variável alvo 'Churn'
le = LabelEncoder()
df["Churn"] = le.fit_transform(df["Churn"]) # 'No' -> 0, 'Yes' -> 1

# One-Hot Encoding para outras variáveis categóricas
df_encoded = pd.get_dummies(df, columns=[c for c in cat_cols if c != "Churn"], drop_first=True)

print("\nFormato do dataframe após encoding:", df_encoded.shape)
print("Primeiras 5 linhas do dataframe após encoding:")
print(df_encoded.head())

# Separação dos dados em treino e teste
X = df_encoded.drop("Churn", axis=1)
y = df_encoded["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nShape de X_train: {X_train.shape}")
print(f"Shape de X_test: {X_test.shape}")
print(f"Proporção de Churn no treino: {y_train.value_counts(normalize=True)[1]:.2f}")
print(f"Proporção de Churn no teste: {y_test.value_counts(normalize=True)[1]:.2f}")

# Normalização das variáveis numéricas
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convertendo de volta para DataFrame para manter nomes das colunas
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

print("\nDados normalizados com sucesso.")

# --- 2. Modelagem Preditiva ---
print("\n--- 2. Modelagem Preditiva ---")

# Modelo 1: Regressão Logística
print("\nTreinando Regressão Logística...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)
y_proba_lr = lr_model.predict_proba(X_test_scaled)[:, 1]

print("\nResultados Regressão Logística:")
print(classification_report(y_test, y_pred_lr))
print(f"Acurácia: {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_proba_lr):.4f}")

# Modelo 2: Random Forest Classifier
print("\nTreinando Random Forest Classifier...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=\"balanced\") # class_weight para lidar com desbalanceamento
rf_model.fit(X_train, y_train) # Random Forest geralmente não precisa de dados escalados
y_pred_rf = rf_model.predict(X_test)
y_proba_rf = rf_model.predict_proba(X_test)[:, 1]

print("\nResultados Random Forest:")
print(classification_report(y_test, y_pred_rf))
print(f"Acurácia: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_proba_rf):.4f}")

# --- 3. Análise de Importância de Variáveis e Visualizações ---
print("\n--- 3. Análise de Importância de Variáveis e Visualizações ---")

# Importância das Variáveis (Random Forest)
importances = rf_model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\nTop 10 Variáveis mais Importantes (Random Forest):")
print(feature_importance_df.head(10))

# Gráfico de Importância das Variáveis
plt.figure(figsize=(12, 8))
sns.barplot(x="Importance", y="Feature", data=feature_importance_df.head(10), palette="viridis")
plt.title("Top 10 Variáveis mais Importantes para Previsão de Churn (Random Forest)")
plt.xlabel("Importância (Gini Impurity)")
plt.ylabel("Variável")
plt.tight_layout()
plt.savefig("feature_importance.png")
print("Gráfico de importância de variáveis salvo como 'feature_importance.png'")

# Visualização da Distribuição de Churn
plt.figure(figsize=(7, 5))
sns.countplot(x="Churn", data=df, palette="coolwarm")
plt.title("Distribuição da Variável Alvo (Churn)")
plt.xlabel("Churn (0 = Não, 1 = Sim)")
plt.ylabel("Contagem")
plt.savefig("churn_distribution.png")
print("Gráfico de distribuição de churn salvo como 'churn_distribution.png'")

# Visualização Churn por Tipo de Contrato
plt.figure(figsize=(10, 6))
sns.countplot(x="Contract", hue="Churn", data=df, palette="pastel")
plt.title("Churn por Tipo de Contrato")
plt.xlabel("Tipo de Contrato")
plt.ylabel("Contagem")
plt.legend(title="Churn", labels=["Não", "Sim"])
plt.savefig("churn_by_contract.png")
print("Gráfico de churn por contrato salvo como 'churn_by_contract.png'")

# Visualização Churn por Faturamento Mensal
plt.figure(figsize=(10, 6))
sns.kdeplot(df[df["Churn"] == 0]["Charges.Monthly"], label="Não Churn", fill=True, alpha=0.5)
sns.kdeplot(df[df["Churn"] == 1]["Charges.Monthly"], label="Churn", fill=True, alpha=0.5)
plt.title("Distribuição de Faturamento Mensal por Status de Churn")
plt.xlabel("Faturamento Mensal")
plt.ylabel("Densidade")
plt.legend()
plt.savefig("churn_by_monthly_charges.png")
print("Gráfico de churn por faturamento mensal salvo como 'churn_by_monthly_charges.png'")

print("\n--- Conclusão ---")
print("Os modelos de Regressão Logística e Random Forest foram treinados e avaliados. O Random Forest apresentou uma boa capacidade preditiva e forneceu insights sobre as variáveis mais importantes para a previsão de churn, como o tipo de contrato e o faturamento mensal. As visualizações confirmam a importância dessas variáveis na análise de evasão de clientes.")

