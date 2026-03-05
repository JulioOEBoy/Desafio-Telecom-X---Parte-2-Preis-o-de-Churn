Desafio Telecom X - Parte 2: Previsão de Churn

Este projeto contém a solução para a Parte 2 do desafio Telecom X do programa Alura Oracle ONE, focado na construção de modelos preditivos para identificar clientes propensos a churn (evasão).

Propósito da Análise

O objetivo principal deste projeto é prever o churn de clientes da empresa fictícia Telecom X, utilizando variáveis relevantes. A capacidade de prever quais clientes estão em risco de cancelar seus serviços permite à empresa tomar ações proativas para retê-los, otimizando estratégias de marketing e atendimento ao cliente.

Estrutura do Projeto




•
telecom_x_part2.ipynb: O notebook principal com toda a análise, modelagem e visualizações.

•
dados_tratados.csv: O conjunto de dados limpo e transformado da Parte 1 do desafio.

•
TelecomX_Data.json: O dataset original em formato JSON.

•
feature_importance.png: Gráfico mostrando a importância das variáveis para o modelo de Random Forest.

•
churn_distribution.png: Gráfico da distribuição da variável alvo (Churn).

•
churn_by_contract.png: Gráfico de churn por tipo de contrato.

•
churn_by_monthly_charges.png: Gráfico de churn por faturamento mensal.

•
metrics.txt: Arquivo de texto com as métricas de avaliação dos modelos.

Descrição do Processo de Preparação dos Dados

1.
Carregamento e Normalização de JSON: Os dados brutos, fornecidos em formato JSON aninhado, foram carregados e normalizados para um formato tabular utilizando pd.json_normalize().

2.
Limpeza de Dados: Valores ausentes na coluna Charges.Total foram removidos, e a coluna customerID foi descartada por não ser preditiva.

3.
Classificação de Variáveis: As variáveis foram classificadas em categóricas e numéricas.

4.
Codificação (Encoding):

•
A variável alvo Churn foi codificada usando LabelEncoder (No para 0 e Yes para 1).

•
As demais variáveis categóricas foram transformadas usando One-Hot Encoding com pd.get_dummies() para evitar a criação de uma ordem artificial.



5.
Separação em Treino e Teste: Os dados foram divididos em conjuntos de treino (80%) e teste (20%) usando train_test_split com stratify=y para manter a proporção da classe Churn em ambos os conjuntos.

6.
Normalização: As variáveis numéricas do conjunto de treino e teste foram padronizadas usando StandardScaler para garantir que todas as features contribuam igualmente para o modelo.

Justificativas para as Escolhas Feitas Durante a Modelagem

•
Regressão Logística: Escolhida por sua simplicidade, interpretabilidade e bom desempenho como baseline para problemas de classificação binária. É eficiente e fornece probabilidades de classe.

•
Random Forest Classifier: Selecionado por sua robustez, capacidade de lidar com não-linearidades e alta performance. É um modelo de ensemble que geralmente apresenta boa acurácia e é menos propenso a overfitting. A inclusão de class_weight="balanced" foi para mitigar o desbalanceamento da classe Churn.

Exemplos de Gráficos e Insights Obtidos (Análise Exploratória de Dados - EDA)

Durante a EDA, foram gerados gráficos para entender a distribuição do churn e a relação com outras variáveis. Os principais insights incluem:

•
Distribuição de Churn: A base de dados apresenta um desbalanceamento, com mais clientes que não cancelaram do que clientes que cancelaram.

•
Churn por Tipo de Contrato: Clientes com contratos mensais (Month-to-month) têm uma taxa de churn significativamente maior em comparação com contratos de longo prazo (One year ou Two year).

•
Churn por Faturamento Mensal: Clientes com faturamento mensal mais alto tendem a ter uma maior probabilidade de churn.

•
Importância das Variáveis: O modelo de Random Forest destacou Contract, tenure (tempo de permanência) e Charges.Total como as variáveis mais importantes para prever o churn.

Instruções para Executar o Notebook

Para executar este notebook, siga os passos abaixo:

1.
Clone o repositório:

Bash


git clone <URL_DO_SEU_REPOSITORIO>
cd <NOME_DO_SEU_REPOSITORIO>





2.
Instale as bibliotecas necessárias:

Bash


pip install pandas numpy matplotlib seaborn scikit-learn





3.
Baixe o dataset original:
Certifique-se de ter o arquivo TelecomX_Data.json na mesma pasta do notebook. Você pode baixá-lo de https://raw.githubusercontent.com/ingridcristh/challenge2-data-science/main/TelecomX_Data.json.

4.
Execute o notebook:
Abra o arquivo telecom_x_part2.ipynb em um ambiente como Jupyter Notebook ou Google Colab e execute todas as células.

Ao final da execução, os modelos serão treinados, avaliados e os gráficos de importância de variáveis e distribuição de churn serão gerados e salvos na mesma pasta.




