Desafio Telecom X - Parte 2: Previsão de Churn
Este projeto faz parte do programa Alura Oracle ONE e tem como objetivo prever o churn (cancelamento) de clientes da empresa fictícia Telecom X.

Objetivo
Identificar clientes com maior risco de cancelar seus serviços, ajudando a empresa a criar estratégias de retenção e melhorar o atendimento.

Estrutura do Projeto
telecom_x_part2.ipynb: Notebook principal com análise, modelagem e gráficos.

dados_tratados.csv: Dados limpos e transformados da Parte 1.

TelecomX_Data.json: Dataset original.

feature_importance.png: Importância das variáveis no modelo.

churn_distribution.png: Distribuição da variável alvo (Churn).

churn_by_contract.png: Churn por tipo de contrato.

churn_by_monthly_charges.png: Churn por faturamento mensal.

metrics.txt: Métricas de avaliação dos modelos.

Preparação dos Dados
Conversão do JSON para formato tabular.

Limpeza de dados (remoção de valores ausentes e colunas irrelevantes).

Classificação de variáveis em categóricas e numéricas.

Codificação das variáveis (LabelEncoder e One-Hot Encoding).

Divisão em treino (80%) e teste (20%).

Padronização das variáveis numéricas.

Modelagem
Regressão Logística: Modelo simples e interpretável, usado como baseline.

Random Forest: Modelo robusto, com bom desempenho e menos propenso a overfitting.

Principais Insights
Distribuição de Churn: Base desbalanceada, com mais clientes que não cancelaram.

Tipo de Contrato: Contratos mensais têm maior taxa de churn.

Faturamento Mensal: Clientes com cobranças mais altas tendem a cancelar mais.

Variáveis Importantes: Contract, tenure e Charges.Total foram as mais relevantes.

Como Executar
Clone o repositório:

bash
git clone <URL_DO_SEU_REPOSITORIO>
cd <NOME_DO_SEU_REPOSITORIO>
Instale as dependências:

bash
pip install pandas numpy matplotlib seaborn scikit-learn
Baixe o dataset original:
TelecomX_Data.json  
Coloque-o na mesma pasta do notebook.

Execute o notebook em Jupyter Notebook ou Google Colab.
