# Consumer Credit Risk

Projeto de Ciência de Dados para **avaliação de risco de crédito**, desenvolvido seguindo a metodologia **CRISP-DM**.

O objetivo do projeto é estimar a probabilidade de um cliente apresentar **inadimplência de 90 dias ou mais nos próximos 2 anos**, permitindo transformar a previsão de um modelo de Machine Learning em uma ferramenta de apoio à análise de crédito.

O projeto percorre todo o ciclo de desenvolvimento:

**Entendimento do negócio → Entendimento dos dados → Preparação dos dados → Modelagem → Avaliação → Deploy**

## Aplicação

O modelo final foi disponibilizado por meio de uma aplicação web desenvolvida com **Streamlit**.

A aplicação recebe os dados de um cliente, calcula sua probabilidade de inadimplência e aplica o limiar operacional definido no projeto.

**Acesse a aplicação:**

https://consumer-credit-risk-dropa.streamlit.app

## Modelo final

O modelo selecionado para o deploy foi o **XGBoost**.

A aplicação utiliza um limiar operacional de **10% de probabilidade de inadimplência** para transformar a probabilidade prevista pelo modelo em uma decisão operacional.

Além da previsão, a aplicação utiliza **SHAP** para fornecer uma explicação individual, mostrando quais características mais contribuíram para a previsão de cada cliente.

## Metodologia

O projeto foi desenvolvido seguindo as etapas da metodologia **CRISP-DM (Cross-Industry Standard Process for Data Mining)**.

### 1. Entendimento do negócio

O problema foi tratado como uma tarefa de classificação binária, na qual:

- `0` representa cliente sem inadimplência;
- `1` representa cliente com inadimplência de 90 dias ou mais no horizonte de 2 anos.

Além do desempenho estatístico, foi considerado o impacto dos erros de classificação no negócio, atribuindo maior importância aos falsos negativos.

### 2. Entendimento dos dados

Foram realizadas análises exploratórias para compreender:

- distribuição da variável alvo;
- desbalanceamento das classes;
- valores ausentes;
- valores extremos e códigos especiais;
- comportamento das variáveis de atraso;
- relação das principais características com a inadimplência.

### 3. Preparação dos dados

O processamento dos dados foi estruturado em um pipeline reproduzível, incluindo:

- tratamento de valores ausentes;
- criação de variáveis indicadoras para valores originalmente ausentes;
- tratamento dos valores especiais nas variáveis de atraso;
- controle de valores extremos;
- criação de variáveis derivadas;
- aplicação consistente das transformações entre treino, teste e novos clientes.

Os dados foram separados em **75% para treino e 25% para teste**, preservando a proporção da variável alvo por meio de `stratify`.

### 4. Modelagem

Foram avaliados diferentes algoritmos de classificação, incluindo:

- Dummy Classifier;
- Decision Tree;
- Random Forest;
- XGBoost;
- LightGBM.

A comparação dos modelos utilizou validação cruzada estratificada com **5 folds**, além da avaliação em uma base de teste mantida separada durante o desenvolvimento.

Entre as métricas analisadas estiveram:

- ROC-AUC;
- PR-AUC;
- Acurácia;
- Matriz de confusão.

### 5. Avaliação e decisão operacional

Além das métricas tradicionais de Machine Learning, o projeto considerou o custo associado aos diferentes tipos de erro.

O modelo final utiliza um **limiar operacional de 10%**, definido durante a etapa de avaliação, em vez de utilizar automaticamente o limiar padrão de 50%.

### 6. Explicabilidade

Foi utilizado **SHAP (SHapley Additive exPlanations)** para interpretar o comportamento do modelo.

A aplicação apresenta uma explicação individual da previsão, permitindo identificar quais características contribuíram para aumentar ou reduzir a saída do modelo para cada cliente.

### 7. Deploy

O modelo final foi integrado a uma aplicação desenvolvida com **Streamlit** e publicado no **Streamlit Community Cloud**.

O deploy utiliza o código versionado no GitHub e permite executar previsões sem necessidade de acessar diretamente notebooks ou código Python.

## Estrutura do projeto

```text
consumer_credit_risk/
├── data/
├── models/
│   └── modelo_credito_xgb.joblib
├── notebooks/
│   ├── 01_entendimento_dados.ipynb
│   ├── 02_preparacao_dados.ipynb
│   ├── 03_modelagem.ipynb
│   ├── 04_avaliacao_deploy.ipynb
│   └── 05_streamlit_deploy.ipynb
├── reports/
├── src/
│   └── consumer_credit_risk/
│       ├── __init__.py
│       ├── data.py
│       └── preprocessing.py
├── app.py
├── pyproject.toml
├── uv.lock
└── README.md
```

Os notebooks documentam as diferentes etapas do desenvolvimento, enquanto o código reutilizável de preparação dos dados foi organizado no pacote `consumer_credit_risk`.

O arquivo `app.py` contém a aplicação Streamlit e o modelo treinado utilizado em produção está armazenado em `models/modelo_credito_xgb.joblib`.

## Tecnologias utilizadas

- Python 3.12
- pandas
- NumPy
- scikit-learn
- XGBoost
- LightGBM
- SHAP
- Streamlit
- Jupyter
- uv
- Git
- GitHub

## Execução local

Com o projeto clonado e o ambiente configurado com `uv`, a aplicação pode ser executada com:

```bash
uv run streamlit run app.py
```

Após a inicialização, o Streamlit disponibiliza a aplicação localmente no navegador.

## Aplicação publicada

A versão em produção está disponível em:

https://consumer-credit-risk-dropa.streamlit.app

## Sobre o projeto

 Este projeto foi desenvolvido por Marcos Giovani Dropa de Bortoli como estudo aplicado do curso de Pos-Graduação em Data Analytics e Inteligência Artificial Aplicada a Negócios. Referes-se ao material apresentado no módulo de Data Analysis and Predictive Insights, que avalia a classificação e modelos árvore, ensemble e boosting. O projryo cobre o ciclo completo de construção de uma solução de Machine Learning: definição do problema de negócio, exploração e preparação dos dados, modelagem, avaliação, explicabilidade, definição de uma regra operacional e deploy.

O objetivo foi desenvolver não apenas um modelo preditivo, mas uma solução reproduzível e utilizável, conectando as etapas de análise de dados à disponibilização do modelo em uma aplicação web.

