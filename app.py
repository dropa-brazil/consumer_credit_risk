import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st
from pathlib import Path


# ============================================================
# Configuração da página
# ============================================================

st.set_page_config(
    page_title="Análise de Risco de Crédito",
    page_icon="💳",
    layout="wide",
)


# ============================================================
# Caminhos do projeto
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "modelo_credito_xgb.joblib"
)


# ============================================================
# Carregamento do artefato
# ============================================================

artefato = joblib.load(MODEL_PATH)

pipeline = artefato["pipeline"]
limiar = artefato["limiar"]
nome_modelo = artefato["modelo"]


# ============================================================
# Interface
# ============================================================

st.title("💳 Análise de Risco de Crédito")

st.write(
    f"Sistema baseado no modelo **{nome_modelo}** "
    f"(Limiar operacional: **{limiar:.0%}**)"
)

st.success("Modelo carregado com sucesso.")


# ============================================================
# Formulário de entrada
# ============================================================

st.subheader("Preencha os dados do cliente:")

with st.form("formulario_credito"):

    col1, col2 = st.columns(2)

    with col1:

        idade = st.number_input(
            "Idade",
            min_value=18,
            max_value=120,
            value=35,
            step=1,
        )

        renda_mensal = st.number_input(
            "Renda Mensal (R$)",
            min_value=0.0,
            value=5000.0,
            step=100.0,
        )

        dependentes = st.number_input(
            "Nº de Dependentes",
            min_value=0,
            value=0,
            step=1,
        )

        uso_limite_rotativo = st.number_input(
            "Uso do Limite Rotativo",
            min_value=0.0,
            value=0.30,
            step=0.01,
            format="%.2f",
        )

        razao_divida = st.number_input(
            "Razão da Dívida",
            min_value=0.0,
            value=0.20,
            step=0.01,
            format="%.2f",
        )

    with col2:

        linhas_credito_abertas = st.number_input(
            "Linhas de Crédito Abertas",
            min_value=0,
            value=5,
            step=1,
        )

        financiamentos_imobiliarios = st.number_input(
            "Financiamentos Imobiliários",
            min_value=0,
            value=1,
            step=1,
        )

        atrasos_30_59_dias = st.number_input(
            "Qtd. Atrasos (30–59 dias)",
            min_value=0,
            value=0,
            step=1,
        )

        atrasos_60_89_dias = st.number_input(
            "Qtd. Atrasos (60–89 dias)",
            min_value=0,
            value=0,
            step=1,
        )

        atrasos_90_mais_dias = st.number_input(
            "Qtd. Atrasos (90+ dias)",
            min_value=0,
            value=0,
            step=1,
        )

    analisar = st.form_submit_button(
        "Realizar Análise de Crédito",
        use_container_width=True,
    )
    
# ============================================================
# Análise do cliente
# ============================================================

if analisar:

    cliente = pd.DataFrame(
        [{
            "idade": idade,
            "renda_mensal": renda_mensal,
            "dependentes": dependentes,
            "uso_limite_rotativo": uso_limite_rotativo,
            "razao_divida": razao_divida,
            "linhas_credito_abertas": linhas_credito_abertas,
            "financiamentos_imobiliarios": financiamentos_imobiliarios,
            "atrasos_30_59_dias": atrasos_30_59_dias,
            "atrasos_60_89_dias": atrasos_60_89_dias,
            "atrasos_90_mais_dias": atrasos_90_mais_dias,
        }]
    )

    st.divider()

    st.subheader("Dados enviados ao modelo")

    st.dataframe(
        cliente,
        use_container_width=True,
        hide_index=True,
    )
    
    # ========================================================
    # Previsão
    # ========================================================

    probabilidades = pipeline.predict_proba(cliente)

    probabilidade_inadimplencia = probabilidades[0, 1]

    classificacao = int(
        probabilidade_inadimplencia >= limiar
    )


    # ========================================================
    # Resultado
    # ========================================================

    st.divider()

    st.header("Resultado da Decisão")

    st.metric(
        "Probabilidade de Inadimplência",
        f"{probabilidade_inadimplencia:.2%}",
    )

    if classificacao == 0:

        st.success("✅ CRÉDITO APROVADO")

        st.info(
            "A probabilidade estimada de inadimplência "
            "ficou abaixo do limiar operacional."
        )

    else:

        st.error("❌ CRÉDITO RECUSADO")

        st.warning(
            "A probabilidade estimada de inadimplência "
            "atingiu ou ultrapassou o limiar operacional."
        )
        
    # ========================================================
    # SHAP individual
    # ========================================================

    st.divider()

    st.header("Fatores Determinantes deste Cliente (SHAP)")

    st.write(
        "Abaixo estão as 5 características que mais "
        "influenciaram a previsão específica deste cliente."
    )

    # Aplicar preparação e imputação sem executar o modelo
    cliente_preparado = pipeline[:-1].transform(cliente)

    # Recuperar somente o XGBoost treinado
    modelo_xgb = pipeline.named_steps["modelo"]

    # Criar explicador SHAP
    explainer = shap.TreeExplainer(modelo_xgb)

    # Calcular SHAP para este cliente
    explicacao = explainer(cliente_preparado)

    # Valores SHAP da única linha analisada
    valores_shap = explicacao.values[0]

    # Criar tabela com variável, valor do cliente e SHAP
    df_shap_cliente = pd.DataFrame({
        "Variável": cliente_preparado.columns,
        "Valor do Cliente": cliente_preparado.iloc[0].values,
        "SHAP": valores_shap,
    })

    # Magnitude absoluta para ordenar importância
    df_shap_cliente["Impacto Absoluto"] = (
        df_shap_cliente["SHAP"].abs()
    )

    # Selecionar os 5 fatores mais importantes
    top_5_shap = (
        df_shap_cliente
        .sort_values(
            "Impacto Absoluto",
            ascending=False
        )
        .head(5)
        .reset_index(drop=True)
    )

    st.dataframe(
        top_5_shap[
            ["Variável", "Valor do Cliente", "SHAP"]
        ],
        use_container_width=True,
        hide_index=True,
    )  
        # ========================================================
    # Gráfico SHAP Waterfall
    # ========================================================

    st.subheader("Como as características influenciaram a previsão")

    st.write(
        "O gráfico abaixo mostra como as características "
        "do cliente contribuíram para a saída do modelo."
    )

    # Criar a figura
    fig, ax = plt.subplots(figsize=(10, 6))

    # Gerar o waterfall para o cliente analisado
    shap.plots.waterfall(
        explicacao[0],
        max_display=6,
        show=False
    )

    # Exibir no Streamlit
    st.pyplot(plt.gcf(), clear_figure=True)

    # Fechar a figura para liberar memória
    plt.close()