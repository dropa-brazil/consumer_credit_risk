import pandas as pd


COLUNAS_ATRASO = [
    "atrasos_30_59_dias",
    "atrasos_60_89_dias",
    "atrasos_90_mais_dias",
]
TETO_RENDA = 50_000

def preparar_dados_credito(
    X_treino: pd.DataFrame,
    X_teste: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aplica os tratamentos definidos na etapa de preparação dos dados.

    Os parâmetros necessários para os tratamentos são obtidos
    exclusivamente a partir da base de treino e depois aplicados
    às bases de treino e teste.
    """

    X_treino = X_treino.copy()
    X_teste = X_teste.copy()
    
    # 1. Tratamento da renda

    # Identificar os valores que originalmente estavam ausentes
    X_treino["renda_estimada"] = (
        X_treino["renda_mensal"].isna().astype(int)
    )

    X_teste["renda_estimada"] = (
        X_teste["renda_mensal"].isna().astype(int)
    )

    # Calcular a mediana exclusivamente na base de treino
    mediana_renda = X_treino["renda_mensal"].median()

    # Aplicar a mediana aprendida no treino às duas bases
    X_treino["renda_mensal"] = (
        X_treino["renda_mensal"].fillna(mediana_renda)
    )

    X_teste["renda_mensal"] = (
        X_teste["renda_mensal"].fillna(mediana_renda)
    )
    
    # 2. Tratamento dos dependentes

    # Identificar os valores que originalmente estavam ausentes
    X_treino["dependentes_estimados"] = (
        X_treino["dependentes"].isna().astype(int)
    )

    X_teste["dependentes_estimados"] = (
        X_teste["dependentes"].isna().astype(int)
    )

    # Calcular a mediana exclusivamente na base de treino
    mediana_dependentes = X_treino["dependentes"].median()

    # Aplicar a mediana aprendida no treino às duas bases
    X_treino["dependentes"] = (
        X_treino["dependentes"].fillna(mediana_dependentes)
    )

    X_teste["dependentes"] = (
        X_teste["dependentes"].fillna(mediana_dependentes)
    )
    
    # 3. Tratamento dos atrasos

    # Identificar clientes com os códigos especiais 96 ou 98
    X_treino["atrasos_96_98"] = (
        X_treino[COLUNAS_ATRASO]
        .isin([96, 98])
        .any(axis=1)
        .astype(int)
    )

    X_teste["atrasos_96_98"] = (
        X_teste[COLUNAS_ATRASO]
        .isin([96, 98])
        .any(axis=1)
        .astype(int)
    )

    # Substituir os códigos especiais 96 e 98 por 20
    X_treino[COLUNAS_ATRASO] = (
        X_treino[COLUNAS_ATRASO].replace({96: 20, 98: 20})
    )

    X_teste[COLUNAS_ATRASO] = (
        X_teste[COLUNAS_ATRASO].replace({96: 20, 98: 20})
    )
    
    # 4. Aplicar teto de R$ 50.000 à renda

    X_treino["renda_mensal"] = (
        X_treino["renda_mensal"].clip(upper=50000)
    )

    X_teste["renda_mensal"] = (
        X_teste["renda_mensal"].clip(upper=50000)
    )
    
    # 5. Feature engineering

    # 5.1 Renda por dependente
    X_treino["renda_por_dependente"] = (
        X_treino["renda_mensal"] / (X_treino["dependentes"] + 1)
    )

    X_teste["renda_por_dependente"] = (
        X_teste["renda_mensal"] / (X_teste["dependentes"] + 1)
    )

    # 5.2 Sobra de caixa
    X_treino["sobra_caixa"] = (
        X_treino["renda_mensal"] * (1 - X_treino["razao_divida"])
    )

    X_teste["sobra_caixa"] = (
        X_teste["renda_mensal"] * (1 - X_teste["razao_divida"])
    )
    return X_treino, X_teste

 