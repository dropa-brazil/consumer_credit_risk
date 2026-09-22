import pandas as pd


COLUNAS_ATRASO = [
    "atrasos_30_59_dias",
    "atrasos_60_89_dias",
    "atrasos_90_mais_dias",
]

def preparar_dados_credito(X: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica os tratamentos definidos na etapa de preparação dos dados.

    Parameters
    ----------
    X : pd.DataFrame
        Base de dados a ser preparada.

    Returns
    -------
    pd.DataFrame
        Base de dados após a aplicação dos tratamentos.
    """

    X = X.copy()
    
    # 1. Tratamento da renda

    # Identificar os valores que originalmente estavam ausentes
    X["renda_estimada"] = (
        X["renda_mensal"].isna().astype(int)
    )

    # 2. Tratamento dos dependentes
    
    # Identificar os valores que originalmente estavam ausentes
    X["dependentes_estimados"] = (
        X["dependentes"].isna().astype(int)
    )
    
    # 3. Tratamento dos atrasos
    
    # Identificar clientes com os códigos especiais 96 ou 98
    X["atrasos_96_98"] = (
        X[COLUNAS_ATRASO]
        .isin([96, 98])
        .any(axis=1)
        .astype(int)
    )

    # Substituir os códigos especiais 96 e 98 por 20
    X[COLUNAS_ATRASO] = (
        X[COLUNAS_ATRASO].replace({96: 20, 98: 20})
    )
    
    # 4. Aplicar teto de R$ 50.000 à renda

    X["renda_mensal"] = (
       X["renda_mensal"].clip(upper=50000)
    )
    
    # 5. Feature engineering

    # 5.1 Renda por dependente
    X["renda_por_dependente"] = (
        X["renda_mensal"] / (X["dependentes"] + 1)
    )

    # 5.2 Sobra de caixa
    X["sobra_caixa"] = (
        X["renda_mensal"] * (1 - X["razao_divida"])
    )
    
    return X

 