from pathlib import Path

import pandas as pd


# Raiz do projeto
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Caminhos dos dados
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Arquivo de entrada
CREDIT_DATA_FILE = RAW_DATA_DIR / "credito_tratado.csv"


def carregar_dados_credito() -> pd.DataFrame:
    """
    Carrega a base de dados de crédito utilizada no projeto.

    Returns
    -------
    pd.DataFrame
        DataFrame contendo os dados de crédito.
    """
    df = pd.read_csv(CREDIT_DATA_FILE)

    return df