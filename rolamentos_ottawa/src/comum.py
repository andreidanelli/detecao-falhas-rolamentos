import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

PASTA = Path(__file__).resolve().parents[1]
PROCESSADOS = PASTA / "data" / "processed"
RESULTADOS = PASTA / "results"
FIGURAS = RESULTADOS / "figures"

SEED = 42

COLUNAS_INFO = ["arquivo", "rolamento", "familia", "tipo_falha", "estado", "estado_nome", "janela",
                "rpm", "carga_N", "temp_invalida_frac", "y_iminente", "y_defeito"]


def fixar_sementes(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    import torch
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def carregar():
    df = pd.read_csv(PROCESSADOS / "features.csv")
    espectros = np.load(PROCESSADOS / "spectra.npy")
    folds = pd.read_csv(PROCESSADOS / "folds.csv")

    manter = (df["estado"] != 2).to_numpy()
    df = df[manter].reset_index(drop=True)
    df["y"] = df["y_iminente"].astype(int)
    return df, espectros[manter], folds[manter].reset_index(drop=True)


def separar_atributos(df):
    cols = [c for c in df.columns if c not in COLUNAS_INFO + ["y"]]
    temp = [c for c in cols if c.startswith("temp_")]
    acel = [c for c in cols if c.startswith("acel_")]
    acus = [c for c in cols if c.startswith("acus_")]
    return {
        "todos": acel + acus + temp,
        "todos_temp_primeiro": temp + acel + acus,
        "sem_temp": acel + acus,
        "acel": acel,
        "temp": temp,
    }


def grupo_do_atributo(nome):
    if nome.startswith("temp_"):
        return "temperatura"
    if nome.startswith("acel_"):
        return "vibração"
    if nome.startswith("acus_"):
        return "acústica"
    return "embedding"
