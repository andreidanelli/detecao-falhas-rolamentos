import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PASTA = Path(__file__).resolve().parents[1]
PASTA_RAW = PASTA / "data" / "raw"
SAIDA = PASTA / "data" / "processed"

FS = 42000
JANELA_S = 0.5
N_BINS = 256
SEED = 42
N_FOLDS = 4
N_REPS = 5
TEMP_MAX = 80.0

TIPOS = {"H": "saudavel", "I": "pista_interna", "O": "pista_externa", "B": "esfera", "C": "gaiola"}
ESTADOS = {0: "saudavel", 1: "falha_em_desenvolvimento", 2: "com_falha"}
FAMILIAS = ["pista_interna", "pista_externa", "esfera", "gaiola"]
FAIXAS = [(0, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 6000),
          (6000, 9000), (9000, 13000), (13000, 17000), (17000, 21000)]


def atributos_tempo(x, nome):
    x = x - x.mean()
    rms = np.sqrt(np.mean(x ** 2))
    pico = np.max(np.abs(x))
    return {
        f"{nome}_rms": rms,
        f"{nome}_pico": pico,
        f"{nome}_fator_crista": pico / rms if rms > 0 else 0.0,
        f"{nome}_curtose": stats.kurtosis(x),
        f"{nome}_assimetria": stats.skew(x),
        f"{nome}_pico_a_pico": np.ptp(x),
    }


def energia_por_faixa(x, nome):
    energia = np.abs(np.fft.rfft(x - x.mean())) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / FS)
    total = energia.sum()
    res = {}
    for ini, fim in FAIXAS:
        e = energia[(freqs >= ini) & (freqs < fim)].sum()
        res[f"{nome}_energia_{ini}_{fim}Hz"] = e / total if total > 0 else 0.0
    return res


def espectro(x):
    mag = np.abs(np.fft.rfft((x - x.mean()) * np.hanning(len(x))))
    partes = np.array_split(mag[1:], N_BINS)
    return np.log1p(np.array([p.mean() for p in partes], dtype=np.float32))


def processar_arquivo(caminho):
    m = re.match(r"([HIOBC])_(\d+)_(\d)\.csv$", caminho.name)
    tipo, rolamento, estado = m.group(1), int(m.group(2)), int(m.group(3))

    dados = pd.read_csv(caminho)
    rpm = float(dados["Speed"].iloc[0])
    carga = float(dados["Load"].iloc[0])
    acel = dados["Accelerometer"].to_numpy()
    acus = dados["Acoustic"].to_numpy()
    temp = dados["Temperature Difference"].to_numpy(dtype=float)

    invalida = np.abs(temp) > TEMP_MAX
    if invalida.any():
        temp = pd.Series(np.where(invalida, np.nan, temp)).interpolate(limit_direction="both").to_numpy()
        print(f"   {caminho.name}: {invalida.sum()} leituras de temperatura interpoladas")

    n = int(JANELA_S * FS)
    linhas, espectros = [], []
    for j in range(len(acel) // n):
        a = acel[j * n:(j + 1) * n]
        s = acus[j * n:(j + 1) * n]
        t = temp[j * n:(j + 1) * n]

        linha = {
            "arquivo": caminho.name,
            "rolamento": rolamento,
            "familia": FAMILIAS[(rolamento - 1) // 5],
            "tipo_falha": TIPOS[tipo],
            "estado": estado,
            "estado_nome": ESTADOS[estado],
            "janela": j,
            "rpm": rpm,
            "carga_N": carga,
        }
        linha.update(atributos_tempo(a, "acel"))
        linha.update(energia_por_faixa(a, "acel"))
        linha.update(atributos_tempo(s, "acus"))
        linha["temp_invalida_frac"] = invalida[j * n:(j + 1) * n].mean()
        linha["temp_media"] = t.mean()
        linha["temp_std"] = t.std()
        linha["temp_inclinacao"] = np.polyfit(np.arange(n) / FS, t, 1)[0]

        linhas.append(linha)
        espectros.append(espectro(a))
    return linhas, espectros


def gerar_folds(df):
    familia = df.groupby("rolamento")["familia"].first()
    ids = familia.index.to_numpy()
    folds = pd.DataFrame(index=df.index)
    usadas = set()

    for r in range(N_REPS):
        rng = np.random.default_rng(SEED + r)
        while True:
            partes = np.array_split(rng.permutation(ids), N_FOLDS)
            divisao = frozenset(frozenset(p) for p in partes)
            familias_diferentes = all(familia[p].nunique() == len(p) for p in partes)
            if familias_diferentes and divisao not in usadas:
                break
        usadas.add(divisao)
        fold_do_rolamento = {rol: k for k, p in enumerate(partes) for rol in p}
        folds[f"fold_rep{r}"] = df["rolamento"].map(fold_do_rolamento).to_numpy()
    return folds


def main():
    arquivos = sorted(PASTA_RAW.glob("*.csv"))
    if not arquivos:
        raise SystemExit("nenhum CSV em data/raw, rode antes o download_data.py")

    SAIDA.mkdir(parents=True, exist_ok=True)
    todas, espectros = [], []
    for i, arq in enumerate(arquivos, 1):
        linhas, esp = processar_arquivo(arq)
        todas += linhas
        espectros += esp
        print(f"[{i}/{len(arquivos)}] {arq.name}: {len(linhas)} janelas")

    df = pd.DataFrame(todas)
    df["y_iminente"] = df["estado"].map({0: 0, 1: 1, 2: np.nan})
    df["y_defeito"] = (df["estado"] > 0).astype(int)

    folds = gerar_folds(df)
    df.to_csv(SAIDA / "features.csv", index=False)
    np.save(SAIDA / "spectra.npy", np.stack(espectros))
    folds.to_csv(SAIDA / "folds.csv", index=False)

    print("tabela:", df.shape)
    print(df.groupby(["estado_nome", "tipo_falha"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
