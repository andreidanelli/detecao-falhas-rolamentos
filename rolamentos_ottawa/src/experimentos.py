import argparse
import time
import warnings

import numpy as np
import pandas as pd
import torch  # importar antes do xgboost, senão dá segfault no mac
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef, recall_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from comum import RESULTADOS, SEED, carregar, fixar_sementes, separar_atributos

warnings.filterwarnings("ignore")

GRADE_XGB = {
    "max_depth": [2, 4],
    "n_estimators": [100, 300],
    "learning_rate": [0.05, 0.2],
}
GRADE_LR = {"clf__C": list(np.logspace(-3, 2, 8))}

CONDICOES = ["lr_todos", "lr_sem_temp", "xgb_todos", "xgb_sem_temp", "xgb_acel", "ae_xgb",
             "xgb_todos_temp_primeiro", "tabpfn_todos", "tabpfn_sem_temp"]


def novo_xgb(seed):
    return XGBClassifier(objective="binary:logistic", eval_metric="logloss", subsample=0.8,
                         colsample_bytree=0.8, random_state=seed, n_jobs=1, verbosity=0)


def busca_hiperparametros(modelo, grade, X, y, grupos):
    busca = GridSearchCV(modelo, grade, scoring="f1_macro", cv=GroupKFold(n_splits=3), n_jobs=-1)
    busca.fit(X, y, groups=grupos)
    return busca.best_estimator_, busca.best_params_


class Autoencoder:
    def __init__(self, latente=16, epocas=300, seed=SEED):
        self.latente = latente
        self.epocas = epocas
        self.seed = seed

    def fit(self, S):
        from torch import nn
        torch.manual_seed(self.seed)
        self.media = S.mean(0)
        self.desvio = S.std(0) + 1e-6
        Z = torch.tensor((S - self.media) / self.desvio, dtype=torch.float32)

        d = S.shape[1]
        self.enc = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, self.latente))
        self.dec = nn.Sequential(nn.Linear(self.latente, 64), nn.ReLU(), nn.Linear(64, d))
        otimizador = torch.optim.Adam(list(self.enc.parameters()) + list(self.dec.parameters()),
                                      lr=1e-3, weight_decay=1e-5)
        gerador = torch.Generator().manual_seed(self.seed)

        for _ in range(self.epocas):
            for lote in torch.randperm(len(Z), generator=gerador).split(32):
                otimizador.zero_grad()
                erro = ((self.dec(self.enc(Z[lote])) - Z[lote]) ** 2).mean()
                erro.backward()
                otimizador.step()
        self.erro_final = float(((self.dec(self.enc(Z)) - Z) ** 2).mean())
        return self

    def transform(self, S):
        with torch.no_grad():
            Z = torch.tensor((S - self.media) / self.desvio, dtype=torch.float32)
            return self.enc(Z).numpy()


def novo_tabpfn(seed):
    from tabpfn import TabPFNClassifier
    return TabPFNClassifier(device="cpu", random_state=seed)


def calcular_metricas(y, proba):
    pred = (proba >= 0.5).astype(int)
    return {
        "f1_macro": f1_score(y, pred, average="macro"),
        "auc_pr": average_precision_score(y, proba),
        "mcc": matthews_corrcoef(y, pred),
        "recall_falha": recall_score(y, pred, pos_label=1),
        "recall_saudavel": recall_score(y, pred, pos_label=0),
    }


def rodar_fold(cond, df, espectros, tr, te, seed, atributos):
    y_tr = df.loc[tr, "y"].to_numpy()
    y_te = df.loc[te, "y"].to_numpy()
    grupos = df.loc[tr, "rolamento"].to_numpy()
    hp = {}

    if cond.startswith("lr_"):
        X = df[atributos[cond[3:]]]
        pipe = Pipeline([("sc", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=5000, random_state=seed))])
        modelo, hp = busca_hiperparametros(pipe, GRADE_LR, X.loc[tr], y_tr, grupos)
        proba = modelo.predict_proba(X.loc[te])[:, 1]

    elif cond.startswith("xgb_"):
        X = df[atributos[cond[4:]]]
        modelo, hp = busca_hiperparametros(novo_xgb(seed), GRADE_XGB, X.loc[tr], y_tr, grupos)
        proba = modelo.predict_proba(X.loc[te])[:, 1]

    elif cond == "ae_xgb":
        ae = Autoencoder(seed=seed).fit(espectros[tr])
        emb_tr = ae.transform(espectros[tr])
        emb_te = ae.transform(espectros[te])
        modelo, hp = busca_hiperparametros(novo_xgb(seed), GRADE_XGB, emb_tr, y_tr, grupos)
        hp["ae_mse_treino"] = ae.erro_final
        proba = modelo.predict_proba(emb_te)[:, 1]

    elif cond.startswith("tabpfn_"):
        X = df[atributos[cond[7:]]]
        modelo = novo_tabpfn(seed).fit(X.loc[tr].to_numpy(), y_tr)
        proba = modelo.predict_proba(X.loc[te].to_numpy())[:, 1]

    else:
        raise ValueError(cond)
    return proba, y_te, hp


def rodar_condicao(cond, df, espectros, folds, atributos):
    linhas, previsoes = [], []
    for rep, col in enumerate(folds.columns):
        for k in sorted(folds[col].unique()):
            te = np.where(folds[col] == k)[0]
            tr = np.where(folds[col] != k)[0]
            seed = SEED + rep
            fixar_sementes(seed)

            inicio = time.time()
            proba, y_te, hp = rodar_fold(cond, df, espectros, tr, te, seed, atributos)
            rolamentos_teste = " ".join(str(r) for r in sorted(df.loc[te, "rolamento"].unique()))

            linha = {"condicao": cond, "rep": rep, "fold": k, "tempo_s": time.time() - inicio,
                     "rolamentos_teste": rolamentos_teste}
            linha.update(calcular_metricas(y_te, proba))
            linha.update({f"hp_{nome}": valor for nome, valor in hp.items()})
            linhas.append(linha)
            previsoes.append(pd.DataFrame({"condicao": cond, "rep": rep, "fold": k, "idx": te,
                                           "y": y_te, "proba": proba}))

        f1 = pd.DataFrame(linhas)["f1_macro"].mean()
        print(f"  {cond} rep {rep}: f1 medio = {f1:.3f}")
    return pd.DataFrame(linhas), pd.concat(previsoes, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    sem_tabpfn = [c for c in CONDICOES if not c.startswith("tabpfn")]
    parser.add_argument("--condicoes", nargs="+", default=sem_tabpfn, choices=CONDICOES)
    args = parser.parse_args()

    df, espectros, folds = carregar()
    atributos = separar_atributos(df)
    saida = RESULTADOS / "cv"
    saida.mkdir(parents=True, exist_ok=True)
    print(f"{len(df)} janelas, {df.rolamento.nunique()} rolamentos")

    for cond in args.condicoes:
        print(f"\n== {cond}")
        metricas, previsoes = rodar_condicao(cond, df, espectros, folds, atributos)
        metricas.to_csv(saida / f"{cond}_metricas.csv", index=False)
        previsoes.to_csv(saida / f"{cond}_preds.csv", index=False)
        print(metricas[["f1_macro", "auc_pr", "mcc"]].agg(["mean", "std"]).round(3))


if __name__ == "__main__":
    main()
