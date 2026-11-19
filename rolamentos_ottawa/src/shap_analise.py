import argparse
import warnings

import numpy as np
import shap

from comum import RESULTADOS, SEED, carregar, fixar_sementes, separar_atributos
from experimentos import GRADE_XGB, Autoencoder, busca_hiperparametros, novo_tabpfn, novo_xgb

warnings.filterwarnings("ignore")

REP = 0
FUNDO_XGB = 100
FUNDO_TABPFN = 20


def explicar_xgb(modelo, X_tr, X_te, rng):
    fundo = X_tr[rng.choice(len(X_tr), min(FUNDO_XGB, len(X_tr)), replace=False)]
    explainer = shap.TreeExplainer(modelo, data=fundo, feature_perturbation="interventional",
                                   model_output="probability")
    return explainer.shap_values(X_te)


def explicar_fold(cond, df, espectros, tr, te, atributos, rng):
    y_tr = df.loc[tr, "y"].to_numpy()
    grupos = df.loc[tr, "rolamento"].to_numpy()
    seed = SEED + REP

    if cond in ("xgb_todos", "xgb_sem_temp", "xgb_todos_temp_primeiro"):
        nomes = atributos[cond[4:]]
        X_tr = df.loc[tr, nomes].to_numpy()
        X_te = df.loc[te, nomes].to_numpy()
        modelo, _ = busca_hiperparametros(novo_xgb(seed), GRADE_XGB, X_tr, y_tr, grupos)
        return explicar_xgb(modelo, X_tr, X_te, rng), X_te, nomes, modelo.predict_proba(X_te)[:, 1]

    if cond == "ae_xgb":
        ae = Autoencoder(seed=seed).fit(espectros[tr])
        E_tr = ae.transform(espectros[tr])
        E_te = ae.transform(espectros[te])
        modelo, _ = busca_hiperparametros(novo_xgb(seed), GRADE_XGB, E_tr, y_tr, grupos)
        nomes = [f"emb_{i:02d}" for i in range(E_tr.shape[1])]
        return explicar_xgb(modelo, E_tr, E_te, rng), E_te, nomes, modelo.predict_proba(E_te)[:, 1]

    if cond == "tabpfn_todos":
        nomes = atributos["todos"]
        X_tr = df.loc[tr, nomes].to_numpy()
        X_te = df.loc[te, nomes].to_numpy()
        modelo = novo_tabpfn(seed).fit(X_tr, y_tr)
        fundo = X_tr[rng.choice(len(X_tr), FUNDO_TABPFN, replace=False)]

        def prever(X):
            return modelo.predict_proba(X)[:, 1]

        explainer = shap.PermutationExplainer(
            prever, shap.maskers.Independent(fundo, max_samples=FUNDO_TABPFN), seed=seed)
        valores = explainer(X_te, max_evals=2 * len(nomes) + 1, batch_size=len(X_te))
        return valores.values, X_te, nomes, prever(X_te)

    raise ValueError(cond)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condicoes", nargs="+", default=["xgb_todos", "xgb_sem_temp", "ae_xgb"])
    args = parser.parse_args()

    df, espectros, folds = carregar()
    atributos = separar_atributos(df)
    col = folds.columns[REP]
    saida = RESULTADOS / "shap"
    saida.mkdir(parents=True, exist_ok=True)

    for cond in args.condicoes:
        fixar_sementes(SEED + REP)
        rng = np.random.default_rng(SEED + REP)
        idx, valores, X, proba = [], [], [], []

        for k in sorted(folds[col].unique()):
            te = np.where(folds[col] == k)[0]
            tr = np.where(folds[col] != k)[0]
            sv, X_te, nomes, p = explicar_fold(cond, df, espectros, tr, te, atributos, rng)
            idx.append(te)
            valores.append(sv)
            X.append(X_te)
            proba.append(p)
            print(f"  {cond} fold {k} ok")

        idx = np.concatenate(idx)
        np.savez(saida / f"{cond}.npz", idx=idx, shap=np.concatenate(valores), X=np.concatenate(X),
                 proba=np.concatenate(proba), y=df.loc[idx, "y"].to_numpy(), nomes=np.array(nomes))
        print(f"{cond} salvo")


if __name__ == "__main__":
    main()
