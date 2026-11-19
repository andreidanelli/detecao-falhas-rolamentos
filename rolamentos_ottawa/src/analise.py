import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr, wilcoxon

from comum import FIGURAS, RESULTADOS, carregar, grupo_do_atributo, separar_atributos

warnings.filterwarnings("ignore")

TABELAS = RESULTADOS / "tabelas"
METRICAS = ["f1_macro", "auc_pr", "mcc"]
NOME_METRICA = {"f1_macro": "F1-macro", "auc_pr": "AUC-PR", "mcc": "MCC"}
NOMES = {
    "lr_todos": "Reg. Logística (com temp.)",
    "lr_sem_temp": "Reg. Logística (sem temp.)",
    "xgb_todos": "XGBoost (com temp.)",
    "xgb_sem_temp": "XGBoost (sem temp.)",
    "xgb_acel": "XGBoost (só vibração)",
    "ae_xgb": "Autoencoder + XGBoost",
    "xgb_todos_temp_primeiro": "XGBoost (com temp., temp. 1ª coluna)",
    "tabpfn_todos": "TabPFN (com temp.)",
    "tabpfn_sem_temp": "TabPFN (sem temp.)",
}
CORES = {"temperatura": "#d1495b", "vibração": "#00798c", "acústica": "#edae49", "embedding": "#66a182"}
VERMELHO, AZUL = "#d1495b", "#00798c"

plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 9})


def salvar(fig, nome):
    fig.savefig(FIGURAS / f"{nome}.pdf", bbox_inches="tight")
    plt.close(fig)


def ler_resultados_cv():
    metricas, previsoes = {}, {}
    for cond in NOMES:
        arq = RESULTADOS / "cv" / f"{cond}_metricas.csv"
        if arq.exists():
            metricas[cond] = pd.read_csv(arq).sort_values(["rep", "fold"]).reset_index(drop=True)
            previsoes[cond] = pd.read_csv(RESULTADOS / "cv" / f"{cond}_preds.csv")
    return metricas, previsoes


def tabela_resultados(metricas):
    linhas = []
    for cond, m in metricas.items():
        linha = {"condicao": NOMES[cond]}
        for k in METRICAS + ["recall_falha", "recall_saudavel"]:
            linha[k] = f"{m[k].mean():.3f} ± {m[k].std():.3f}"
        linhas.append(linha)
    tab = pd.DataFrame(linhas)
    tab.to_csv(TABELAS / "resultados.csv", index=False)

    tex = tab[["condicao"] + METRICAS].copy()
    tex.columns = ["Condição", "F1-macro", "AUC-PR", "MCC"]
    tex = tex.apply(lambda c: c.str.replace("±", "$\\pm$", regex=False))
    (TABELAS / "resultados.tex").write_text(tex.to_latex(index=False, escape=False))
    return tab


def wilcoxon_pareado(a, b, metrica):
    x = a[metrica].to_numpy()
    y = b[metrica].to_numpy()
    dif = x - y
    n_dif = int(np.sum(np.abs(dif) > 1e-12))
    p = 1.0 if n_dif == 0 else float(wilcoxon(x, y, zero_method="zsplit").pvalue)
    return {"media_a": x.mean(), "media_b": y.mean(), "dif_media": dif.mean(), "n_pares": len(dif),
            "pares_diferentes": n_dif, "p": p}


def testes_estatisticos(metricas):
    comparacoes = [
        ("H1", "xgb_todos", "xgb_sem_temp"),
        ("H1", "tabpfn_todos", "tabpfn_sem_temp"),
        ("H1 (apoio)", "lr_todos", "lr_sem_temp"),
        ("H2", "xgb_acel", "ae_xgb"),
        ("sensibilidade", "xgb_todos_temp_primeiro", "xgb_todos"),
        ("contexto", "xgb_todos", "tabpfn_todos"),
        ("contexto", "xgb_todos", "lr_todos"),
    ]
    linhas = []
    for hip, a, b in comparacoes:
        if a not in metricas or b not in metricas:
            continue
        for k in METRICAS:
            linha = {"hipotese": hip, "a": NOMES[a], "b": NOMES[b], "metrica": k}
            linha.update(wilcoxon_pareado(metricas[a], metricas[b], k))
            linhas.append(linha)
    tab = pd.DataFrame(linhas)
    tab.to_csv(TABELAS / "wilcoxon.csv", index=False)

    tex = tab[["hipotese", "a", "b", "metrica", "dif_media", "pares_diferentes", "p"]].copy()
    tex["dif_media"] = tex["dif_media"].map(lambda v: f"{v:+.3f}")
    tex["p"] = tex["p"].map(lambda v: f"{v:.3f}" if v >= 0.001 else "<0,001")
    tex["metrica"] = tex["metrica"].map(NOME_METRICA)
    tex.columns = ["Hip.", "A", "B", "Métrica", "A$-$B", "Pares $\\neq$", "$p$"]
    (TABELAS / "wilcoxon.tex").write_text(tex.to_latex(index=False, escape=False))
    return tab


def grafico_boxplot(metricas):
    conds = list(metricas)
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.2))
    for ax, k in zip(axs, METRICAS):
        ax.boxplot([metricas[c][k] for c in conds], vert=False, widths=0.6)
        ax.set_yticks(range(1, len(conds) + 1))
        ax.set_yticklabels([NOMES[c] for c in conds] if ax is axs[0] else [])
        ax.set_title(NOME_METRICA[k])
    fig.suptitle("Distribuição nos 20 folds externos (4 folds × 5 repetições)")
    salvar(fig, "boxplot_metricas")


def acerto_por_rolamento(previsoes, df):
    tabelas = []
    for cond, p in previsoes.items():
        p = p.copy()
        p["rolamento"] = df.loc[p["idx"], "rolamento"].to_numpy()
        p["familia"] = df.loc[p["idx"], "familia"].to_numpy()
        p["acerto"] = ((p["proba"] >= 0.5).astype(int) == p["y"]).astype(float)
        t = p.groupby(["familia", "rolamento", "y"])["acerto"].mean().unstack("y")
        t.columns = ["acerto_saudavel", "acerto_falha"]
        t["condicao"] = NOMES[cond]
        tabelas.append(t.reset_index())
    tab = pd.concat(tabelas)
    tab.to_csv(TABELAS / "acerto_por_rolamento.csv", index=False)

    recall = tab.pivot_table(index="condicao", columns="rolamento", values="acerto_falha").round(2)
    recall.to_csv(TABELAS / "recall_falha_por_rolamento.csv")
    return recall


def grafico_temperatura(df):
    fig, ax = plt.subplots(figsize=(6, 3))
    for estado, cor, nome in [(0, AZUL, "saudável"), (1, VERMELHO, "falha em desenvolvimento")]:
        media = df[df.estado == estado].groupby("rolamento")["temp_media"].mean()
        ax.scatter(media.index.astype(str), media.values, color=cor, label=nome, s=40)
    ax.set_xlabel("rolamento")
    ax.set_ylabel("ΔT média (°C)")
    ax.legend(frameon=False)
    ax.set_title("Diferença de temperatura média por rolamento e estado")
    salvar(fig, "temperatura_por_estado")


def ler_shap(cond):
    arq = RESULTADOS / "shap" / f"{cond}.npz"
    if not arq.exists():
        return None
    z = np.load(arq, allow_pickle=False)
    return {k: z[k] for k in z.files}


def ranking(sv):
    return pd.Series(np.abs(sv["shap"]).mean(0), index=sv["nomes"]).sort_values(ascending=False)


def importancia_por_grupo(imp):
    soma = imp.groupby(imp.index.map(grupo_do_atributo)).sum()
    return (soma / soma.sum()).sort_values(ascending=False)


def graficos_shap(cond, sv, df):
    nomes = list(sv["nomes"])
    exp = shap.Explanation(values=sv["shap"], data=sv["X"], feature_names=nomes)
    titulo = NOMES[cond]

    plt.figure()
    shap.plots.bar(exp, max_display=12, show=False)
    plt.title(f"SHAP global — {titulo}")
    salvar(plt.gcf(), f"shap_bar_{cond}")

    plt.figure()
    shap.plots.beeswarm(exp, max_display=12, show=False)
    plt.title(f"SHAP beeswarm — {titulo}")
    salvar(plt.gcf(), f"shap_beeswarm_{cond}")

    fig, axs = plt.subplots(1, 2, figsize=(8, 3))
    cor = np.where(sv["y"] == 1, VERMELHO, AZUL)
    for ax, atributo in zip(axs, ranking(sv).index[:2]):
        j = nomes.index(atributo)
        ax.scatter(sv["X"][:, j], sv["shap"][:, j], c=cor, s=8, alpha=0.7)
        ax.set_xlabel(atributo)
        ax.set_ylabel("valor SHAP")
        if atributo.startswith(("acel_", "acus_")):
            ax.set_xscale("symlog")
    fig.suptitle(f"Dependência dos 2 principais atributos — {titulo} (vermelho = falha)")
    salvar(fig, f"shap_dependencia_{cond}")

    pred = (sv["proba"] >= 0.5).astype(int)
    y = sv["y"]
    proba = sv["proba"]
    rolamento = df.loc[sv["idx"], "rolamento"].to_numpy()

    casos = {}
    acertos_falha = np.where((pred == 1) & (y == 1))[0]
    acertos_saudavel = np.where((pred == 0) & (y == 0))[0]
    erros = np.where(pred != y)[0]
    falhas_rol12 = np.where((rolamento == 12) & (y == 1))[0]
    if len(acertos_falha):
        casos["acerto_falha"] = acertos_falha[np.argmax(proba[acertos_falha])]
    if len(acertos_saudavel):
        casos["acerto_saudavel"] = acertos_saudavel[np.argmin(proba[acertos_saudavel])]
    if len(erros):
        casos["erro"] = erros[np.argmax(np.abs(proba[erros] - y[erros]))]
    if len(falhas_rol12):
        casos["rolamento12_falha"] = falhas_rol12[np.argsort(proba[falhas_rol12])[len(falhas_rol12) // 2]]

    for caso, i in casos.items():
        base = float(proba[i] - sv["shap"][i].sum())
        e = shap.Explanation(values=sv["shap"][i], base_values=base, data=sv["X"][i], feature_names=nomes)
        plt.figure()
        shap.plots.waterfall(e, max_display=10, show=False)
        plt.title(f"{titulo} — {caso.replace('_', ' ')} (rolamento {rolamento[i]}, y={y[i]}, "
                  f"p={proba[i]:.2f})", fontsize=8)
        salvar(plt.gcf(), f"shap_local_{cond}_{caso}")
    return {caso: int(rolamento[i]) for caso, i in casos.items()}


def grafico_grupos(rankings):
    dados = pd.DataFrame({NOMES[c]: importancia_por_grupo(r) for c, r in rankings.items()}).fillna(0).T
    fig, ax = plt.subplots(figsize=(6, 0.5 + 0.45 * len(dados)))
    esquerda = np.zeros(len(dados))
    for g in CORES:
        if g in dados.columns:
            ax.barh(dados.index, dados[g], left=esquerda, color=CORES[g], label=g)
            esquerda += dados[g].to_numpy()
    ax.set_xlabel("fração do |SHAP| total")
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.25))
    ax.set_title("Quanto da explicação vem de cada grupo de atributos")
    salvar(fig, "shap_grupos")
    return dados


def embeddings_vs_atributos(sv, df):
    top = list(ranking(sv).index[:4])
    atributos = separar_atributos(df)
    cols = atributos["sem_temp"] + atributos["temp"]
    X = df.loc[sv["idx"], cols].reset_index(drop=True)
    nomes = list(sv["nomes"])

    corr = pd.DataFrame({d: [spearmanr(sv["X"][:, nomes.index(d)], X[c]).correlation for c in cols]
                         for d in top}, index=cols)
    corr.to_csv(TABELAS / "embedding_vs_atributos.csv")

    fig, ax = plt.subplots(figsize=(4.5, 7))
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols, fontsize=7)
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(top, rotation=45)
    fig.colorbar(im, ax=ax, shrink=0.6, label="Spearman")
    ax.set_title("Embeddings mais importantes vs. atributos originais", fontsize=8)
    salvar(fig, "embedding_vs_atributos")
    return {d: corr[d].abs().sort_values(ascending=False).head(3).round(2).to_dict() for d in top}


def avaliar_h1(rankings):
    h1 = {}
    for cond in ["xgb_todos", "xgb_todos_temp_primeiro", "tabpfn_todos"]:
        if cond not in rankings:
            continue
        r = rankings[cond]
        usados = r[r > 1e-6]
        posicoes_temp = [i + 1 for i, nome in enumerate(usados.index) if nome.startswith("temp_")]
        melhor = min(posicoes_temp) if posicoes_temp else None
        h1[cond] = {"top5": list(usados.index[:5]),
                    "atributos_usados": len(usados),
                    "melhor_posicao_temperatura": melhor,
                    "temp_no_top3": melhor is not None and melhor <= 3,
                    "fracao_shap_temperatura": float(importancia_por_grupo(r).get("temperatura", 0))}
    if "xgb_todos" in rankings and "tabpfn_todos" in rankings:
        xgb = rankings["xgb_todos"]
        tabpfn = rankings["tabpfn_todos"].reindex(xgb.index)
        h1["spearman_rankings_xgb_tabpfn"] = float(spearmanr(xgb, tabpfn).correlation)
    return h1


def main():
    TABELAS.mkdir(parents=True, exist_ok=True)
    FIGURAS.mkdir(parents=True, exist_ok=True)
    df, _, _ = carregar()
    metricas, previsoes = ler_resultados_cv()
    resumo = {}

    resultados = tabela_resultados(metricas)
    wil = testes_estatisticos(metricas)
    grafico_boxplot(metricas)
    grafico_temperatura(df)
    recall = acerto_por_rolamento(previsoes, df)

    rankings = {}
    for cond in ["xgb_todos", "xgb_sem_temp", "xgb_todos_temp_primeiro", "tabpfn_todos", "ae_xgb"]:
        sv = ler_shap(cond)
        if sv is None:
            continue
        rankings[cond] = ranking(sv)
        rankings[cond].to_csv(TABELAS / f"shap_ranking_{cond}.csv", header=["mean_abs_shap"])
        resumo[f"casos_locais_{cond}"] = graficos_shap(cond, sv, df)
        if cond == "ae_xgb":
            resumo["embeddings_interpretacao"] = embeddings_vs_atributos(sv, df)
    grupos = grafico_grupos({c: r for c, r in rankings.items() if c != "ae_xgb"})

    h1 = avaliar_h1(rankings)
    resumo["H1"] = h1
    resumo["wilcoxon"] = wil.round(4).to_dict(orient="records")
    (RESULTADOS / "resumo.json").write_text(json.dumps(resumo, indent=1, ensure_ascii=False, default=str))

    with open(RESULTADOS / "resumo.md", "w") as f:
        f.write("# Resultados\n\n" + resultados.to_markdown(index=False) + "\n\n")
        f.write("## Wilcoxon pareado\n\n" + wil.round(4).to_markdown(index=False) + "\n\n")
        f.write("## Recall da classe falha por rolamento\n\n" + recall.to_markdown() + "\n\n")
        f.write("## Fração do |SHAP| por grupo\n\n" + grupos.round(3).to_markdown() + "\n\n")
        f.write("## H1\n\n```\n" + json.dumps(h1, indent=1, ensure_ascii=False) + "\n```\n")
        if "embeddings_interpretacao" in resumo:
            f.write("\n## Embeddings x atributos (|Spearman| top 3)\n\n```\n"
                    + json.dumps(resumo["embeddings_interpretacao"], indent=1) + "\n```\n")
    print((RESULTADOS / "resumo.md").read_text())


if __name__ == "__main__":
    main()
