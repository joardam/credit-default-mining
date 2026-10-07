"""
Agrupamento com K-Means — revalidacao de k e ajuste final na base completa
Projeto: Mineracao de Dados Aplicada a Predicao de Inadimplencia

Roda K-Means (nao mais em subamostra) nos 3 conjuntos de features
escolhidos:
  1. reduzidas            -> LIMIT_BAL, AGE, AVG_UTIL_RATIO, AVG_PAY_RATIO,
                              N_MESES_ATRASO, SEM_FATURA_POSITIVA
  2. perfil_uso_credito    -> AVG_UTIL_RATIO, AVG_PAY_RATIO
  3. perfil_socioeconomico -> LIMIT_BAL, AGE

Para cada conjunto:
  - recalcula a curva de cotovelo (WCSS) e a curva de silhueta, k = 2..10,
    na base de treino inteira (nao mais na subamostra de 3.000 usada na
    exploracao anterior);
  - escolhe k pelo maior indice de silhueta;
  - ajusta o K-Means final com esse k;
  - salva os pontos com o rotulo de cluster, o perfil medio de cada
    cluster (nas proprias features padronizadas) e o cruzamento com a
    variavel-alvo (taxa de inadimplencia por cluster).

Uso:
    python src/kmeans_final.py
Saida (por conjunto de features):
    reports/figures/kmeans_<nome>_cotovelo_silhueta.png
    reports/figures/kmeans_<nome>_clusters.png
    reports/kmeans_<nome>_atribuicoes.csv
    reports/kmeans_<nome>_perfil_clusters.csv
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

TRAIN_PATH = os.path.join("data", "processed", "treino_processado.csv")
FIG_DIR = os.path.join("reports", "figures")
REPORT_DIR = "reports"
TARGET = "default payment next month"
RANDOM_STATE = 42
K_RANGE = range(2, 11)  # silhueta exige k >= 2

FEATURE_SETS = {
    "reduzidas": ["LIMIT_BAL", "AGE", "AVG_UTIL_RATIO", "AVG_PAY_RATIO",
                  "N_MESES_ATRASO", "SEM_FATURA_POSITIVA"],
    "perfil_uso_credito": ["AVG_UTIL_RATIO", "AVG_PAY_RATIO"],
    "perfil_socioeconomico": ["LIMIT_BAL", "AGE"],
}

CORES_CICLO = ["#2a78d6", "#eb6834", "#3fa66a", "#9463c9", "#c9a227",
               "#d6477a", "#4bbfb8", "#8a8d91"]
C_GRID, C_MUTED, C_TEXT = "#d8d7d2", "#52514e", "#0b0b0b"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
    "font.family": "DejaVu Sans", "axes.edgecolor": C_MUTED,
    "axes.labelcolor": C_TEXT, "text.color": C_TEXT, "xtick.color": C_MUTED,
    "ytick.color": C_MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def cotovelo_e_silhueta(X, nome):
    wcss, silhuetas = [], []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
        wcss.append(km.inertia_)
        silhuetas.append(silhouette_score(X, km.labels_))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))

    ax = axes[0]
    ax.grid(color=C_GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.plot(list(K_RANGE), wcss, marker="o", color="#2a78d6", linewidth=1.6, zorder=3)
    ax.set_xlabel("N\u00famero de clusters (k)")
    ax.set_ylabel("WCSS (in\u00e9rcia)")
    ax.set_title("Cotovelo", fontsize=9.5, loc="left")
    ax.set_xticks(list(K_RANGE))

    ax = axes[1]
    ax.grid(color=C_GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.plot(list(K_RANGE), silhuetas, marker="o", color="#eb6834", linewidth=1.6, zorder=3)
    melhor_k = list(K_RANGE)[int(np.argmax(silhuetas))]
    ax.axvline(melhor_k, color=C_MUTED, linestyle="--", linewidth=1.0)
    ax.set_xlabel("N\u00famero de clusters (k)")
    ax.set_ylabel("\u00cdndice de silhueta")
    ax.set_title(f"Silhueta (melhor k = {melhor_k})", fontsize=9.5, loc="left")
    ax.set_xticks(list(K_RANGE))

    fig.suptitle(f"Revalida\u00e7\u00e3o de k \u2014 {nome} (base de treino completa, n={len(X)})",
                 fontsize=10.5, x=0.01, ha="left", y=1.04)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"kmeans_{nome}_cotovelo_silhueta.png"), bbox_inches="tight")
    plt.close(fig)

    return melhor_k, wcss, silhuetas


def plotar_clusters(X, labels, cols, nome, k):
    if X.shape[1] > 2:
        pontos_2d = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X)
        eixo_x, eixo_y = "Componente principal 1", "Componente principal 2"
    else:
        pontos_2d = X
        eixo_x, eixo_y = cols[0], cols[1]

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.grid(color=C_GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for c in range(k):
        mask = labels == c
        ax.scatter(pontos_2d[mask, 0], pontos_2d[mask, 1], s=6, alpha=0.3,
                   color=CORES_CICLO[c % len(CORES_CICLO)], label=f"Cluster {c}", zorder=3)
    ax.set_xlabel(eixo_x)
    ax.set_ylabel(eixo_y)
    ax.set_title(f"K-Means (k={k}) \u2014 {nome}", fontsize=10.5, loc="left", pad=10)
    ax.legend(frameon=False, fontsize=8, markerscale=2.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"kmeans_{nome}_clusters.png"), bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    df = pd.read_csv(TRAIN_PATH)

    for nome, cols in FEATURE_SETS.items():
        print(f"\n=== {nome} ({', '.join(cols)}) ===")
        X = df[cols].values

        melhor_k, wcss, silhuetas = cotovelo_e_silhueta(X, nome)
        print(f"k revalidado (maior silhueta, base completa): {melhor_k}")
        for k, w, s in zip(K_RANGE, wcss, silhuetas):
            marca = "  <-- escolhido" if k == melhor_k else ""
            print(f"  k={k:2d}  WCSS={w:12.1f}  silhueta={s:.4f}{marca}")

        km_final = KMeans(n_clusters=melhor_k, n_init=10, random_state=RANDOM_STATE).fit(X)
        labels = km_final.labels_

        plotar_clusters(X, labels, cols, nome, melhor_k)

        # Atribuicoes: features originais (padronizadas) + cluster + alvo
        saida = df[cols + [TARGET]].copy()
        saida["cluster"] = labels
        saida.to_csv(os.path.join(REPORT_DIR, f"kmeans_{nome}_atribuicoes.csv"), index=False)

        # Perfil de cada cluster: media das features + tamanho + taxa de
        # inadimplencia -- e o que da substancia pra uma leitura de negocio
        # dos clusters, se vocês decidirem usar isso depois.
        perfil = saida.groupby("cluster").agg(
            **{c: (c, "mean") for c in cols},
            n_clientes=(TARGET, "size"),
            taxa_inadimplencia=(TARGET, "mean"),
        ).round(4)
        perfil["taxa_inadimplencia"] = (perfil["taxa_inadimplencia"] * 100).round(2)
        perfil.to_csv(os.path.join(REPORT_DIR, f"kmeans_{nome}_perfil_clusters.csv"))
        print(perfil.to_string())

    print(f"\nFiguras em {FIG_DIR}/, tabelas em {REPORT_DIR}/")


if __name__ == "__main__":
    main()