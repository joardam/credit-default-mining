"""
Agrupamento — exploracao comparativa (NAO e a entrega da Atividade 6)
Projeto: Mineracao de Dados Aplicada a Predicao de Inadimplencia

Compara 4 algoritmos de agrupamento (K-Means, DBSCAN, MeanShift e
Hierarquico aglomerativo com ligacao de Ward) em 4 recortes de features,
puramente para decidir qual abordagem usar na entrega oficial (secoes
2.2/3.5/4.1, que cobrem 1 algoritmo so, conforme o enunciado da Aula 6).

Mesmas decisoes do kmeans_final.py, para a comparacao valer para a entrega:
  - entrada: treino_limpo.csv (mesmo split, sem escala), com StandardScaler
    ajustado por conjunto de features;
  - clientes sem fatura positiva fora dos conjuntos que usam as razoes de
    uso/amortizacao, e a flag SEM_FATURA_POSITIVA fora das features;
  - k = 2..10 pelo maior indice de silhueta.

Hierarquico: a versao anterior usava distancia de cosseno, que mede so o
angulo em relacao a origem. Com dados padronizados a origem e o cliente
medio, entao (0,1; 0,1) e (3; 3) ficavam identicos e os clusters saiam como
fatias a partir do centro. Substituido por Ward (distancia euclidiana), o
mesmo criterio de variancia intra-cluster do K-Means.

perfil_atraso (N_MESES_ATRASO x PAY_1): mantido apenas como registro. As duas
variaveis sao inteiras e formam poucas combinacoes distintas (coluna
n_combinacoes_distintas da tabela de metricas); os "clusters" encontrados sao
essas combinacoes, por isso a silhueta alta nao indica estrutura real e o
conjunto foi descartado.

Uso:
    python src/agrupamento_exploratorio.py
Saida:
    reports/figures/clusters_<feature_set>.png   (1 por conjunto de features)
    reports/metricas_agrupamento.csv
"""
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, MeanShift, AgglomerativeClustering
from sklearn.cluster import estimate_bandwidth
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")

TRAIN_PATH = os.path.join("data", "processed", "treino_limpo.csv")
TARGET = "default payment next month"
FIG_DIR = os.path.join("reports", "figures")
REPORT_DIR = "reports"
RANDOM_STATE = 42

# Subamostra: Hierarquico aglomerativo com matriz de distancia completa e
# O(n^2) em memoria e tempo -- com n=23.972 isso significa ~574 milhoes de
# pares, inviavel no notebook de qualquer um do grupo. Por isso a comparacao
# exploratoria inteira usa a MESMA subamostra para os 4 algoritmos, o que
# tambem garante que a comparacao seja justa (mesmos pontos em todo mundo).
N_SUB = 3000

FEATURE_SETS = {
    "reduzidas": ["LIMIT_BAL", "AGE", "AVG_UTIL_RATIO", "AVG_PAY_RATIO",
                  "N_MESES_ATRASO"],
    "perfil_socioeconomico": ["LIMIT_BAL", "AGE"],
    "perfil_uso_credito": ["AVG_UTIL_RATIO", "AVG_PAY_RATIO"],
    "perfil_atraso": ["N_MESES_ATRASO", "PAY_1"],
}

EXCLUI_SEM_FATURA = {"reduzidas", "perfil_uso_credito"}

CORES_CICLO = ["#2a78d6", "#eb6834", "#3fa66a", "#9463c9", "#c9a227",
               "#d6477a", "#4bbfb8", "#8a8d91"]
COR_RUIDO = "#b5b4b0"
C_GRID, C_MUTED, C_TEXT = "#d8d7d2", "#52514e", "#0b0b0b"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
    "font.family": "DejaVu Sans", "axes.edgecolor": C_MUTED,
    "axes.labelcolor": C_TEXT, "text.color": C_TEXT, "xtick.color": C_MUTED,
    "ytick.color": C_MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def escolher_k(X, k_min=2, k_max=10):
    """Escolhe k pelo maior Indice de Silhueta no intervalo, coerente com os
    dois indices de avaliacao de agrupamento vistos na Aula 6 (Silhueta e
    Davies-Bouldin)."""
    melhor_k, melhor_sil = k_min, -1
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
        sil = silhouette_score(X, km.labels_)
        if sil > melhor_sil:
            melhor_k, melhor_sil = k, sil
    return melhor_k


def estimar_eps_dbscan(X, min_samples):
    """Heuristica do grafico k-distancia: ordena a distancia ao k-esimo
    vizinho mais proximo e usa o percentil 90 como estimativa do 'cotovelo',
    sem exigir inspecao visual manual do grafico para cada um dos 16 casos."""
    nn = NearestNeighbors(n_neighbors=min_samples).fit(X)
    dist, _ = nn.kneighbors(X)
    k_dist = np.sort(dist[:, -1])
    eps = float(np.percentile(k_dist, 90))
    if eps <= 0:
        # Atributos de baixa cardinalidade (ex.: N_MESES_ATRASO, PAY_1) geram
        # muitos pontos coincidentes -- o percentil 90 pode cair em zero.
        # Recorre-se ao menor valor positivo observado como piso de eps.
        positivos = k_dist[k_dist > 0]
        eps = float(positivos.min()) if positivos.size else 0.05
    return eps


def rotulos_validos(labels):
    """Numero de clusters efetivos, ignorando o rotulo -1 (ruido) do DBSCAN."""
    uniq = set(labels)
    uniq.discard(-1)
    return len(uniq)


def avaliar(X, labels, pct_ruido=0.0):
    n_clusters = rotulos_validos(labels)
    if n_clusters < 2:
        return n_clusters, np.nan, np.nan
    # Silhueta e Davies-Bouldin exigem rotular os pontos de ruido do DBSCAN
    # de alguma forma; a convencao aqui e exclui-los da metrica (eles nao
    # pertencem a nenhum cluster por definicao do algoritmo).
    mask = labels != -1
    if mask.sum() < 2 or len(set(labels[mask])) < 2:
        return n_clusters, np.nan, np.nan
    sil = silhouette_score(X[mask], labels[mask])
    dbi = davies_bouldin_score(X[mask], labels[mask])
    return n_clusters, sil, dbi


def rodar_algoritmos(X, k):
    resultados, params = {}, {}

    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
    resultados["K-Means"] = km.labels_
    params["K-Means"] = f"k={k}"

    min_samples = max(2 * X.shape[1], 5)
    eps = estimar_eps_dbscan(X, min_samples)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    resultados["DBSCAN"] = db.labels_
    params["DBSCAN"] = f"eps={eps:.3f}; min_samples={min_samples}"

    bw = estimate_bandwidth(X, quantile=0.2, random_state=RANDOM_STATE)
    ms = MeanShift(bandwidth=bw if bw > 0 else None, bin_seeding=True).fit(X)
    resultados["MeanShift"] = ms.labels_
    params["MeanShift"] = f"bandwidth={bw:.3f}"

    # Ward: distancia euclidiana, minimiza a variancia dentro dos grupos.
    agg = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(X)
    resultados["Hier. Ward"] = agg.labels_
    params["Hier. Ward"] = f"k={k}; ligacao=ward"

    return resultados, params


def plotar_clusters(ax, pontos_2d, labels, titulo):
    ax.grid(color=C_GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for i, c in enumerate(sorted(set(labels))):
        cor = COR_RUIDO if c == -1 else CORES_CICLO[i % len(CORES_CICLO)]
        rotulo = "Ru\u00eddo" if c == -1 else f"Cluster {c}"
        mask = labels == c
        ax.scatter(pontos_2d[mask, 0], pontos_2d[mask, 1], s=8, alpha=0.45,
                   color=cor, label=rotulo, zorder=3)
    ax.set_title(titulo, fontsize=9, loc="left")
    ax.legend(frameon=False, fontsize=6, loc="best", markerscale=1.5)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    df = pd.read_csv(TRAIN_PATH)
    linhas_metricas = []

    for nome_set, cols in FEATURE_SETS.items():
        base = df
        if nome_set in EXCLUI_SEM_FATURA:
            base = base.loc[base["SEM_FATURA_POSITIVA"] == 0]
        X_full = StandardScaler().fit_transform(base[cols].values)
        idx = np.random.RandomState(RANDOM_STATE).choice(len(X_full), size=N_SUB, replace=False)
        X = X_full[idx]
        n_comb = int(pd.DataFrame(base[cols].values).drop_duplicates().shape[0])

        k = escolher_k(X)
        resultados, params = rodar_algoritmos(X, k)

        if X.shape[1] > 2:
            pontos_2d = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X)
            eixo_x, eixo_y = "Componente principal 1", "Componente principal 2"
        else:
            pontos_2d = X
            eixo_x, eixo_y = cols[0], cols[1]

        fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.6))
        for ax, (nome_alg, labels) in zip(axes, resultados.items()):
            pct_ruido = (labels == -1).mean() * 100 if nome_alg == "DBSCAN" else 0.0
            n_clusters, sil, dbi = avaliar(X, labels)
            if nome_alg == "DBSCAN":
                titulo = f"DBSCAN ({n_clusters} grupos + {pct_ruido:.1f}% ru\u00eddo)"
            elif nome_alg == "MeanShift":
                titulo = f"MeanShift ({n_clusters} grupos encontrados)"
            else:
                titulo = f"{nome_alg} (k={n_clusters})"
            plotar_clusters(ax, pontos_2d, labels, titulo)
            linhas_metricas.append({
                "feature_set": nome_set, "algoritmo": nome_alg,
                "parametros": params[nome_alg],
                "n_clusters_efetivos": n_clusters,
                "silhueta": round(sil, 3) if sil == sil else None,
                "davies_bouldin": round(dbi, 3) if dbi == dbi else None,
                "pct_ruido_dbscan": round(pct_ruido, 1),
                "n_combinacoes_distintas": n_comb,
            })
        axes[0].set_ylabel(eixo_y)
        for ax in axes:
            ax.set_xlabel(eixo_x)
        fig.suptitle(f"Compara\u00e7\u00e3o de algoritmos \u2014 {nome_set} "
                      f"(subamostra n={N_SUB})", fontsize=11, x=0.01, ha="left", y=1.05)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, f"clusters_{nome_set}.png"), bbox_inches="tight")
        plt.close(fig)
        print(f"[{nome_set}] k escolhido via silhueta: {k}")

    metricas = pd.DataFrame(linhas_metricas)
    metricas.to_csv(os.path.join(REPORT_DIR, "metricas_agrupamento.csv"), index=False)
    print()
    print(metricas.to_string(index=False))


if __name__ == "__main__":
    main()
