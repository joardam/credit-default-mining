"""
Agrupamento com K-Means -- escolha de k, ajuste final e perfil dos clusters
Projeto: Mineracao de Dados Aplicada a Predicao de Inadimplencia

Entrada:
    data/processed/treino_limpo.csv -- gerado pelo preprocessamento.py: mesma
    base de treino (mesmo split) do pipeline principal, mas SEM escalonamento,
    para que medias e graficos fiquem em unidades reais. Nao ha mais
    reconstrucao do pipeline a partir do xls bruto.

Conjuntos de features:
    reduzidas             -> LIMIT_BAL, AGE, AVG_UTIL_RATIO, AVG_PAY_RATIO,
                             N_MESES_ATRASO
    perfil_uso_credito    -> AVG_UTIL_RATIO, AVG_PAY_RATIO
    perfil_socioeconomico -> LIMIT_BAL, AGE

Clientes sem nenhuma fatura positiva (SEM_FATURA_POSITIVA = 1):
    ficam FORA do agrupamento nos conjuntos que usam AVG_UTIL_RATIO e
    AVG_PAY_RATIO (para eles as duas razoes valem 0 por construcao, nao por
    comportamento) e sao reportados como uma linha a parte na tabela de
    perfil. A flag em si nao entra como feature: padronizada, uma binaria com
    ~3% de 1s vira um z-score em torno de 5,5 e passa a dominar a distancia
    euclidiana, criando um cluster "por construcao".

Escalonamento:
    StandardScaler ajustado em cada conjunto, sobre as linhas efetivamente
    agrupadas. Nao e o mesmo scaler do pipeline supervisionado (que e ajustado
    em todo o treino) -- para o K-Means so importa que as features fiquem na
    mesma escala.

Escolha de k:
    k = 2..10, maior indice de silhueta. Davies-Bouldin e registrado como
    segundo indice. Quando o segundo melhor k fica a menos de 0,005 da
    silhueta maxima, o empate e registrado no resumo.

Uso:
    python src/AGRUPAMENTO/kmeans_final.py
Saida (por conjunto):
    reports/figures/kmeans_<nome>_cotovelo_silhueta.png
    reports/figures/kmeans_<nome>_clusters.png   (so conjuntos com 2 features)
    reports/kmeans_<nome>_metricas_k.csv
    reports/kmeans_<nome>_perfil_clusters.csv
    reports/kmeans_<nome>_atribuicoes.csv
    reports/kmeans_resumo.json
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import json
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

TRAIN_PATH = os.path.join("data", "processed", "treino_limpo.csv")
FIG_DIR = os.path.join("reports", "figures")
REPORT_DIR = "reports"
TARGET = "default payment next month"
RANDOM_STATE = 42
LIMIAR_EMPATE = 0.005
K_RANGE = range(2, 11)
TETO_WINSORIZACAO = 5.0

# Conjuntos em que os clientes sem fatura positiva ficam fora do agrupamento.
EXCLUI_SEM_FATURA = {"reduzidas", "perfil_uso_credito"}

FEATURE_SETS = {
    "reduzidas": ["LIMIT_BAL", "AGE", "AVG_UTIL_RATIO", "AVG_PAY_RATIO",
                  "N_MESES_ATRASO"],
    "perfil_uso_credito": ["AVG_UTIL_RATIO", "AVG_PAY_RATIO"],
    "perfil_socioeconomico": ["LIMIT_BAL", "AGE"],
}

# Rotulos (baixo, alto) usados para nomear os clusters pelo perfil. Para
# N_MESES_ATRASO e SEM_FATURA_POSITIVA so o lado "alto" costuma ser
# informativo (poucos clientes ficam no extremo baixo de atraso = "sem
# atraso", que ja e o normal da base).
ROTULOS_FEATURE = {
    "LIMIT_BAL": ("limite baixo", "limite alto"),
    "AGE": ("mais jovens", "mais velhos"),
    "AVG_UTIL_RATIO": ("baixa utiliza\u00e7\u00e3o", "alta utiliza\u00e7\u00e3o"),
    "AVG_PAY_RATIO": ("baixa amortiza\u00e7\u00e3o", "alta amortiza\u00e7\u00e3o"),
    "N_MESES_ATRASO": ("sem atraso", "atraso recorrente"),
    "SEM_FATURA_POSITIVA": ("com fatura", "sem fatura positiva"),
}

# Paleta deliberadamente sem azul (#2a78d6) e laranja (#eb6834): essas cores
# ja identificam adimplente/inadimplente no restante do artigo.
CORES_CLUSTER = ["#3fa66a", "#9463c9", "#c9a227", "#4bbfb8", "#d6477a", "#8a8d91"]
C_GRID, C_MUTED, C_TEXT = "#d8d7d2", "#52514e", "#0b0b0b"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
    "font.family": "DejaVu Sans", "axes.edgecolor": C_MUTED,
    "axes.labelcolor": C_TEXT, "text.color": C_TEXT, "xtick.color": C_MUTED,
    "ytick.color": C_MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def nomear_cluster(centro_real, medias_pop, desvios_pop, cols, top_n=2):
    """Nomeia um cluster pelas 1-2 features que mais o distinguem da media
    da populacao (em z-score), usando os rotulos legiveis definidos acima."""
    zs = [(c, (centro_real[c] - medias_pop[c]) / desvios_pop[c]) for c in cols]
    zs.sort(key=lambda t: -abs(t[1]))
    partes = []
    for col, z in zs[:top_n]:
        if abs(z) < 0.25:
            continue
        baixo, alto = ROTULOS_FEATURE[col]
        rotulo = alto if z > 0 else baixo
        # |z| > 2: o cluster esta no extremo da distribuicao, nao so acima/abaixo
        # da media -- evita dois clusters com o mesmo nome.
        partes.append(f"{rotulo} (extrema)" if abs(z) > 2 else rotulo)
    return " + ".join(partes) if partes else "perfil pr\u00f3ximo da m\u00e9dia"


def cotovelo_e_silhueta(X_scaled, nome, n_total):
    wcss, silhuetas, dbis = [], [], []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X_scaled)
        wcss.append(km.inertia_)
        silhuetas.append(silhouette_score(X_scaled, km.labels_))
        dbis.append(davies_bouldin_score(X_scaled, km.labels_))

    ks = list(K_RANGE)
    melhor_k = ks[int(np.argmax(silhuetas))]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    ax = axes[0]
    ax.grid(color=C_GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.plot(ks, wcss, marker="o", color="#52514e", linewidth=1.6, zorder=3)
    ax.set_xlabel("N\u00famero de clusters (k)")
    ax.set_ylabel("WCSS (in\u00e9rcia)")
    ax.set_title("Cotovelo", fontsize=9.5, loc="left")
    ax.set_xticks(ks)

    ax = axes[1]
    ax.grid(color=C_GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.plot(ks, silhuetas, marker="o", color="#9463c9", linewidth=1.6, zorder=3)
    ax.axvline(melhor_k, color=C_MUTED, linestyle="--", linewidth=1.0)
    ax.set_xlabel("N\u00famero de clusters (k)")
    ax.set_ylabel("\u00cdndice de silhueta")
    ax.set_title(f"Silhueta (melhor k = {melhor_k})", fontsize=9.5, loc="left")
    ax.set_xticks(ks)

    fig.suptitle(f"Escolha de k \u2014 {nome} (n={n_total})",
                 fontsize=10.5, x=0.01, ha="left", y=1.04)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"kmeans_{nome}_cotovelo_silhueta.png"), bbox_inches="tight")
    plt.close(fig)

    metricas = pd.DataFrame({"k": ks, "wcss": wcss, "silhueta": silhuetas,
                             "davies_bouldin": dbis}).round(4)
    metricas.to_csv(os.path.join(REPORT_DIR, f"kmeans_{nome}_metricas_k.csv"), index=False)
    return melhor_k, metricas


def plotar_clusters_reais(df_bruto, labels, cols, nome, nomes_cluster, taxas, k,
                           nota_winsorizacao=None):
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    ax.grid(color=C_GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for c in range(k):
        mask = labels == c
        rotulo = f"{nomes_cluster[c]} ({taxas[c]:.1f}% inadimpl\u00eancia)"
        ax.scatter(df_bruto.loc[mask, cols[0]], df_bruto.loc[mask, cols[1]],
                   s=7, alpha=0.3, color=CORES_CLUSTER[c % len(CORES_CLUSTER)],
                   label=rotulo, zorder=3)

    if nota_winsorizacao is not None:
        ax.axhline(TETO_WINSORIZACAO, color=C_MUTED, linestyle=":", linewidth=1.0, zorder=4)
        ax.text(ax.get_xlim()[0], TETO_WINSORIZACAO, f"  teto de winsoriza\u00e7\u00e3o "
                f"({nota_winsorizacao} clientes)", fontsize=7, color=C_MUTED, va="bottom")

    ax.set_xlabel(cols[0])
    ax.set_ylabel(cols[1])
    ax.set_title(f"K-Means (k={k}) \u2014 {nome} (unidades reais)", fontsize=10.5, loc="left", pad=10)
    ax.legend(frameon=False, fontsize=7.5, markerscale=2.2, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"kmeans_{nome}_clusters.png"), bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    df_treino = pd.read_csv(TRAIN_PATH)
    print(f"Treino (sem escalonamento): {len(df_treino)} linhas")
    resumo = {}

    for nome, cols in FEATURE_SETS.items():
        print(f"\n=== {nome} ({', '.join(cols)}) ===")
        base = df_treino
        linha_sem_fatura = None
        nota_winsorizacao = None

        if nome in EXCLUI_SEM_FATURA:
            sem_fatura = base["SEM_FATURA_POSITIVA"] == 1
            fora = base.loc[sem_fatura]
            linha_sem_fatura = {
                "nome_cluster": "sem fatura positiva (fora do agrupamento)",
                **{c: round(float(fora[c].mean()), 2) for c in cols},
                "n_clientes": int(len(fora)),
                "taxa_inadimplencia": round(float(fora[TARGET].mean()) * 100, 2),
            }
            print(f"  {len(fora)} clientes sem fatura positiva fora do agrupamento "
                  f"(inadimpl\u00eancia {linha_sem_fatura['taxa_inadimplencia']}%)")
            base = base.loc[~sem_fatura].reset_index(drop=True)

        if "AVG_PAY_RATIO" in cols and len(cols) == 2:
            nota_winsorizacao = int((base["AVG_PAY_RATIO"] >= TETO_WINSORIZACAO).sum())

        X_real = base[cols].values
        X_scaled = StandardScaler().fit_transform(X_real)

        melhor_k, metricas = cotovelo_e_silhueta(X_scaled, nome, len(base))
        sil_ord = metricas.sort_values("silhueta", ascending=False)
        empate = (sil_ord["silhueta"].iloc[0] - sil_ord["silhueta"].iloc[1]) < LIMIAR_EMPATE
        print(f"  k escolhido (maior silhueta): {melhor_k}"
              + (f" -- empate t\u00e9cnico com k={int(sil_ord['k'].iloc[1])}" if empate else ""))

        km_final = KMeans(n_clusters=melhor_k, n_init=10, random_state=RANDOM_STATE).fit(X_scaled)
        labels = km_final.labels_

        saida = base[cols + [TARGET]].copy()
        saida["cluster"] = labels

        medias_pop = base[cols].mean()
        desvios_pop = base[cols].std()
        perfil = saida.groupby("cluster").agg(
            **{c: (c, "mean") for c in cols},
            n_clientes=(TARGET, "size"),
            taxa_inadimplencia=(TARGET, "mean"),
        )
        perfil["taxa_inadimplencia"] = perfil["taxa_inadimplencia"] * 100
        nomes_cluster = {c: nomear_cluster(perfil.loc[c], medias_pop, desvios_pop, cols)
                         for c in perfil.index}
        perfil.insert(0, "nome_cluster", perfil.index.map(nomes_cluster))
        perfil = perfil.round(2)
        perfil_saida = perfil.reset_index()
        if linha_sem_fatura is not None:
            perfil_saida = pd.concat(
                [perfil_saida, pd.DataFrame([{"cluster": "fora", **linha_sem_fatura}])],
                ignore_index=True)
        perfil_saida.to_csv(os.path.join(REPORT_DIR, f"kmeans_{nome}_perfil_clusters.csv"), index=False)
        saida.to_csv(os.path.join(REPORT_DIR, f"kmeans_{nome}_atribuicoes.csv"), index=False)
        print(perfil_saida.to_string(index=False))

        linha_k = metricas.loc[metricas["k"] == melhor_k].iloc[0]
        resumo[nome] = {
            "features": cols,
            "n_agrupados": int(len(base)),
            "n_fora_sem_fatura": linha_sem_fatura["n_clientes"] if linha_sem_fatura else 0,
            "k_escolhido": int(melhor_k),
            "silhueta": float(linha_k["silhueta"]),
            "davies_bouldin": float(linha_k["davies_bouldin"]),
            "empate_tecnico_com_k": int(sil_ord["k"].iloc[1]) if empate else None,
        }

        if len(cols) == 2:
            taxas = perfil["taxa_inadimplencia"].to_dict()
            plotar_clusters_reais(base, labels, cols, nome, nomes_cluster, taxas,
                                  melhor_k, nota_winsorizacao)

    with open(os.path.join(REPORT_DIR, "kmeans_resumo.json"), "w", encoding="utf-8") as f:
        json.dump(resumo, f, indent=2, ensure_ascii=False)
    print(f"\nFiguras em {FIG_DIR}/, tabelas em {REPORT_DIR}/")


if __name__ == "__main__":
    main()
