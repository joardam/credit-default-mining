"""
Curva de cotovelo (Elbow Method) — K-Means
Projeto: Mineracao de Dados Aplicada a Predicao de Inadimplencia
Etapa: Agrupamento — exercicio de entendimento da base (atividade avulsa,
nao integra o pipeline principal do artigo)

Testa a curva de cotovelo separadamente para 3 recortes de features,
discutidos como possiveis agrupamentos de negocio:
  1. Perfil socioeconomico   -> LIMIT_BAL, AGE
  2. Perfil de uso do credito -> AVG_UTIL_RATIO, AVG_PAY_RATIO
  3. Perfil de atraso         -> N_MESES_ATRASO, PAY_1

Uso:
    python src/curva_cotovelo.py
Saida:
    reports/figures/cotovelo_agrupamentos.png
    reports/wcss_por_agrupamento.csv
"""
import os

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

TRAIN_PATH = os.path.join("data", "processed", "treino_processado.csv")
FIG_DIR = os.path.join("reports", "figures")
REPORT_DIR = "reports"
RANDOM_STATE = 42
K_RANGE = range(1, 11)

GRUPOS = {
    "Perfil socioeconomico": ["LIMIT_BAL", "AGE"],
    "Perfil de uso do credito": ["AVG_UTIL_RATIO", "AVG_PAY_RATIO"],
    "Perfil de atraso": ["N_MESES_ATRASO", "PAY_1"],
}
CORES = ["#2a78d6", "#eb6834", "#3fa66a"]

C_GRID, C_MUTED, C_TEXT = "#d8d7d2", "#52514e", "#0b0b0b"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
    "font.family": "DejaVu Sans", "axes.edgecolor": C_MUTED,
    "axes.labelcolor": C_TEXT, "text.color": C_TEXT, "xtick.color": C_MUTED,
    "ytick.color": C_MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    # As features ja estao padronizadas (StandardScaler ajustado no treino,
    # conforme preprocessamento.py) -> nenhum escalonamento adicional e
    # necessario para o K-Means, que e sensivel a escala.
    df = pd.read_csv(TRAIN_PATH)

    linhas_wcss = []
    fig, axes = plt.subplots(1, len(GRUPOS), figsize=(5.0 * len(GRUPOS), 3.6))

    for ax, cor, (nome, cols) in zip(axes, CORES, GRUPOS.items()):
        X = df[cols].values
        wcss = []
        for k in K_RANGE:
            km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
            km.fit(X)
            wcss.append(km.inertia_)
            linhas_wcss.append({"agrupamento": nome, "k": k, "wcss": km.inertia_})

        ax.grid(color=C_GRID, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.plot(list(K_RANGE), wcss, marker="o", color=cor, linewidth=1.6, zorder=3)
        ax.set_xlabel("N\u00famero de clusters (k)")
        ax.set_ylabel("WCSS (in\u00e9rcia)")
        ax.set_title(f"{nome}\n({' x '.join(cols)})", fontsize=9.5, loc="left", pad=8)
        ax.set_xticks(list(K_RANGE))

    fig.suptitle("Curva de cotovelo por agrupamento candidato", fontsize=11, x=0.01, ha="left", y=1.04)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "cotovelo_agrupamentos.png"), bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(linhas_wcss).to_csv(
        os.path.join(REPORT_DIR, "wcss_por_agrupamento.csv"), index=False)
    print(f"Figura salva em {FIG_DIR}/cotovelo_agrupamentos.png")


if __name__ == "__main__":
    main()