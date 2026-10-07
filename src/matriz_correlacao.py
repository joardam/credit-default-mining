"""
Matriz de correlacao — features da base processada
Projeto: Mineracao de Dados Aplicada a Predicao de Inadimplencia
Etapa: Agrupamento (Clustering) — entendimento de features antes de aplicar
algoritmos de agrupamento (K-Means, DBSCAN, Hierarquico, etc.)

Uso:
    python src/correlacao_clustering.py
Saida:
    reports/figures/matriz_correlacao.png
    reports/matriz_correlacao.csv
    reports/pares_alta_correlacao.csv
"""
import os

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TRAIN_PATH = os.path.join("data", "processed", "treino_processado.csv")
FIG_DIR = os.path.join("reports", "figures")
REPORT_DIR = "reports"
TARGET = "default payment next month"

# Limiar acima do qual duas features sao consideradas redundantes para fins
# de agrupamento (distancias baseadas nessas features ficariam dominadas
# pelo par correlacionado).
LIMIAR_REDUNDANCIA = 0.7

C_TEXT, C_MUTED = "#0b0b0b", "#52514e"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 8,
    "font.family": "DejaVu Sans", "text.color": C_TEXT,
    "axes.labelcolor": C_TEXT, "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    df = pd.read_csv(TRAIN_PATH)

    # O alvo e removido da matriz: agrupamento e nao supervisionado, entao a
    # variavel-alvo nao deve orientar a formacao dos clusters nem a analise
    # de redundancia entre as features de entrada. Ela so volta depois, para
    # caracterizar os clusters encontrados (crosstab cluster x alvo).
    features = df.drop(columns=[TARGET])

    # -----------------------------------------------------------------
    # 1. Matriz de correlacao (Pearson)
    # -----------------------------------------------------------------
    corr = features.corr()
    corr.to_csv(os.path.join(REPORT_DIR, "matriz_correlacao.csv"))

    # -----------------------------------------------------------------
    # 2. Heatmap
    # -----------------------------------------------------------------
    n = len(corr.columns)
    fig, ax = plt.subplots(figsize=(max(9, n * 0.33), max(8, n * 0.30)))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(corr.columns, fontsize=7)

    # Anota so os coeficientes com |r| >= 0.4, para nao poluir a figura com
    # 900 numeros — os pares relevantes para a analise de redundancia ficam
    # legiveis, o resto fica so na escala de cor.
    for i in range(n):
        for j in range(n):
            v = corr.values[i, j]
            if i != j and abs(v) >= 0.4:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                         fontsize=6, color="white" if abs(v) > 0.6 else C_TEXT)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Coeficiente de correla\u00e7\u00e3o de Pearson")
    ax.set_title("Matriz de correla\u00e7\u00e3o — features da base processada (treino)",
                  fontsize=11, loc="left", pad=14)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "matriz_correlacao.png"), bbox_inches="tight")
    plt.close(fig)

    # -----------------------------------------------------------------
    # 3. Pares com alta correlacao (candidatos a redundantes p/ clustering)
    # -----------------------------------------------------------------
    pares = (
        corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        .stack()
        .rename("correlacao")
        .reset_index()
        .rename(columns={"level_0": "feature_1", "level_1": "feature_2"})
    )
    pares_altos = (
        pares[pares["correlacao"].abs() >= LIMIAR_REDUNDANCIA]
        .sort_values("correlacao", key=abs, ascending=False)
    )
    pares_altos.to_csv(os.path.join(REPORT_DIR, "pares_alta_correlacao.csv"), index=False)

    print(f"=== Pares com |r| >= {LIMIAR_REDUNDANCIA} ===")
    print(pares_altos.to_string(index=False))
    print()
    print(f"Figuras salvas em {FIG_DIR}/")


if __name__ == "__main__":
    main()