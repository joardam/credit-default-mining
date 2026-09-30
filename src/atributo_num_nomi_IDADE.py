"""
Exercicio: distribuicao de frequencia, histograma, dispersao e boxplot
Atributo numerico escolhido: LIMIT_BAL
Atributo nominal escolhido: default payment next month (classe)

Uso:
    python src/atributo_num_nominal.py
Saida:
    reports/figures/hist_limit_bal.png
    reports/figures/scatter_limit_bal_age.png
    reports/figures/boxplot_limit_bal.png
    reports/distribuicao_frequencia_classe.csv
"""
import os

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAW_PATH = os.path.join("data", "raw", "default_of_credit_card_clients.xls")
FIG_DIR = os.path.join("reports", "figures")
REPORT_DIR = "reports"
TARGET = "default payment next month"
NUM = "LIMIT_BAL"

C_ADIMP, C_INAD = "#2a78d6", "#eb6834"
C_GRID, C_MUTED, C_TEXT = "#d8d7d2", "#52514e", "#0b0b0b"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
    "font.family": "DejaVu Sans", "axes.edgecolor": C_MUTED,
    "axes.labelcolor": C_TEXT, "text.color": C_TEXT, "xtick.color": C_MUTED,
    "ytick.color": C_MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def carregar_base_limpa():
    """Recria a base limpa (mesma logica do bloco 2 de preprocessamento.py),
    sem escalonar nem codificar — os graficos deste exercicio usam a escala
    original (NT$, anos) para ficarem interpretaveis."""
    df = pd.read_excel(RAW_PATH, header=1)
    df = df.drop(columns=["ID"]).drop_duplicates().rename(columns={"PAY_0": "PAY_1"})
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    return df


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    df = carregar_base_limpa()

    # -----------------------------------------------------------------
    # 1. Distribuicao de frequencia do atributo nominal (classe)
    # -----------------------------------------------------------------
    freq = df[TARGET].value_counts().sort_index()
    freq_rel = (freq / len(df) * 100).round(2)
    tabela_freq = pd.DataFrame({
        "Classe": ["Adimplente (0)", "Inadimplente (1)"],
        "Frequencia absoluta": freq.values,
        "Frequencia relativa (%)": freq_rel.values,
    })
    print("=== Distribuicao de frequencia — classe (nominal) ===")
    print(tabela_freq.to_string(index=False))
    tabela_freq.to_csv(os.path.join(REPORT_DIR, "distribuicao_frequencia_classe.csv"), index=False)

    # -----------------------------------------------------------------
    # 2. Histograma do atributo numerico
    # -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.grid(axis="y", color=C_GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.hist(df[NUM], bins=30, color=C_ADIMP, edgecolor="white", zorder=3)
    ax.set_xlabel("Limite de credito (NT$)")
    ax.set_ylabel("N\u00ba de clientes")
    ax.set_title("Histograma de LIMIT_BAL (30 classes de frequ\u00eancia)", fontsize=10, loc="left", pad=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "hist_limit_bal.png"), bbox_inches="tight")
    plt.close(fig)

    # -----------------------------------------------------------------
    # 3. Grafico de dispersao — LIMIT_BAL x AGE, colorido pela classe
    # -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.grid(color=C_GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for cls, cor, rotulo in [(0, C_ADIMP, "Adimplente"), (1, C_INAD, "Inadimplente")]:
        sub = df[df[TARGET] == cls]
        ax.scatter(sub["AGE"], sub[NUM], s=6, alpha=0.25, color=cor, label=rotulo, zorder=3)
    ax.set_xlabel("Idade (anos)")
    ax.set_ylabel("Limite de cr\u00e9dito (NT$)")
    ax.set_title("Dispers\u00e3o: LIMIT_BAL x AGE, por classe", fontsize=10, loc="left", pad=10)
    ax.legend(frameon=False, loc="upper right", markerscale=3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "scatter_limit_bal_age.png"), bbox_inches="tight")
    plt.close(fig)

    # -----------------------------------------------------------------
    # 4. Boxplot — LIMIT_BAL geral e por classe
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2), gridspec_kw={"width_ratios": [1, 2]})

    ax = axes[0]
    ax.grid(axis="y", color=C_GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.boxplot(df[NUM], widths=0.4, patch_artist=True,
               flierprops=dict(marker="o", markersize=2.5, markerfacecolor=C_MUTED,
                                markeredgecolor="none", alpha=0.35),
               medianprops=dict(color="white", linewidth=1.4),
               boxprops=dict(facecolor=C_ADIMP, edgecolor="none"),
               whiskerprops=dict(color=C_MUTED, linewidth=1.0),
               capprops=dict(color=C_MUTED, linewidth=1.0), zorder=3)
    ax.set_xticks([])
    ax.set_ylabel("Limite de cr\u00e9dito (NT$)")
    ax.set_title("Geral", fontsize=9, loc="left")

    ax = axes[1]
    ax.grid(axis="y", color=C_GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    data = [df[df[TARGET] == 0][NUM], df[df[TARGET] == 1][NUM]]
    bp = ax.boxplot(data, widths=0.5, patch_artist=True, tick_labels=["Adimplente", "Inadimplente"],
                     flierprops=dict(marker="o", markersize=2.5, markerfacecolor=C_MUTED,
                                      markeredgecolor="none", alpha=0.3),
                     medianprops=dict(color="white", linewidth=1.4),
                     whiskerprops=dict(color=C_MUTED, linewidth=1.0),
                     capprops=dict(color=C_MUTED, linewidth=1.0), zorder=3)
    for patch, cor in zip(bp["boxes"], [C_ADIMP, C_INAD]):
        patch.set_facecolor(cor)
    ax.set_title("Por classe", fontsize=9, loc="left")

    fig.suptitle("Boxplot de LIMIT_BAL", fontsize=10, x=0.01, ha="left", y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "boxplot_limit_bal.png"), bbox_inches="tight")
    plt.close(fig)

    # -----------------------------------------------------------------
    # Estatisticas de apoio (console)
    # -----------------------------------------------------------------
    print()
    print("=== LIMIT_BAL — geral ===")
    print(df[NUM].describe().round(2))
    print()
    print("=== LIMIT_BAL — por classe ===")
    print(df.groupby(TARGET)[NUM].describe().round(2))
    print()
    print(f"Figuras salvas em {FIG_DIR}/")


if __name__ == "__main__":
    main()