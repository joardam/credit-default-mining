"""
Exercicio: distribuicao de frequencia, histograma, dispersao e boxplot
Atributo numerico escolhido: LIMIT_BAL
Atributo nominal escolhido: EDUCATION (escolaridade)

Uso:
    python src/atributo_num_nominal.py
Saida:
    reports/figures/hist_limit_bal.png
    reports/figures/scatter_limit_bal_age_educacao.png
    reports/figures/boxplot_limit_bal_educacao.png
    reports/distribuicao_frequencia_educacao.csv
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
NOM = "EDUCATION"
NUM = "LIMIT_BAL"
ROTULOS = {1: "P\u00f3s-gradua\u00e7\u00e3o", 2: "Universidade", 3: "Ensino m\u00e9dio", 4: "Outros"}
CORES = {1: "#2a78d6", 2: "#eb6834", 3: "#3fa66a", 4: "#9463c9"}

C_GRID, C_MUTED, C_TEXT = "#d8d7d2", "#52514e", "#0b0b0b"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
    "font.family": "DejaVu Sans", "axes.edgecolor": C_MUTED,
    "axes.labelcolor": C_TEXT, "text.color": C_TEXT, "xtick.color": C_MUTED,
    "ytick.color": C_MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def carregar_base_limpa():
    df = pd.read_excel(RAW_PATH, header=1)
    df = df.drop(columns=["ID"]).drop_duplicates().rename(columns={"PAY_0": "PAY_1"})
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    return df


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    df = carregar_base_limpa()
    df["EDU_ROTULO"] = df[NOM].map(ROTULOS)

    # -----------------------------------------------------------------
    # 1. Distribuicao de frequencia do atributo nominal (EDUCATION)
    # -----------------------------------------------------------------
    freq = df[NOM].value_counts().sort_index()
    freq_rel = (freq / len(df) * 100).round(2)
    tabela_freq = pd.DataFrame({
        "Categoria": [ROTULOS[i] for i in freq.index],
        "Frequencia absoluta": freq.values,
        "Frequencia relativa (%)": freq_rel.values,
    })
    print("=== Distribuicao de frequencia — EDUCATION (nominal) ===")
    print(tabela_freq.to_string(index=False))
    tabela_freq.to_csv(os.path.join(REPORT_DIR, "distribuicao_frequencia_educacao.csv"), index=False)

    # -----------------------------------------------------------------
    # 2. Histograma do atributo numerico (nao depende do nominal)
    # -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.grid(axis="y", color=C_GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.hist(df[NUM], bins=30, color="#2a78d6", edgecolor="white", zorder=3)
    ax.set_xlabel("Limite de credito (NT$)")
    ax.set_ylabel("N\u00ba de clientes")
    ax.set_title("Histograma de LIMIT_BAL (30 classes de frequ\u00eancia)", fontsize=10, loc="left", pad=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "hist_limit_bal.png"), bbox_inches="tight")
    plt.close(fig)

    # -----------------------------------------------------------------
    # 3. Grafico de dispersao — LIMIT_BAL x AGE, um painel por categoria
    # -----------------------------------------------------------------
    # Com 4 categorias sobrepostas no mesmo eixo a nuvem de pontos fica
    # ilegivel (excesso de sobreposicao/overplotting). Pequenos multiplos
    # (um painel por categoria, mesma escala nos dois eixos) mantêm a
    # comparabilidade sem empilhar 4 cores no mesmo espaço.
    ordem = [1, 2, 3, 4]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.2), sharex=True, sharey=True)
    for ax, cod in zip(axes, ordem):
        sub = df[df[NOM] == cod]
        ax.grid(color=C_GRID, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.scatter(sub["AGE"], sub[NUM], s=5, alpha=0.25, color=CORES[cod], zorder=3)
        ax.set_title(f"{ROTULOS[cod]} (n={len(sub)})", fontsize=8.5, loc="left")
        ax.set_xlabel("Idade (anos)", fontsize=8)
    axes[0].set_ylabel("Limite de cr\u00e9dito (NT$)")
    fig.suptitle("Dispers\u00e3o: LIMIT_BAL x AGE, por escolaridade", fontsize=10, x=0.01, ha="left", y=1.04)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "scatter_limit_bal_age_educacao.png"), bbox_inches="tight")
    plt.close(fig)

    # -----------------------------------------------------------------
    # 4. Boxplot — LIMIT_BAL por categoria de EDUCATION
    # -----------------------------------------------------------------
    ordem = [1, 2, 3, 4]
    data = [df[df[NOM] == c][NUM] for c in ordem]
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    ax.grid(axis="y", color=C_GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    bp = ax.boxplot(data, widths=0.5, patch_artist=True,
                     tick_labels=[ROTULOS[c] for c in ordem],
                     flierprops=dict(marker="o", markersize=2.5, markerfacecolor=C_MUTED,
                                      markeredgecolor="none", alpha=0.3),
                     medianprops=dict(color="white", linewidth=1.4),
                     whiskerprops=dict(color=C_MUTED, linewidth=1.0),
                     capprops=dict(color=C_MUTED, linewidth=1.0), zorder=3)
    for patch, c in zip(bp["boxes"], ordem):
        patch.set_facecolor(CORES[c])
    ax.set_ylabel("Limite de cr\u00e9dito (NT$)")
    ax.set_title("Boxplot de LIMIT_BAL por escolaridade", fontsize=10, loc="left", pad=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "boxplot_limit_bal_educacao.png"), bbox_inches="tight")
    plt.close(fig)

    # -----------------------------------------------------------------
    # Estatisticas de apoio (console)
    # -----------------------------------------------------------------
    print()
    print("=== LIMIT_BAL por EDUCATION ===")
    print(df.groupby("EDU_ROTULO")[NUM].describe().round(2))
    print()
    print(f"Figuras salvas em {FIG_DIR}/")


if __name__ == "__main__":
    main()