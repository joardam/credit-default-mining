"""
Análise Descritiva dos Dados — Predição de Inadimplência em Cartões de Crédito
UPE — Escola Politécnica de Pernambuco

Gera as estatísticas e as figuras que sustentam a seção 3.3 do artigo.

Uso:
    python src/analise_descritiva.py
Saída:
    reports/figures/*.png
    reports/estatisticas_descritivas.json
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAW_PATH = os.path.join("data", "raw", "default_of_credit_card_clients.xls")
FIG_DIR = os.path.join("reports", "figures")
REPORT_DIR = "reports"
TARGET = "default payment next month"

# Paleta categórica (dois níveis, validada para daltonismo e contraste).
C_ADIMP = "#2a78d6"   # slot 1 - azul
C_INAD = "#eb6834"    # slot 2 - laranja
C_GRID = "#d8d7d2"
C_TEXT = "#0b0b0b"
C_MUTED = "#52514e"

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 9,
    "font.family": "DejaVu Sans",
    "axes.edgecolor": C_MUTED,
    "axes.labelcolor": C_TEXT,
    "text.color": C_TEXT,
    "xtick.color": C_MUTED,
    "ytick.color": C_MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def _grid(ax, axis="y"):
    ax.grid(axis=axis, color=C_GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def carregar():
    df = pd.read_excel(RAW_PATH, header=1)
    return df.rename(columns={"PAY_0": "PAY_1"})


def fig1_distribuicao_alvo(df, stats):
    """Barras — magnitude de duas categorias. Rótulos diretos, sem legenda."""
    cont = df[TARGET].value_counts().sort_index()
    pct = cont / len(df) * 100
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    _grid(ax)
    bars = ax.bar(["Adimplente (0)", "Inadimplente (1)"], cont.values,
                  color=[C_ADIMP, C_INAD], width=0.55, zorder=3)
    for b, c, p in zip(bars, cont.values, pct.values):
        ax.text(b.get_x() + b.get_width() / 2, c + 400, f"{c:,}\n({p:.1f}%)".replace(",", "."),
                ha="center", va="bottom", fontsize=8.5, color=C_TEXT)
    ax.set_ylabel("Nº de clientes")
    ax.set_ylim(0, cont.max() * 1.22)
    ax.set_title("Distribuição da variável-alvo", fontsize=10, loc="left", pad=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig1_distribuicao_alvo.png"), bbox_inches="tight")
    plt.close(fig)
    stats["alvo"] = {"adimplentes": int(cont[0]), "inadimplentes": int(cont[1]),
                     "pct_inadimplentes": round(float(pct[1]), 2)}


def fig2_taxa_por_pay1(df, stats):
    """Taxa de inadimplência por status de pagamento mais recente."""
    g = df.groupby("PAY_1")[TARGET].agg(["count", "mean"])
    g = g[g["count"] >= 10]  # códigos com amostra irrelevante ficam de fora
    taxa_geral = df[TARGET].mean()
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    _grid(ax)
    cores = [C_INAD if v > taxa_geral else C_ADIMP for v in g["mean"]]
    ax.bar(g.index.astype(str), g["mean"] * 100, color=cores, width=0.62, zorder=3)
    ax.axhline(taxa_geral * 100, color=C_MUTED, linewidth=1.0, linestyle="--", zorder=4)
    ax.text(-0.42, taxa_geral * 100 + 2.0, f"média geral {taxa_geral*100:.1f}%",
            fontsize=8, color=C_MUTED, ha="left")
    for i, (v, n) in enumerate(zip(g["mean"], g["count"])):
        ax.text(i, v * 100 + 1.2, f"{v*100:.0f}%", ha="center", fontsize=8, color=C_TEXT)
    ax.set_xlabel("PAY_1 — status do pagamento no mês mais recente")
    ax.set_ylabel("Taxa de inadimplência (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Inadimplência por status de pagamento recente", fontsize=10, loc="left", pad=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig2_taxa_por_pay1.png"), bbox_inches="tight")
    plt.close(fig)
    stats["taxa_por_pay1"] = {str(k): round(float(v), 4) for k, v in g["mean"].items()}
    stats["contagem_por_pay1"] = {str(k): int(v) for k, v in g["count"].items()}


def fig3_boxplots(df, stats):
    """Boxplots de LIMIT_BAL e AGE — evidência do critério IQR de outliers."""
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8))
    for ax, col, label in zip(axes, ["LIMIT_BAL", "AGE"],
                              ["Limite de crédito (NT$)", "Idade (anos)"]):
        _grid(ax, axis="x")
        bp = ax.boxplot(df[col], vert=False, widths=0.45, patch_artist=True,
                        flierprops=dict(marker="o", markersize=2.5,
                                        markerfacecolor=C_INAD,
                                        markeredgecolor="none", alpha=0.35),
                        medianprops=dict(color="white", linewidth=1.4),
                        boxprops=dict(facecolor=C_ADIMP, edgecolor="none"),
                        whiskerprops=dict(color=C_MUTED, linewidth=1.0),
                        capprops=dict(color=C_MUTED, linewidth=1.0), zorder=3)
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((df[col] < lo) | (df[col] > hi)).sum())
        ax.set_yticks([])
        ax.set_xlabel(label)
        ax.set_title(f"{col} — {n_out} outliers (IQR)", fontsize=9, loc="left", pad=8)
        stats.setdefault("iqr", {})[col] = {
            "q1": float(q1), "q3": float(q3), "iqr": float(iqr),
            "limite_inferior": float(lo), "limite_superior": float(hi),
            "n_outliers": n_out, "pct_outliers": round(n_out / len(df) * 100, 2),
            "min": float(df[col].min()), "max": float(df[col].max()),
            "media": round(float(df[col].mean()), 2),
            "mediana": float(df[col].median()),
            "desvio_padrao": round(float(df[col].std()), 2),
        }
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig3_boxplots_limit_age.png"), bbox_inches="tight")
    plt.close(fig)


def fig4_correlacao(df, stats):
    """Correlação de Pearson com o alvo — ordenada, barras horizontais."""
    corr = (df.drop(columns=["ID"]).corr()[TARGET]
            .drop(TARGET).sort_values())
    fig, ax = plt.subplots(figsize=(5.4, 5.6))
    _grid(ax, axis="x")
    cores = [C_INAD if v > 0 else C_ADIMP for v in corr.values]
    ax.barh(corr.index, corr.values, color=cores, height=0.68, zorder=3)
    ax.axvline(0, color=C_MUTED, linewidth=0.8)
    for i, v in enumerate(corr.values):
        off = 0.006 if v >= 0 else -0.006
        ax.text(v + off, i, f"{v:.3f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=7.5, color=C_TEXT)
    ax.set_xlim(corr.min() - 0.09, corr.max() + 0.09)
    ax.set_xlabel("Coeficiente de correlação de Pearson com o alvo")
    ax.set_title("Correlação linear das variáveis com a inadimplência",
                 fontsize=10, loc="left", pad=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig4_correlacao_alvo.png"), bbox_inches="tight")
    plt.close(fig)
    stats["correlacao_alvo"] = {k: round(float(v), 4) for k, v in
                               corr.sort_values(key=abs, ascending=False).items()}


def fig5_taxa_por_categoria(df, stats):
    """Taxa de inadimplência por atributo demográfico."""
    mapas = {
        "SEX": {1: "Masculino", 2: "Feminino"},
        "EDUCATION": {0: "0 (n/d)", 1: "Pós-grad.", 2: "Universidade",
                      3: "Ensino médio", 4: "Outros", 5: "5 (n/d)", 6: "6 (n/d)"},
        "MARRIAGE": {0: "0 (n/d)", 1: "Casado", 2: "Solteiro", 3: "Outros"},
    }
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.9))
    taxa_geral = df[TARGET].mean() * 100
    for ax, col in zip(axes, mapas):
        _grid(ax)
        g = df.groupby(col)[TARGET].agg(["count", "mean"])
        rot = [mapas[col].get(i, str(i)) for i in g.index]
        ax.bar(rot, g["mean"] * 100, color=C_ADIMP, width=0.6, zorder=3)
        ax.axhline(taxa_geral, color=C_MUTED, linewidth=1.0, linestyle="--", zorder=4)
        for i, (v, n) in enumerate(zip(g["mean"], g["count"])):
            ax.text(i, v * 100 + 0.8, f"{v*100:.1f}", ha="center", fontsize=7.5, color=C_TEXT)
        ax.set_title(col, fontsize=9, loc="left", pad=8)
        ax.set_ylim(0, 32)
        ax.tick_params(axis="x", labelrotation=45, labelsize=7.5)
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right")
        stats.setdefault("taxa_por_categoria", {})[col] = {
            str(k): {"n": int(n), "taxa": round(float(v), 4)}
            for k, (n, v) in g[["count", "mean"]].iterrows()
        }
    axes[0].set_ylabel("Taxa de inadimplência (%)")
    fig.suptitle("Inadimplência por atributo demográfico (linha tracejada = média geral)",
                 fontsize=10, x=0.01, ha="left", y=1.04)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig5_taxa_por_categoria.png"), bbox_inches="tight")
    plt.close(fig)


def estatisticas_gerais(df, stats):
    bill = [f"BILL_AMT{i}" for i in range(1, 7)]
    payamt = [f"PAY_AMT{i}" for i in range(1, 7)]
    stats["shape"] = list(df.shape)
    stats["valores_ausentes"] = int(df.isna().sum().sum())
    stats["duplicatas_sem_id"] = int(df.drop(columns=["ID"]).duplicated().sum())
    stats["education_nao_documentado"] = int(df["EDUCATION"].isin([0, 5, 6]).sum())
    stats["marriage_nao_documentado"] = int((df["MARRIAGE"] == 0).sum())
    stats["faturas_negativas_celulas"] = int((df[bill] < 0).sum().sum())
    stats["faturas_negativas_clientes"] = int((df[bill] < 0).any(axis=1).sum())
    stats["faturas_zero_celulas"] = int((df[bill] == 0).sum().sum())
    stats["descritiva"] = json.loads(
        df[["LIMIT_BAL", "AGE"] + bill + payamt].describe().round(2).to_json())
    stats["pay1_distribuicao"] = {str(k): int(v) for k, v in
                                  df["PAY_1"].value_counts().sort_index().items()}


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    df = carregar()
    stats = {}
    estatisticas_gerais(df, stats)
    fig1_distribuicao_alvo(df, stats)
    fig2_taxa_por_pay1(df, stats)
    fig3_boxplots(df, stats)
    fig4_correlacao(df, stats)
    fig5_taxa_por_categoria(df, stats)
    with open(os.path.join(REPORT_DIR, "estatisticas_descritivas.json"), "w",
              encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"Figuras geradas em {FIG_DIR}/")
    print(json.dumps({k: stats[k] for k in
                      ["shape", "valores_ausentes", "duplicatas_sem_id",
                       "faturas_negativas_clientes"]}, indent=2))


if __name__ == "__main__":
    main()
