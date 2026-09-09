"""
Pré-processamento de Dados — Predição de Inadimplência em Cartões de Crédito
Projeto: Mineração de Dados Aplicada à Predição de Inadimplência
UPE — Escola Politécnica de Pernambuco

Metodologia: CRISP-DM (fase de Data Preparation)
Etapas: Limpeza -> Transformação -> Redução -> Split -> Escalonamento

Uso:
    python src/preprocessamento.py
Saída:
    data/processed/treino_processado.csv
    data/processed/teste_processado.csv
    reports/estatisticas_preprocessamento.json
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
TARGET = "default payment next month"

RAW_PATH = os.path.join("data", "raw", "default_of_credit_card_clients.xls")
OUT_DIR = os.path.join("data", "processed")
REPORT_DIR = "reports"

log = {}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. CARGA DOS DADOS
    # -----------------------------------------------------------------------
    # O arquivo original da UCI traz um cabeçalho de duas linhas (X1..X23 na
    # primeira, os nomes das variáveis na segunda) -> header=1.
    df = pd.read_excel(RAW_PATH, header=1)
    log["linhas_brutas"], log["colunas_brutas"] = df.shape
    log["valores_ausentes"] = int(df.isna().sum().sum())
    print(f"[1] Base bruta: {df.shape[0]} linhas x {df.shape[1]} colunas")
    print(f"[1] Valores ausentes na base: {log['valores_ausentes']}")

    # -----------------------------------------------------------------------
    # 2. LIMPEZA (Data Cleaning)
    # -----------------------------------------------------------------------

    # 2.1 A coluna ID é um identificador sequencial, sem poder preditivo, e
    # seria interpretada como variável numérica pelos algoritmos -> removida.
    df = df.drop(columns=["ID"])

    # 2.2 Remoção de registros duplicados (perfis idênticos nas 24 colunas
    # restantes). Duplicatas não agregam informação e podem inflar métricas de
    # validação caso a mesma observação caia em treino e em teste.
    n_before = len(df)
    df = df.drop_duplicates()
    log["duplicatas_removidas"] = n_before - len(df)
    log["linhas_apos_limpeza"] = len(df)
    print(f"[2.2] Duplicatas removidas: {log['duplicatas_removidas']}")

    # 2.3 Padronização do rótulo PAY_0 -> PAY_1, mantendo a nomenclatura
    # coerente com BILL_AMT1..6 e PAY_AMT1..6 (as seis janelas mensais passam a
    # seguir o mesmo índice 1..6, sendo 1 o mês mais recente).
    df = df.rename(columns={"PAY_0": "PAY_1"})

    # 2.4 Categorias não documentadas
    # EDUCATION: o dicionário oficial define 1=pós-graduação, 2=universidade,
    # 3=ensino médio, 4=outros. Os códigos 0, 5 e 6 não têm definição e, somados,
    # representam menos de 1,2% da base; são consolidados em "4 - Outros".
    log["education_nao_documentado"] = int(df["EDUCATION"].isin([0, 5, 6]).sum())
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})

    # MARRIAGE: dicionário define 1=casado, 2=solteiro, 3=outros. O código 0 não
    # é documentado e é consolidado em "3 - Outros".
    log["marriage_nao_documentado"] = int((df["MARRIAGE"] == 0).sum())
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    print(f"[2.4] EDUCATION fora do dicionário: {log['education_nao_documentado']} | "
          f"MARRIAGE fora do dicionário: {log['marriage_nao_documentado']}")

    # 2.5 Outliers em LIMIT_BAL e AGE (critério IQR, ver seção 3.3): optou-se
    # por NÃO remover esses registros. São clientes de limite alto ou de faixa
    # etária mais avançada — um subgrupo real da população, não erro de coleta.
    # Excluí-los reduziria a capacidade de generalização do modelo para esse
    # perfil. O efeito de escala é tratado no bloco 7 (StandardScaler) e os
    # modelos baseados em árvore são naturalmente robustos a esse tipo de valor.
    for col in ["LIMIT_BAL", "AGE"]:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((df[col] < lo) | (df[col] > hi)).sum())
        log[f"outliers_iqr_{col}"] = n_out
        log[f"limites_iqr_{col}"] = [float(lo), float(hi)]
    print(f"[2.5] Outliers IQR mantidos -> LIMIT_BAL: {log['outliers_iqr_LIMIT_BAL']} | "
          f"AGE: {log['outliers_iqr_AGE']}")

    # -----------------------------------------------------------------------
    # 3. TRANSFORMAÇÃO (Feature Engineering)
    # -----------------------------------------------------------------------
    pay_cols = [f"PAY_{i}" for i in range(1, 7)]
    bill_cols = [f"BILL_AMT{i}" for i in range(1, 7)]
    payamt_cols = [f"PAY_AMT{i}" for i in range(1, 7)]

    # 3.1 AVG_UTIL_RATIO — utilização média do limite ao longo dos seis meses
    # (média do saldo faturado sobre o limite concedido). Sinaliza o quão perto
    # do teto de crédito o cliente opera, indicador clássico de behavior scoring.
    # Valores negativos são legítimos: indicam saldo credor junto ao emissor.
    df["AVG_UTIL_RATIO"] = df[bill_cols].mean(axis=1) / df["LIMIT_BAL"]

    # 3.2 AVG_PAY_RATIO — taxa média de amortização: quanto do valor faturado
    # foi efetivamente pago, mês a mês.
    # Só faz sentido calcular a razão quando existe dívida a amortizar, ou seja,
    # quando BILL_AMT > 0. Meses com fatura zero (sem consumo) ou negativa
    # (saldo credor) são EXCLUÍDOS da média em vez de convertidos em valor
    # absoluto — tomar o módulo de uma fatura negativa inverteria o significado
    # financeiro da razão.
    bill_pos = df[bill_cols].where(df[bill_cols] > 0)
    ratio = df[payamt_cols].values / bill_pos.values
    with np.errstate(invalid="ignore"):
        sem_fatura = np.isnan(ratio).all(axis=1)
        avg_ratio = np.full(len(df), np.nan)
        avg_ratio[~sem_fatura] = np.nanmean(ratio[~sem_fatura], axis=1)

    # 3.3 SEM_FATURA_POSITIVA — indicador explícito para os clientes que não
    # tiveram nenhuma fatura positiva nos seis meses. Sem essa marcação, o valor
    # 0 atribuído ao AVG_PAY_RATIO desses clientes seria indistinguível de
    # "cliente que devia e não pagou nada", que é o oposto em termos de risco.
    df["SEM_FATURA_POSITIVA"] = sem_fatura.astype(int)
    log["sem_fatura_positiva"] = int(sem_fatura.sum())

    # Winsorização do AVG_PAY_RATIO: a distribuição tem cauda extremamente longa
    # (p95 ~ 4,6 e máximo acima de 4.000) quando a fatura do mês é muito pequena
    # em relação ao pagamento. O corte em 5 preserva a ordenação dos casos de
    # pagamento acima do faturado sem deixar que valores extremos dominem o
    # escalonamento.
    log["avg_pay_ratio_acima_de_5"] = int((pd.Series(avg_ratio) > 5).sum())
    df["AVG_PAY_RATIO"] = pd.Series(avg_ratio, index=df.index).fillna(0.0).clip(upper=5)

    # 3.4 N_MESES_ATRASO — número de meses, entre os seis observados, em que o
    # cliente esteve em atraso (PAY_i > 0). Sintetiza o histórico de
    # pontualidade (X6-X11) em um único indicador de tendência.
    df["N_MESES_ATRASO"] = (df[pay_cols] > 0).sum(axis=1)

    # Observação: PAY_1 (status de pagamento mais recente) já existe na base e é
    # mantido sem alteração — é a variável de maior peso comportamental,
    # conforme evidenciado por Choi et al. (2025) e confirmado pela correlação
    # com o alvo apurada na seção 3.3.

    # -----------------------------------------------------------------------
    # 4. REDUÇÃO (Feature/Dimensionality Reduction)
    # -----------------------------------------------------------------------
    # 4.1 ID já removido no bloco 2.1 (redução de 25 para 24 colunas).
    # 4.2 As variáveis BILL_AMT1..6 apresentam correlação de Pearson próxima de
    # zero com o alvo (|r| < 0,02). São mantidas nesta etapa por sustentarem a
    # variável derivada AVG_UTIL_RATIO e por serem informativas para modelos não
    # lineares, mas ficam registradas como candidatas à remoção em uma segunda
    # rodada de seleção de atributos (RFE ou importância de atributos) durante a
    # fase de modelagem.
    # 4.3 Nenhuma técnica de balanceamento (SMOTE) é aplicada aqui: para evitar
    # vazamento de dados, o balanceamento será aplicado exclusivamente sobre o
    # conjunto de treino, após o split do bloco 6, já na fase de modelagem.

    # -----------------------------------------------------------------------
    # 5. ENCODING DE VARIÁVEIS CATEGÓRICAS
    # -----------------------------------------------------------------------
    # SEX, EDUCATION e MARRIAGE são nominais -> one-hot encoding com
    # drop_first=True. A remoção da primeira categoria elimina a colinearidade
    # perfeita entre as dummies (a soma das colunas de um mesmo atributo seria
    # constante), condição que degrada a estimação de coeficientes em Regressão
    # Logística. Categorias de referência: SEX=1 (masculino), EDUCATION=1
    # (pós-graduação) e MARRIAGE=1 (casado).
    # PAY_1..PAY_6 permanecem numéricas: a ordem dos códigos (-2 a 8) já carrega
    # significado ordinal (grau de atraso).
    df = pd.get_dummies(
        df,
        columns=["SEX", "EDUCATION", "MARRIAGE"],
        prefix=["SEX", "EDU", "MAR"],
        drop_first=True,
    )

    # -----------------------------------------------------------------------
    # 6. SEPARAÇÃO TREINO/TESTE (antes do escalonamento, para evitar vazamento)
    # -----------------------------------------------------------------------
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    log["n_treino"], log["n_teste"] = len(X_train), len(X_test)
    log["taxa_default_treino"] = round(float(y_train.mean()), 4)
    log["taxa_default_teste"] = round(float(y_test.mean()), 4)
    print(f"[6] Treino: {len(X_train)} | Teste: {len(X_test)}")
    print(f"[6] Proporção de inadimplentes -> treino: {y_train.mean():.4f} | "
          f"teste: {y_test.mean():.4f}")

    # -----------------------------------------------------------------------
    # 7. ESCALONAMENTO (Normalização)
    # -----------------------------------------------------------------------
    # StandardScaler ajustado (fit) apenas no treino e reaplicado (transform) no
    # teste, para não vazar estatísticas do conjunto de teste. Necessário para
    # modelos sensíveis à escala (Regressão Logística, k-NN, redes neurais);
    # modelos de árvore são invariantes a essa etapa, mas manter uma única base
    # permite comparar todos os classificadores sob as mesmas condições.
    # As dummies e o indicador binário SEM_FATURA_POSITIVA são deixados fora do
    # escalonamento para preservar a interpretação 0/1.
    num_cols = [
        "LIMIT_BAL", "AGE",
        *[f"PAY_{i}" for i in range(1, 7)],
        *[f"BILL_AMT{i}" for i in range(1, 7)],
        *[f"PAY_AMT{i}" for i in range(1, 7)],
        "AVG_UTIL_RATIO", "AVG_PAY_RATIO", "N_MESES_ATRASO",
    ]
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

    log["n_colunas_finais"] = int(X_train_scaled.shape[1])
    log["colunas_finais"] = list(X_train_scaled.columns)
    print(f"[7] Colunas finais ({X_train_scaled.shape[1]}): {list(X_train_scaled.columns)}")

    # -----------------------------------------------------------------------
    # 8. EXPORTAÇÃO
    # -----------------------------------------------------------------------
    X_train_scaled.assign(**{TARGET: y_train.values}).to_csv(
        os.path.join(OUT_DIR, "treino_processado.csv"), index=False)
    X_test_scaled.assign(**{TARGET: y_test.values}).to_csv(
        os.path.join(OUT_DIR, "teste_processado.csv"), index=False)
    with open(os.path.join(REPORT_DIR, "estatisticas_preprocessamento.json"), "w",
              encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    print("[8] Arquivos 'treino_processado.csv' e 'teste_processado.csv' gerados "
          f"em {OUT_DIR}/")


if __name__ == "__main__":
    main()
