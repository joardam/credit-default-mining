# Mineração de Dados Aplicada à Predição de Inadimplência em Clientes de Cartão de Crédito

Projeto da disciplina de Mineração de Dados — Universidade de Pernambuco,
Escola Politécnica de Pernambuco, Graduação em Engenharia da Computação.

**Grupo:** Laís Silva Oliveira · Pedro Henrique França Dias · Peri de Lima
Macedo · Victor José de Carvalho Ferreira

---

## Objetivo

Desenvolver e avaliar modelos de mineração de dados para prever a ocorrência de
inadimplência em clientes de cartão de crédito a partir do histórico de
pagamentos e de variáveis cadastrais, sob a ótica de *behavior scoring*.

**Base:** *Default of Credit Card Clients* (UCI Machine Learning Repository) —
30.000 titulares de cartão em Taiwan, transações de abril a setembro de 2005,
com o alvo referente a outubro de 2005.
DOI: [10.24432/C55S3H](https://doi.org/10.24432/C55S3H)

**Metodologia:** CRISP-DM.

| Fase | Situação |
|---|---|
| Business Understanding | concluída (Seções 1 e 2 do artigo) |
| Data Understanding | concluída (Seções 3.2 e 3.3) |
| Data Preparation | concluída (Seção 3.4) |
| Modeling | em andamento |
| Evaluation | pendente |
| Deployment | fora do escopo |

---

## Estrutura do repositório

```
.
├── data/
│   ├── raw/                     # base bruta da UCI (.xls)
│   └── processed/               # saída do pipeline (treino/teste)
├── notebooks/
│   ├── 01_analise_descritiva.ipynb    # Seção 3.3
│   └── 02_preprocessamento.ipynb      # Seção 3.4
├── src/
│   ├── analise_descritiva.py    # gera as figuras e estatísticas da 3.3
│   └── preprocessamento.py      # pipeline completo, executável por CLI
├── docs/
│   ├── secao_3.3_analise_descritiva.md
│   ├── secao_3.4_preprocessamento.md
│   └── dicionario_de_dados.md   # dicionário de dados final (Tabela 6)
├── reports/
│   ├── figures/                 # figuras 1 a 5 do artigo
│   ├── estatisticas_descritivas.json
│   └── estatisticas_preprocessamento.json
├── requirements.txt
└── README.md
```

---

## Como executar

### Localmente

```bash
pip install -r requirements.txt
python src/analise_descritiva.py     # figuras e estatísticas da Seção 3.3
python src/preprocessamento.py       # gera data/processed/*.csv
```

Os scripts devem ser executados a partir da raiz do repositório.

### Google Colab / Kaggle

Abra qualquer um dos notebooks em `notebooks/`. A primeira célula localiza a
raiz do projeto e, se estiver rodando fora do repositório, faz o clone
automaticamente — basta ajustar a constante `REPO_URL` no topo da célula para o
endereço deste repositório.

Se preferir não clonar, envie manualmente o arquivo
`default of credit card clients.xls` para `data/raw/` no ambiente da sessão.

---

## Resultado do pré-processamento

| Indicador | Valor |
|---|---|
| Registros na base bruta | 30.000 |
| Valores ausentes | 0 |
| Duplicatas removidas | 35 |
| Registros após a limpeza | 29.965 |
| Categorias não documentadas recodificadas | 399 |
| Outliers (IQR) identificados e mantidos | 439 |
| Atributos derivados criados | 4 |
| Variáveis preditoras finais | 30 |
| Divisão treino / teste (estratificada) | 23.972 / 5.993 |
| Taxa de inadimplência (treino / teste) | 22,13% / 22,13% |

Detalhamento e justificativa de cada decisão em
[`docs/secao_3.4_preprocessamento.md`](docs/secao_3.4_preprocessamento.md).
O dicionário das 31 colunas resultantes está em
[`docs/dicionario_de_dados.md`](docs/dicionario_de_dados.md).

### Decisões que valem destaque

- **Outliers mantidos.** Clientes de limite alto e faixa etária avançada são um
  subgrupo real da população, não erro de coleta; excluí-los reduziria a
  generalização justamente no perfil de maior exposição financeira.
- **Faturas negativas tratadas como saldo credor.** A taxa de amortização é
  calculada apenas sobre os meses com fatura positiva, em vez de tomar o valor
  absoluto — o módulo de uma fatura negativa inverteria o significado do
  indicador.
- **`SEM_FATURA_POSITIVA`.** Flag que separa "cliente sem dívida no período" de
  "cliente que devia e não pagou nada" — situações opostas em risco que, sem a
  marcação, receberiam o mesmo valor de amortização.
- **`drop_first=True` no one-hot encoding.** Evita a colinearidade perfeita
  entre as dummies de um mesmo atributo, que inviabiliza a estimação de
  coeficientes na Regressão Logística.
- **Split antes do escalonamento e sem balanceamento nesta fase.** O
  `StandardScaler` é ajustado só no treino, e o SMOTE ficará restrito ao
  conjunto de treino na fase de modelagem — as duas medidas evitam vazamento de
  dados.

---

## Reprodutibilidade

Semente fixa (`random_state = 42`) na divisão treino/teste. Os notebooks
incluem célula final de verificação (consistência de colunas entre treino e
teste, ausência de nulos, estratificação preservada e padronização efetiva).

## Referências

1. YEH, I.-C.; LIEN, C.-H. *The comparisons of data mining techniques for the
   predictive accuracy of probability of default of credit card clients.*
   Expert Systems with Applications, v. 36, n. 2, p. 2473–2480, 2009.
2. CASTRO, L. N.; FERRARI, D. G. *Introdução à mineração de dados.* São Paulo:
   Saraiva, 2016.
3. ALAM, T. M. et al. *An Investigation of Credit Card Default Prediction in the
   Imbalanced Datasets.* IEEE Access, v. 8, p. 201173–201198, 2020.
4. LIU, X.; WANG, S. *Credit Card Default Prediction Based on XGBoost.* MIDA
   2024, p. 374–382.
5. CHOI, W. C. et al. *A Predictive Accuracy Comparison of Machine Learning
   Models for Credit Card Default Prediction.* AICCC 2025, p. 31–39.
6. YEH, I. *Default of Credit Card Clients* [Dataset]. UCI Machine Learning
   Repository, 2009.
