# 3.4.6 Dicionário de Dados Final

O dicionário abaixo descreve as **31 colunas** (30 preditoras e 1 alvo)
resultantes do pipeline da Seção 3.4, presentes nos arquivos
`data/processed/treino_processado.csv` (23.972 registros) e
`data/processed/teste_processado.csv` (5.993 registros).

Convenções da coluna **Origem**: *Original* — variável presente na base da UCI,
mantida sem alteração de conteúdo; *Derivada* — variável criada na etapa de
transformação; *Codificada* — coluna binária gerada pelo *one-hot encoding*.

O índice *i* das variáveis mensais indica a distância até o mês de referência:
*i* = 1 corresponde a setembro de 2005 (mês mais recente observado) e *i* = 6, a
abril de 2005. A variável-alvo refere-se a outubro de 2005.

**Tabela 6 — Dicionário de dados final.**

| # | Variável | Origem | Tipo | Domínio original | Transformação aplicada | Descrição |
|---|---|---|---|---|---|---|
| 1 | `LIMIT_BAL` | Original (X1) | Contínua | 10.000 a 1.000.000 NT$ | Padronizada (z-score) | Limite de crédito concedido ao cliente e à sua família |
| 2 | `AGE` | Original (X5) | Contínua | 21 a 79 anos | Padronizada (z-score) | Idade do cliente |
| 3 | `PAY_1` | Original (X6) | Ordinal | −2 a 8 | Renomeada de `PAY_0`; padronizada | Status de pagamento no mês *i* = 1. −2 = sem consumo; −1 = pagamento integral; 0 = uso do rotativo; 1 a 8 = meses de atraso |
| 4 | `PAY_2` | Original (X7) | Ordinal | −2 a 8 | Padronizada | Status de pagamento no mês *i* = 2 |
| 5 | `PAY_3` | Original (X8) | Ordinal | −2 a 8 | Padronizada | Status de pagamento no mês *i* = 3 |
| 6 | `PAY_4` | Original (X9) | Ordinal | −2 a 8 | Padronizada | Status de pagamento no mês *i* = 4 |
| 7 | `PAY_5` | Original (X10) | Ordinal | −2 a 8 | Padronizada | Status de pagamento no mês *i* = 5 |
| 8 | `PAY_6` | Original (X11) | Ordinal | −2 a 8 | Padronizada | Status de pagamento no mês *i* = 6 |
| 9 | `BILL_AMT1` | Original (X12) | Contínua | −165.580 a 964.511 NT$ | Padronizada | Valor da fatura no mês *i* = 1. Valores negativos indicam saldo credor |
| 10 | `BILL_AMT2` | Original (X13) | Contínua | −69.777 a 983.931 NT$ | Padronizada | Valor da fatura no mês *i* = 2 |
| 11 | `BILL_AMT3` | Original (X14) | Contínua | −157.264 a 1.664.089 NT$ | Padronizada | Valor da fatura no mês *i* = 3 |
| 12 | `BILL_AMT4` | Original (X15) | Contínua | −170.000 a 891.586 NT$ | Padronizada | Valor da fatura no mês *i* = 4 |
| 13 | `BILL_AMT5` | Original (X16) | Contínua | −81.334 a 927.171 NT$ | Padronizada | Valor da fatura no mês *i* = 5 |
| 14 | `BILL_AMT6` | Original (X17) | Contínua | −339.603 a 961.664 NT$ | Padronizada | Valor da fatura no mês *i* = 6 |
| 15 | `PAY_AMT1` | Original (X18) | Contínua | 0 a 873.552 NT$ | Padronizada | Valor pago no mês *i* = 1 |
| 16 | `PAY_AMT2` | Original (X19) | Contínua | 0 a 1.684.259 NT$ | Padronizada | Valor pago no mês *i* = 2 |
| 17 | `PAY_AMT3` | Original (X20) | Contínua | 0 a 896.040 NT$ | Padronizada | Valor pago no mês *i* = 3 |
| 18 | `PAY_AMT4` | Original (X21) | Contínua | 0 a 621.000 NT$ | Padronizada | Valor pago no mês *i* = 4 |
| 19 | `PAY_AMT5` | Original (X22) | Contínua | 0 a 426.529 NT$ | Padronizada | Valor pago no mês *i* = 5 |
| 20 | `PAY_AMT6` | Original (X23) | Contínua | 0 a 528.666 NT$ | Padronizada | Valor pago no mês *i* = 6 |
| 21 | `AVG_UTIL_RATIO` | Derivada | Contínua | −0,233 a 5,364 | Padronizada | Utilização média do limite: média de `BILL_AMT1..6` ÷ `LIMIT_BAL` |
| 22 | `SEM_FATURA_POSITIVA` | Derivada | Binária | {0, 1} | Nenhuma | 1 quando nenhum dos seis meses registrou fatura positiva (936 clientes; 3,12%) |
| 23 | `AVG_PAY_RATIO` | Derivada | Contínua | 0 a 5 (winsorizada) | Winsorização em 5; padronizada | Taxa média de amortização: média de `PAY_AMTt` ÷ `BILL_AMT(t+1)`, t = 1..5, calculada apenas nos meses com fatura positiva (1 = pagou a fatura inteira). Vale 0 quando `SEM_FATURA_POSITIVA` = 1; mediana do treino nos 440 clientes cuja única fatura positiva é a mais recente |
| 24 | `N_MESES_ATRASO` | Derivada | Discreta | 0 a 6 | Padronizada | Número de meses, entre os seis observados, com `PAY_i` > 0 |
| 25 | `SEX_2` | Codificada (X2) | Binária | {0, 1} | *One-hot*; referência: masculino | 1 = feminino |
| 26 | `EDU_2` | Codificada (X3) | Binária | {0, 1} | *One-hot*; referência: pós-graduação | 1 = universidade |
| 27 | `EDU_3` | Codificada (X3) | Binária | {0, 1} | *One-hot*; referência: pós-graduação | 1 = ensino médio |
| 28 | `EDU_4` | Codificada (X3) | Binária | {0, 1} | *One-hot*; referência: pós-graduação | 1 = outros (inclui os códigos 0, 5 e 6, consolidados na Seção 3.4.1) |
| 29 | `MAR_2` | Codificada (X4) | Binária | {0, 1} | *One-hot*; referência: casado | 1 = solteiro |
| 30 | `MAR_3` | Codificada (X4) | Binária | {0, 1} | *One-hot*; referência: casado | 1 = outros (inclui o código 0, consolidado na Seção 3.4.1) |
| 31 | `default payment next month` | Original (Y) | Binária | {0, 1} | Nenhuma | **Variável-alvo.** 1 = inadimplência no mês seguinte; 0 = adimplência |

## Notas sobre a leitura da tabela

**Variáveis padronizadas.** As 23 variáveis marcadas como "Padronizada"
passaram por `StandardScaler` ajustado no conjunto de treino: nos arquivos
processados apresentam média 0 e desvio-padrão 1, e os valores da coluna
*Domínio original* referem-se à escala anterior à transformação. A coluna
`SEM_FATURA_POSITIVA` e as seis colunas do *one-hot encoding* permanecem em
0/1.

**Categorias de referência.** Como o *one-hot encoding* foi aplicado com
`drop_first=True`, a categoria de referência de cada atributo nominal é
identificada pela ausência: um cliente masculino tem `SEX_2` = 0; um cliente
com pós-graduação tem `EDU_2` = `EDU_3` = `EDU_4` = 0; um cliente casado tem
`MAR_2` = `MAR_3` = 0.

**Variáveis descartadas em relação à base bruta.** A coluna `ID` (identificador
sequencial) foi removida no pipeline e não integra o dicionário final. As
variáveis `SEX`, `EDUCATION` e `MARRIAGE` deixaram de existir como colunas
próprias, tendo sido substituídas pelas seis colunas binárias correspondentes.

**Correspondência com a nomenclatura da base original.** A coluna *Origem*
indica entre parênteses o identificador usado na documentação da UCI (X1 a X23
e Y), permitindo o cruzamento direto com a Tabela 2 da Seção 3.2.
