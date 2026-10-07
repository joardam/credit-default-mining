# Mudanças em relação à primeira versão do script

Registro das alterações feitas em `src/preprocessamento.py` em relação à versão
inicial (`preprocessamento_inadimplencia.py`), com o motivo de cada uma. Os
arquivos em `data/processed/` foram regerados após essas correções.

## 1. One-hot encoding com `drop_first=True`

**Antes:** `pd.get_dummies(...)` sem `drop_first`, gerando 9 colunas binárias
(`SEX_1`, `SEX_2`, `EDU_1..4`, `MAR_1..3`).

**Problema:** para cada atributo, a soma das dummies é constante e igual a 1.
Isso é colinearidade perfeita — `SEX_1` é exatamente `1 − SEX_2`. Em Regressão
Logística (presente no benchmark, conforme a Seção 2.3), a matriz de
delineamento fica singular e a estimação dos coeficientes se torna instável ou
impossível.

**Agora:** uma categoria de referência é suprimida por atributo (`SEX` =
masculino, `EDUCATION` = pós-graduação, `MARRIAGE` = casado), restando 6 colunas
binárias. Nenhuma informação é perdida: a categoria de referência é identificada
pela ausência.

## 2. Taxa de amortização calculada sem valor absoluto

**Antes:** `bill_safe = df[bill_cols].replace(0, np.nan).abs()` — o denominador
da razão tomava o módulo da fatura.

**Problema:** 3.932 células têm fatura negativa (saldo credor do cliente, após
pagamento a maior ou estorno), afetando 1.930 clientes. Tomar o módulo converte
"o emissor deve ao cliente" em "o cliente deve ao emissor", invertendo o
significado financeiro da razão exatamente nos casos de menor risco.

**Agora:** a razão é calculada apenas sobre os meses com `BILL_AMT > 0`. Meses
com fatura zero ou negativa são excluídos da média, e não convertidos.

## 3. Nova variável `SEM_FATURA_POSITIVA`

**Antes:** `df["AVG_PAY_RATIO"].fillna(0)` atribuía 0 aos clientes sem razão
calculável.

**Problema:** o valor 0 passava a designar duas situações opostas em termos de
risco — "cliente que não teve fatura no período" (936 clientes, 3,12%) e
"cliente que devia e não amortizou nada". O modelo não tinha como separá-las.

**Agora:** a flag binária `SEM_FATURA_POSITIVA` marca explicitamente o primeiro
grupo. O valor 0 em `AVG_PAY_RATIO` continua sendo atribuído a esses registros,
mas agora acompanhado do indicador que os distingue.

## 4. Seleção explícita das colunas a escalonar

**Antes:** `num_cols = X_train.select_dtypes(include=["int64", "float64"])`.

**Problema:** a seleção por tipo é frágil. Ela funcionava por acidente — as
dummies do `get_dummies` são `bool` e ficavam de fora —, mas qualquer mudança de
versão do pandas ou de ordem das operações que alterasse o dtype das colunas
binárias passaria a escaloná-las silenciosamente, quebrando a interpretação 0/1.

**Agora:** `num_cols` é uma lista explícita das 23 variáveis contínuas e
ordinais.

## 5. Registro das estatísticas do pipeline

**Antes:** os números apareciam apenas como saída de `print` durante a execução.

**Agora:** o pipeline grava `reports/estatisticas_preprocessamento.json` com
contagens, limites de IQR e proporções. É esse arquivo que sustenta os valores
citados nas Seções 3.3 e 3.4 do artigo — o texto não depende de números
transcritos à mão.

## 6. Estrutura de caminhos

**Antes:** leitura e escrita na pasta corrente
(`pd.read_excel("default_of_credit_card_clients.xls")`).

**Agora:** `data/raw/` para a base bruta e `data/processed/` para as saídas,
com os caminhos montados por `os.path.join`, o que mantém o script funcional em
Linux, Windows, Colab e Kaggle.

---

## O que **não** mudou

As decisões metodológicas da versão original foram mantidas, por estarem
corretas:

- remoção do `ID` e das duplicatas;
- consolidação das categorias não documentadas em vez de exclusão dos registros;
- manutenção dos outliers de `LIMIT_BAL` e `AGE`;
- `PAY_1..6` tratadas como ordinais, sem one-hot;
- separação treino/teste **antes** do escalonamento, com estratificação e
  semente fixa;
- `StandardScaler` ajustado somente no treino;
- ausência de balanceamento nesta fase, adiado para o conjunto de treino na
  modelagem.

## 7. Alinhamento temporal do `AVG_PAY_RATIO`

**Antes:** razão `PAY_AMTi / BILL_AMTi` (pagamento e fatura do mesmo mês).

**Problema:** o pagamento registrado em um mês quita a fatura do mês anterior.
Na própria base, `PAY_AMTt` é exatamente igual a `BILL_AMT(t+1)` em 18.091
casos, contra 3.156 de igualdade com `BILL_AMTt`. A razão antiga comparava o
pagamento com a fatura errada; por isso o percentil 95 era 4,64 e 1.349
clientes passavam do teto de 5.

**Agora:** `PAY_AMTt / BILL_AMT(t+1)`, t = 1..5. O percentil 95 cai para 1,01
(1 = pagou a fatura inteira), só 67 clientes passam do teto e a correlação com o
alvo fica mais forte (r de −0,097 para −0,111). Os 440 clientes cuja única
fatura positiva é a mais recente (pagamento ainda não observado) recebem a
mediana do treino, imputada depois do split. `SEM_FATURA_POSITIVA` não mudou
(936 clientes). Todas as outras 29 colunas de `treino_processado.csv` ficaram
idênticas, e o split é o mesmo.

## 8. Base limpa sem escalonamento

O pipeline passa a exportar também `treino_limpo.csv`, `teste_limpo.csv` (mesmo
split, antes do StandardScaler) e `parametros_scaler.json`. Os scripts de
agrupamento leem essa base, em vez de reconstruir o pipeline a partir do xls.

## 9. Agrupamento

- `src/K-Mean.py` e `src/curva_de_cotovelo.py` foram substituídos por
  `src/AGRUPAMENTO/kmeans_final.py`, que gera cotovelo, silhueta e
  Davies-Bouldin para k = 2..10 no mesmo arquivo.
- `SEM_FATURA_POSITIVA` saiu das features de agrupamento: padronizada, a flag
  dominava a distância e criava um cluster por construção. Os clientes sem
  fatura ficam fora dos conjuntos que usam as razões e aparecem como linha à
  parte na tabela de perfil.
- Comparação exploratória: hierárquico com cosseno trocado por Ward
  (euclidiano); DBSCAN e MeanShift passam a informar "grupos encontrados" em
  vez de "k"; parâmetros de cada algoritmo registrados em
  `metricas_agrupamento.csv`; k = 2..10 também na exploração, para coincidir
  com o K-Means final.
