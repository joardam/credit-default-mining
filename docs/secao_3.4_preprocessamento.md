# 3.4 Pré-processamento dos Dados

Esta seção descreve a preparação da base para a etapa de mineração, seguindo a
fase de *Data Preparation* do CRISP-DM. O pipeline foi organizado em cinco
etapas encadeadas — limpeza, transformação, redução, separação treino/teste e
escalonamento — implementadas em `src/preprocessamento.py` e reproduzíveis pelo
notebook `notebooks/02_preprocessamento.ipynb`. A ordem das etapas não é
arbitrária: a separação treino/teste antecede o escalonamento e qualquer
técnica de balanceamento, precisamente para impedir vazamento de informação do
conjunto de teste para o de treino.

A Tabela 4 resume as etapas e seus efeitos quantitativos sobre a base.

**Tabela 4 — Síntese do pipeline de pré-processamento.**

| # | Etapa | Operação | Efeito |
|---|---|---|---|
| 1 | Limpeza | Remoção do identificador `ID` | 25 → 24 colunas |
| 2 | Limpeza | Remoção de registros duplicados | 30.000 → 29.965 registros |
| 3 | Limpeza | Padronização do rótulo `PAY_0` → `PAY_1` | — |
| 4 | Limpeza | Consolidação de categorias não documentadas | 399 registros recodificados |
| 5 | Limpeza | Decisão sobre outliers (IQR) | 439 valores atípicos mantidos |
| 6 | Transformação | Criação de 4 atributos derivados | 24 → 28 colunas |
| 7 | Transformação | *One-hot encoding* de 3 categóricas | 28 → 31 colunas |
| 8 | Redução | Diagnóstico de atributos redundantes | registrado para a modelagem |
| 9 | Split | Divisão estratificada 80/20 | 23.972 treino / 5.993 teste |
| 10 | Escalonamento | `StandardScaler` ajustado só no treino | 30 preditoras + alvo |

## 3.4.1 Limpeza

**Valores ausentes.** Conforme apurado na Seção 3.3.1, a base não contém
valores nulos. Nenhuma estratégia de imputação foi necessária.

**Remoção do identificador.** A coluna `ID` é um contador sequencial de 1 a
30.000, sem relação com o comportamento do cliente. Mantida, seria interpretada
pelos algoritmos como variável numérica ordenada, introduzindo ruído. Foi
removida como primeira operação do pipeline.

**Registros duplicados.** Após a remoção do `ID`, 35 registros mostraram-se
idênticos a outros já presentes na base nas 24 colunas restantes. Duplicatas
não acrescentam informação e, caso a mesma observação fosse alocada ao conjunto
de treino e ao de teste, produziriam uma estimativa otimista do desempenho do
modelo. A base passou a 29.965 registros.

**Padronização de nomenclatura.** A base original nomeia o status do mês mais
recente como `PAY_0`, enquanto as demais janelas seguem `PAY_2` a `PAY_6` —
descontinuidade que quebra a correspondência com `BILL_AMT1..6` e
`PAY_AMT1..6`. A variável foi renomeada para `PAY_1`, de modo que o índice *i*
passe a designar o mesmo mês em todos os três blocos de variáveis temporais.

**Categorias não documentadas.** Os 345 registros com `EDUCATION` nos códigos
0, 5 e 6 foram consolidados na categoria 4 ("outros"), e os 54 registros com
`MARRIAGE` igual a 0 foram consolidados na categoria 3 ("outros"). A decisão de
consolidar em vez de excluir preserva 1,33% da base que, de outro modo, seria
descartada por uma falha de documentação, e não por qualquer problema com o
próprio registro. A consolidação nas categorias residuais — em vez da criação
de uma categoria nova — apoia-se na evidência da Seção 3.3.4: as taxas de
inadimplência desses códigos são próximas das observadas nas categorias
"outros" e distantes das categorias regulares.

**Outliers.** O critério IQR identificou 167 valores atípicos em `LIMIT_BAL` e
272 em `AGE` (Seção 3.3.3). **Optou-se por não removê-los.** São clientes de
limite elevado ou de faixa etária mais avançada — um subgrupo real da
população-alvo, não erro de coleta. Sua exclusão comprometeria a capacidade de
generalização do modelo justamente para o perfil de maior exposição financeira
por cliente. O efeito da diferença de escala é neutralizado pelo escalonamento
(Seção 3.4.5), e os modelos baseados em árvore e *boosting* previstos no
benchmark são, por construção, invariantes a valores extremos em variáveis
contínuas.

## 3.4.2 Transformação

Quatro atributos foram derivados a partir das variáveis originais, com o
objetivo de expressar em uma única medida padrões que, na base bruta, estão
distribuídos por seis colunas mensais. A Tabela 5 apresenta as definições.

**Tabela 5 — Atributos derivados.**

| Atributo | Definição | Motivação |
|---|---|---|
| `AVG_UTIL_RATIO` | Média de `BILL_AMT1..6` dividida por `LIMIT_BAL` | Mede a proximidade do teto de crédito; torna o valor faturado comparável entre clientes de limites distintos |
| `AVG_PAY_RATIO` | Média, entre os meses com fatura positiva, de `PAY_AMTt / BILL_AMT(t+1)`, t = 1..5 | Mede a fração da fatura anterior efetivamente paga (1 = pagou tudo) |
| `SEM_FATURA_POSITIVA` | 1 se nenhum dos seis meses teve fatura positiva | Distingue ausência de dívida de inadimplemento total |
| `N_MESES_ATRASO` | Contagem de meses com `PAY_i > 0` | Sintetiza a recorrência do atraso na janela de seis meses |

**Utilização média do limite.** A Seção 3.3.5 mostrou que o valor absoluto da
fatura tem correlação praticamente nula com o alvo (|r| < 0,02). A razão entre
fatura e limite corrige o problema: exprime o comprometimento relativo da linha
de crédito, indicador consagrado em *behavior scoring*. Valores negativos são
preservados, pois representam saldo credor do cliente — informação legítima de
baixo risco.

**Taxa média de amortização.** O cálculo da razão entre o valor pago e o valor
faturado exige cuidado com os meses em que não há dívida a amortizar. Das
29.965 linhas, 1.930 clientes possuem ao menos uma fatura negativa e há 18.105
células com fatura igual a zero. Nesses meses a razão é indefinida — dividir
por zero, ou tomar o módulo de uma fatura negativa, produziria um valor que
inverte o significado financeiro do indicador. A solução adotada foi **excluir
da média os meses sem fatura positiva**, calculando o indicador apenas sobre os
meses em que efetivamente existia dívida.

O pagamento registrado em um mês quita a fatura do mês anterior: na própria
base, `PAY_AMTt` é exatamente igual a `BILL_AMT(t+1)` em 18.091 casos, contra
3.156 casos de igualdade com `BILL_AMTt`. A razão é, por isso, calculada sobre
os cinco pares observáveis `PAY_AMTt / BILL_AMT(t+1)`. Em 440 clientes a única
fatura positiva é a mais recente, cujo pagamento ainda não foi observado; para
eles o indicador recebe a mediana do conjunto de treino, imputada depois da
divisão treino/teste.

**Indicador de ausência de fatura.** Em 936 clientes (3,12% da base) nenhum dos
seis meses apresentou fatura positiva, de modo que `AVG_PAY_RATIO` fica
indefinido. Atribuir simplesmente o valor 0 a esses casos os tornaria
indistinguíveis dos clientes que deviam e nada pagaram — situações de risco
bem diferentes: 72,1% de inadimplência entre os 233 clientes que deviam e nada
pagaram, contra 36,5% entre os sem fatura, ambos acima da média de 22,1%. Criou-se, por isso, a variável binária `SEM_FATURA_POSITIVA`,
que marca explicitamente esses registros e permite ao modelo separar as duas
condições.

**Winsorização.** A distribuição de `AVG_PAY_RATIO` apresenta cauda direita
extrema: o percentil 95 situa-se em 1,01, mas o valor máximo ultrapassa 3.300,
resultado de meses em que a fatura foi de poucos NT$ e o pagamento, muito
superior. Os 67 registros (0,22%) com razão acima de 5 foram winsorizados
nesse limite. O corte preserva a ordenação entre clientes que pagam acima do
faturado sem permitir que um punhado de valores extremos domine a média e o
desvio-padrão usados no escalonamento.

**Recorrência de atraso.** `N_MESES_ATRASO` condensa em um único inteiro de 0 a
6 o padrão que a Figura 4 evidencia: não apenas o atraso corrente, mas sua
repetição ao longo da janela observada. A variável `PAY_1` é mantida sem
alteração, por ser individualmente a de maior correlação com o alvo.

## 3.4.3 Codificação de variáveis categóricas

As variáveis `SEX`, `EDUCATION` e `MARRIAGE` são nominais: seus códigos
numéricos não expressam ordem. Mantê-las como inteiros levaria os modelos
lineares a inferir, por exemplo, que "solteiro" (2) é o dobro de "casado" (1).
Aplicou-se *one-hot encoding* com **`drop_first=True`**, isto é, com a supressão
de uma categoria de referência por atributo (`SEX` = masculino, `EDUCATION` =
pós-graduação, `MARRIAGE` = casado). A supressão é necessária porque, mantidas
todas as categorias, a soma das colunas de um mesmo atributo é constante e igual
a 1, produzindo colinearidade perfeita entre as variáveis — condição que
inviabiliza a inversão da matriz na estimação dos coeficientes da Regressão
Logística. As três variáveis nominais originais dão origem a seis colunas
binárias (`SEX_2`, `EDU_2`, `EDU_3`, `EDU_4`, `MAR_2`, `MAR_3`).

As variáveis `PAY_1` a `PAY_6` **não** foram codificadas dessa forma: seus
códigos (−2 a 8) possuem ordem intrínseca — o grau de atraso —, e a codificação
binária destruiria essa informação ordinal ao custo de 60 colunas adicionais.

## 3.4.4 Redução

A redução de dimensionalidade nesta fase restringiu-se à remoção do
identificador `ID` (Seção 3.4.1). O diagnóstico de correlação da Seção 3.3.5
identificou `BILL_AMT1` a `BILL_AMT6` como candidatas à remoção, dada a
correlação linear praticamente nula com o alvo. Optou-se por **mantê-las nesta
etapa** por duas razões: sustentam o cálculo de `AVG_UTIL_RATIO` e a ausência
de correlação *linear* não implica ausência de relação não linear, capturável
pelos modelos de árvore do benchmark. A decisão fica registrada para uma
segunda rodada de seleção de atributos na fase de modelagem, por eliminação
recursiva (RFE) ou por importância de atributos, quando o critério passará a ser
o impacto medido sobre o desempenho preditivo.

Registra-se ainda que **nenhuma técnica de balanceamento de classes foi
aplicada nesta fase**. Embora o desbalanceamento de 22,12% contra 77,88%
(Seção 3.3.2) exija tratamento, aplicar SMOTE ou qualquer reamostragem antes da
separação treino/teste faria com que instâncias sintéticas geradas a partir de
registros de teste contaminassem o treino, inflando artificialmente as métricas.
O balanceamento será aplicado exclusivamente sobre o conjunto de treino, já na
fase de modelagem.

## 3.4.5 Separação treino/teste e escalonamento

A base foi dividida em 80% para treino (23.972 registros) e 20% para teste
(5.993 registros), com amostragem **estratificada** pela variável-alvo e semente
fixa (`random_state = 42`) para garantir reprodutibilidade. A estratificação
preserva a proporção de inadimplentes em ambos os conjuntos: 22,13% no treino e
22,13% no teste.

O escalonamento foi aplicado por `StandardScaler`, que transforma cada variável
para média zero e desvio-padrão unitário. Os parâmetros foram estimados
(`fit`) **exclusivamente sobre o conjunto de treino** e apenas aplicados
(`transform`) ao conjunto de teste. A inversão dessa ordem — ajustar o
escalonador sobre a base completa — é uma forma sutil de vazamento de dados: a
média e o desvio-padrão do conjunto de teste passariam a influenciar a
transformação vista pelo modelo durante o treinamento.

O escalonamento incidiu sobre as 23 variáveis contínuas e ordinais. As seis
colunas binárias resultantes do *one-hot encoding* e o indicador
`SEM_FATURA_POSITIVA` foram deixados fora da transformação, preservando a
interpretação 0/1. A normalização é imprescindível para os modelos sensíveis a
distância e magnitude (Regressão Logística, k-NN, redes neurais) e inócua para
os modelos de árvore; mantê-la em uma base única permite comparar todos os
classificadores do benchmark sob condições idênticas.

O pipeline resulta em **30 variáveis preditoras e 1 variável-alvo**, exportadas
em `data/processed/treino_processado.csv` e `data/processed/teste_processado.csv`.
O dicionário de dados final é apresentado na Seção 3.4.6.
