# Revisão — pipeline e agrupamento

Tudo foi rodado a partir da `main` (commit 8896a8f). A lista técnica completa
das alterações está em `docs/mudancas_no_pipeline.md`, seções 7 a 9.

## Ordem de execução

```
python src/preprocessamento.py
python src/matriz_correlacao.py
python "src/AGRUPAMENTO/Agrupamento_exloratório.py"
python src/AGRUPAMENTO/kmeans_final.py
```

## 1. Correção no pipeline: alinhamento do AVG_PAY_RATIO

O pagamento de um mês quita a fatura do mês anterior. Na base, PAY_AMTt é
igual a BILL_AMT(t+1) em 18.091 casos, contra 3.156 casos de igualdade com
BILL_AMTt. A razão passou a ser PAY_AMTt / BILL_AMT(t+1).

| | Antes | Agora |
|---|---|---|
| Percentil 95 | 4,64 | 1,01 |
| Valor máximo | > 4.400 | > 3.300 |
| Clientes winsorizados (> 5) | 1.349 (4,50%) | 67 (0,22%) |
| Correlação com o alvo | −0,097 | −0,111 |

440 clientes cuja única fatura positiva é a mais recente recebem a mediana do
treino (0,1016), imputada depois do split. O split e as outras 29 colunas
ficaram idênticos.

## 2. Afirmação do artigo que precisa mudar

| Grupo | n (base inteira) | Inadimplência |
|---|---|---|
| Base inteira | 29.965 | 22,1% |
| Sem fatura positiva | 936 | 36,5% |
| Devia e não pagou nada | 233 | 72,1% |
| Teve alguma fatura negativa (saldo credor) | 1.930 | 16,5% |

"Saldo credor = baixo risco" se sustenta. "Sem fatura e devia sem pagar são
opostos em risco" não: os dois estão acima da média, só que em níveis muito
diferentes. A Seção 3.4 em `docs/` já foi ajustada.

Achado colateral: dos 936 clientes sem fatura, 584 têm PAY_1 = 1 (um mês de
atraso) mesmo sem fatura. É uma inconsistência conhecida da codificação dessa
base, e vale uma frase no artigo.

## 3. K-Means final (treino, n = 23.972)

| Conjunto | Agrupados | Fora (sem fatura) | k | Silhueta | Davies-Bouldin |
|---|---|---|---|---|---|
| reduzidas (5 features) | 23.224 | 748 | 3 | 0,297 | 1,262 |
| perfil_uso_credito | 23.224 | 748 | 4 | 0,591 | 0,494 |
| perfil_socioeconomico | 23.972 | 0 | 3 | 0,438 | 0,811 |

### reduzidas

| Grupo | n | Limite médio (NT$) | Idade | Utilização | Amortização | Meses de atraso | Inadimplência |
|---|---|---|---|---|---|---|---|
| atraso recorrente + alta utilização | 2.873 | 88.907 | 35,0 | 0,62 | 0,07 | 4,53 | 60,9% |
| alta utilização + baixa amortização | 10.740 | 112.728 | 34,7 | 0,61 | 0,11 | 0,35 | 18,4% |
| baixa utilização + alta amortização | 9.611 | 248.389 | 36,4 | 0,07 | 0,78 | 0,28 | 13,7% |
| sem fatura positiva (fora do agrupamento) | 748 | 209.171 | 36,6 | 0,00 | 0,00 | 0,63 | 36,1% |

### perfil_uso_credito

| Grupo | n | Utilização | Amortização | Inadimplência |
|---|---|---|---|---|
| alta utilização + baixa amortização | 8.504 | 0,79 | 0,08 | 28,7% |
| baixa amortização + baixa utilização | 7.396 | 0,26 | 0,16 | 20,0% |
| alta amortização + baixa utilização | 7.220 | 0,04 | 0,92 | 15,2% |
| alta amortização (extrema) + baixa utilização | 104 | 0,09 | 4,32 | 23,1% |
| sem fatura positiva (fora do agrupamento) | 748 | 0,00 | 0,00 | 36,1% |

### perfil_socioeconomico

| Grupo | n | Limite médio (NT$) | Idade | Inadimplência |
|---|---|---|---|---|
| limite alto | 5.778 | 352.802 | 36,5 | 13,7% |
| mais jovens + limite baixo | 11.642 | 104.084 | 28,9 | 24,2% |
| mais velhos + limite baixo | 6.552 | 115.937 | 46,3 | 25,8% |

## 4. Comparação exploratória (subamostra de 3.000)

O k do K-Means agora coincide com o do K-Means final nos três conjuntos (3, 4 e
3). Os números completos estão em `reports/metricas_agrupamento.csv`.

- **Hierárquico Ward** fica próximo do K-Means e ligeiramente abaixo na
  silhueta: 0,259 contra 0,292 em reduzidas; 0,383 contra 0,437 no
  socioeconômico; 0,559 contra 0,592 no uso do crédito.
- **DBSCAN** não encontra estrutura útil: um único grupo em reduzidas,
  silhueta negativa no uso do crédito, 15 grupos com silhueta de 0,05 no
  socioeconômico.
- **MeanShift** não tem resultado estável: encontra 2, 8 ou 12 grupos conforme
  o conjunto.
- **perfil_atraso** tem só 44 combinações distintas de valores, e os
  "clusters" são essas combinações. Por isso a silhueta de 0,85 a 0,99 não
  indica estrutura real, e o conjunto foi descartado. O motivo está registrado
  no script.

## 5. Decisões que ficam com vocês

1. **uso_credito com k = 4.** O quarto grupo tem 104 clientes (0,4%) e é só a
   cauda da amortização. Com k = 3 a silhueta é 0,577, contra 0,591. O script
   segue a regra da maior silhueta; se preferirem k = 3, justifiquem pelo
   tamanho mínimo de grupo.
2. **Estrutura fraca em reduzidas** (silhueta de 0,30). O texto deve falar em
   segmentação, não em grupos naturais.
3. **Números do artigo a atualizar:**
   - Seção 3.4 (winsorização): 67 clientes, p95 de 1,01, máximo acima de 3.300.
   - Tabela 4: domínio do AVG_PAY_RATIO.
   - Frase sobre os clientes sem fatura.
   - Os `.md` em `docs/` já estão corrigidos; o `.docx` não foi tocado.

## 6. O que não foi feito

- Ajustes visuais (nota 5), conforme pedido.
- `notebooks/02_preprocessamento.ipynb` ainda tem a razão antiga. O script
  `src/preprocessamento.py` é a versão correta.
- `analise_descritiva.py` não foi rodado de novo: ele lê a base bruta e não
  depende do AVG_PAY_RATIO.
- `src/K-Mean.py`, `src/curva_de_cotovelo.py`, `reports/wcss_por_agrupamento.csv`
  e `reports/figures/cotovelo_agrupamentos.png` foram removidos, porque o
  `kmeans_final.py` substitui todos eles.
