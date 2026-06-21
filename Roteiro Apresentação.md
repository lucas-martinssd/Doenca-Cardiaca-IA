# Roteiro de Apresentação — TCC Previsão de Insuficiência Cardíaca

**Duração:** 20 minutos | **Slides:** 15

**Banca:**
- Orientador: Prof. Dr. Leonardo Lana de Carvalho
- Avaliador: Prof. André Luiz Covre
- Avaliador: Prof. Áthila Rocha Trindade

---

## Estrutura — Visão Geral

| # | Slide | Tempo |
|---|---|---|
| 1 | Capa | 30s |
| 2 | Contextualização | 1,5 min |
| 3 | Objetivo | 1 min |
| 4 | Bancos de dados | 1,5 min |
| 5 | Pré-processamento | 1 min |
| 6 | ADALINE — teoria | 1,5 min |
| 7 | MLP — teoria | 1,5 min |
| 8 | Configurações de experimento | 30s |
| 9 | Resultados — Banco 1 (acurácia) | 2 min |
| 10 | Resultados — Banco 1 (FN clínico) | 1,5 min |
| 11 | Resultados — Banco 2 | 1,5 min |
| 12 | Comparação manual vs biblioteca | 1 min |
| 13 | Conclusão | 1,5 min |
| 14 | Referências principais | 15s |
| 15 | Obrigado / Perguntas | — |
| | **Total estimado** | **~17 min** |

---

## Roteiro de Fala

---

### Slide 1 — Capa (30s)

> "Bom dia a todos. Meu nome é Lucas, e hoje apresento meu Trabalho de Conclusão de Curso intitulado *Previsão de Insuficiência Cardíaca com Redes Neurais Artificiais*, desenvolvido sob orientação do Prof. Dr. Leonardo Lana de Carvalho. Agradeço a presença do Prof. André Luiz Covre e do Prof. Áthila Rocha Trindade na banca avaliadora."

---

### Slide 2 — Contextualização (1,5 min)

> "As doenças cardiovasculares são a principal causa de morte no mundo. No Brasil, segundo dados do Ministério da Saúde, elas respondem por cerca de 30% dos óbitos anuais, o que representa mais de 400 mil mortes por ano."
>
> "A insuficiência cardíaca, em particular, é uma condição crônica e progressiva com alta taxa de reinternação e mortalidade. O diagnóstico precoce é fundamental, mas depende de exames clínicos e laboratoriais que nem sempre estão disponíveis a tempo."
>
> "Nesse contexto, o uso de inteligência artificial para auxiliar no diagnóstico surge como uma ferramenta promissora — capaz de identificar padrões em dados clínicos e sinalizar risco antes que o quadro se agrave."

---

### Slide 3 — Objetivo (1 min)

> "O objetivo deste trabalho é desenvolver e comparar modelos de redes neurais artificiais para a previsão de doença cardíaca."
>
> "Foram implementados dois tipos de redes: o ADALINE, um modelo mais simples de neurônio único, e o MLP, uma rede de múltiplas camadas. Para cada um, fizemos duas versões: uma implementação manual em NumPy, do zero, e uma versão usando a biblioteca Scikit-learn."
>
> "Isso nos permite comparar não só a acurácia dos modelos, mas também validar se nossas implementações manuais produzem resultados equivalentes aos da biblioteca."

---

### Slide 4 — Bancos de dados (1,5 min)

> "Utilizamos dois conjuntos de dados públicos."
>
> "O Banco 1 é o *Heart Disease Dataset* da UCI, com 918 pacientes e 11 variáveis clínicas como idade, pressão arterial em repouso, colesterol, tipo de dor torácica e inclinação do segmento ST. A variável-alvo indica presença ou ausência de doença cardíaca."
>
> "O Banco 2 é o *Heart Failure Clinical Records*, com 299 pacientes e variáveis como fração de ejeção, creatinina sérica e sódio sérico. Aqui a variável-alvo é o óbito durante o acompanhamento."
>
> "Os dois bancos representam contextos clínicos distintos — diagnóstico de doença cardíaca versus prognóstico de insuficiência cardíaca —, o que enriquece a análise comparativa."

---

### Slide 5 — Pré-processamento (1 min)

> "Antes de treinar os modelos, os dados passaram por algumas etapas de preparação."
>
> "Variáveis categóricas como tipo de dor torácica foram convertidas com *one-hot encoding*. Variáveis binárias como sexo e angina por exercício foram mapeadas para zero e um. Variáveis numéricas contínuas, como idade e frequência cardíaca, foram binarizadas em faixas clínicas."
>
> "A divisão entre treino e teste seguiu a proporção 70/30, com estratificação para preservar a proporção de casos positivos. Toda essa pipeline foi implementada manualmente, sem uso de funções prontas do Scikit-learn para essa etapa."

---

### Slide 6 — ADALINE — teoria (1,5 min)

> "O ADALINE, sigla para *Adaptive Linear Neuron*, é um modelo de neurônio único que aprende ajustando seus pesos para minimizar o erro quadrático entre a saída e o valor esperado."
>
> "Neste trabalho, adaptamos o ADALINE para classificação binária aplicando a função sigmoide na saída, transformando-o em um modelo de regressão logística treinado por gradiente descendente em lote."
>
> "A regra de atualização dos pesos segue a fórmula do gradiente descendente: o peso é ajustado proporcionalmente ao erro e à taxa de aprendizado. O modelo converge quando o erro para de diminuir significativamente."

---

### Slide 7 — MLP — teoria (1,5 min)

> "O MLP, ou *Multilayer Perceptron*, é uma rede com pelo menos uma camada oculta entre a entrada e a saída. Ele consegue aprender relações não-lineares que o ADALINE não consegue capturar."
>
> "Nossa implementação usa 10 neurônios na camada oculta com função de ativação ReLU, e um neurônio de saída com sigmoide para classificação binária. Os pesos são inicializados pelo método de Xavier, que evita saturação nos primeiros passos do treinamento."
>
> "O aprendizado ocorre pelo algoritmo de *backpropagation*: calcula-se o erro na saída e propaga-se esse erro para trás, ajustando os pesos de cada camada pelo gradiente descendente."

---

### Slide 8 — Configurações de experimento (30s)

> "Cada modelo foi treinado em três configurações, variando o número de épocas e a taxa de aprendizado: C1 com 500 épocas e taxa 0,01; C2 com 1000 épocas e taxa 0,001; e C3 com 3000 épocas e taxa 0,0001. Isso nos permite observar o efeito do tempo de treinamento e da granularidade do ajuste sobre o desempenho."

---

### Slide 9 — Resultados — Banco 1 (2 min)

> "O Banco 1 conta com 918 pacientes no total."
>
> "Em acurácia, o melhor resultado foi do MLP Com Biblioteca na configuração C2, com 88,04%. O MLP Sem Biblioteca apresentou evolução consistente ao longo das configurações, partindo de 82,97% em C1 e chegando a 87,68% em C3, enquanto o MLP Com Biblioteca caiu para 82,61% em C3, indicando sensibilidade diferente ao número de épocas. O ADALINE, nas versões Sem Biblioteca e Com Biblioteca, convergiu para resultados próximos entre si, entre 85,87% e 86,96%, o que era esperado, pois ambos seguem o mesmo algoritmo."
>
> "Em relação aos falsos negativos, o destaque clínico é o MLP Sem Biblioteca em C3, com apenas 15 falsos negativos. Em contraste, o MLP Com Biblioteca em C3 atingiu 35 falsos negativos — o pior resultado do Banco 1 —, apesar de ter a maior acurácia em C2. O ADALINE apresentou um panorama intermediário: a versão Sem Biblioteca melhorou progressivamente com o treinamento, partindo de 28 falsos negativos em C1 e chegando a 20 em C3, enquanto a versão Com Biblioteca manteve 21 falsos negativos estáveis nas três configurações, demonstrando consistência, mas sem evolução."
>
> "Considerando os dois critérios em conjunto, o MLP Sem Biblioteca em C3 se destaca como o modelo mais equilibrado: 87,68% de acurácia e apenas 15 falsos negativos, combinando bom desempenho geral com a maior segurança clínica. O ADALINE, apesar de resultados levemente inferiores, se mostrou a opção mais estável entre as configurações, uma vantagem relevante quando a previsibilidade do modelo é importante."

---

### Slide 10 — Resultados — Banco 1 (FN clínico) (1,5 min)

> "Acurácia nem sempre é o critério mais relevante em diagnóstico clínico. Um falso negativo — dizer que o paciente está saudável quando ele tem a doença — é muito mais grave que um falso positivo."
>
> "Por isso, analisamos também os falsos negativos. O MLP NumPy em C3 obteve apenas 15 falsos negativos, com sensibilidade de 90,2% — o melhor resultado clínico do Banco 1. O MLP Sklearn em C2, apesar de ter maior acurácia, teve 20 falsos negativos."
>
> "Isso reforça que a escolha do modelo deve considerar o contexto de aplicação: quando o custo de um diagnóstico perdido é alto, minimizar falsos negativos é prioritário."

---

### Slide 11 — Resultados — Banco 2 (1,5 min)

> "No Banco 2, com apenas 90 pacientes no teste, os resultados foram mais modestos — o que é esperado dado o tamanho reduzido do conjunto."
>
> "O MLP NumPy em C1 apresentou o melhor equilíbrio: 78,89% de acurácia e apenas 9 falsos negativos. Um ponto importante aqui é que, neste banco, os MLPs pioram com mais épocas — o MLP NumPy cai de 78,89% em C1 para 71,11% em C3. Esse comportamento é característico de overfitting: o modelo memoriza os dados de treino e perde capacidade de generalização."
>
> "O ADALINE Sklearn, por sua vez, manteve o mesmo resultado nas três configurações, demonstrando estabilidade, porém com menor desempenho."

---

### Slide 12 — Comparação manual vs biblioteca (1 min)

> "Um dos objetivos do trabalho era validar as implementações manuais. Os resultados mostram que o ADALINE NumPy produziu resultados idênticos ou muito próximos ao ADALINE Sklearn em praticamente todas as configurações — o que confirma a correção da implementação."
>
> "Para o MLP, os resultados divergem mais, pois as implementações usam estratégias levemente diferentes de otimização. Ainda assim, o MLP NumPy superou o Sklearn no Banco 1 em C3, o que demonstra que a implementação manual não é apenas didática — ela é competitiva."

---

### Slide 13 — Conclusão (1,5 min)

> "Este trabalho demonstrou que é possível construir modelos de redes neurais do zero, sem bibliotecas de alto nível, e obter resultados competitivos com as ferramentas consolidadas do mercado."
>
> "O melhor resultado de acurácia foi do MLP Sklearn no Banco 1, com 88,04%. O melhor resultado clínico, considerando falsos negativos, foi do MLP NumPy, com apenas 15 casos não detectados e sensibilidade de 90,2%."
>
> "Como limitação, o Banco 2 tem poucos pacientes, o que limita a generalização dos resultados nesse contexto. Como trabalhos futuros, a aplicação de técnicas como *dropout*, validação cruzada e uso de datasets maiores poderia melhorar ainda mais o desempenho."

---

### Slide 14 — Referências (15s)

> "As principais referências utilizadas incluem os artigos originais dos datasets, as diretrizes brasileiras de cardiologia de 2025, e os trabalhos clássicos de Widrow e Hoff sobre ADALINE e Rumelhart sobre backpropagation."

---

### Slide 15 — Obrigado / Perguntas

> "Agradeço a atenção de todos. Fico à disposição para responder às perguntas da banca."

---

## Dicas para a apresentação

- Tenha os números das tabelas de resultados decorados — os slides 9, 10 e 11 são os mais vulneráveis a perguntas da banca.
- O argumento mais forte é o do falso negativo: alta acurácia não é suficiente em contexto clínico.
- Admita proativamente a limitação do Banco 2 (apenas 299 pacientes) — mostra maturidade na análise.
- Guarde ~3 min de margem para transições e imprevistos.
