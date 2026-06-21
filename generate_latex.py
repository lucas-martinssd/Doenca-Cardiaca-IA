import os

latex_content = r"""\chapter{Fundamentação Teórica}
\label{cap:fundamentacao}

Neste capítulo, apresentam-se os conceitos teóricos fundamentais para a compreensão da pesquisa, detalhando o funcionamento e a base histórica das arquiteturas de Redes Neurais Artificiais (RNAs). A fundamentação aqui exposta busca consolidar o suporte teórico necessário para a aplicação desses modelos em problemas complexos de previsão.

\section{Fundamentação Teórica das Redes Neurais Artificiais}
\label{sec:fundamentacao_redes}

As Redes Neurais Artificiais constituem um paradigma de computação distribuída e paralela, cujos fundamentos residem na tentativa de simular, de forma matematicamente simplificada, o processamento de informações realizado pelo sistema nervoso humano \cite{abdi1994}. Conforme Silva, Spatti e Flauzino \cite{silva2016}, tais estruturas são compostas por unidades de processamento fundamentais, os neurônios artificiais, que, por meio de conexões sinápticas ponderadas por pesos, adquirem a capacidade de extrair e aprender padrões complexos diretamente a partir de dados observacionais.

No escopo deste trabalho, a compreensão teórica dessas arquiteturas é essencial para fundamentar as escolhas metodológicas aplicadas à previsão de doenças cardíacas. A precisão do diagnóstico clínico via modelos computacionais depende essencialmente da habilidade da rede em mapear correlações sutis e interdependências não lineares presentes nas variáveis clínicas \cite{braga2007}. Assim, esta seção percorre a evolução histórica e técnica dessas tecnologias, iniciando pelos modelos associativos primários e progredindo até as arquiteturas multicamadas de alta complexidade \cite{silva2016}.

\section{Evolução Histórica e Modelos Associativos}

\subsection{Memória Auto-associativa com Aprendizagem Hebbiana}

A arquitetura de memória auto-associativa fundamentada na aprendizagem hebbiana constitui um dos pilares históricos da neurocomputação, representando a transição das teorias biológicas do aprendizado para modelos matemáticos de processamento distribuído \cite{abdi1994}. Diferente de sistemas de classificação convencionais, esta estrutura tem como objetivo primordial a simulação da capacidade de retenção e recuperação de informações, permitindo que um sistema recupere um padrão de dados integral a partir de estímulos de entrada parciais, ruidosos ou incompletos \cite{silva2016}.

No modelo auto-associativo, a rede busca associar um estímulo a ele próprio, operando como um mecanismo de auto-correção e restauração informacional. Essa funcionalidade é inspirada nos processos cognitivos de reconhecimento, em que o cérebro é capaz de recuperar uma lembrança completa a partir de fragmentos de percepção \cite{braga2007}. A base teórica desse mecanismo reside na plasticidade sináptica, conceito que descreve como as conexões entre neurônios são fortalecidas ou atenuadas em resposta à atividade celular coordenada.

\subsubsection{O Postulado de Hebb e a Correlação Sináptica}

O alicerce deste modelo foi estabelecido por Donald Hebb em 1949, cujo postulado central afirma que o processo de aprendizado neurobiológico é derivado da ativação simultânea de neurônios interconectados. Segundo Abdi \cite{abdi1994}, a formalização matemática dessa hipótese transformou-se na Regra de Hebb, a qual estipula que a variação do peso sináptico entre duas unidades de processamento é proporcional ao produto de suas ativações instantâneas. 

Se denotarmos os estados de ativação de dois neurônios $i$ e $j$ como $x_i$ e $x_j$, respectivamente, a atualização do peso $w_{ij}$ que os conecta é regida pela equação de correlação:
\begin{equation}
\Delta w_{ij} = \eta \cdot x_i \cdot x_j
\end{equation}
onde $\eta$ atua como a taxa de aprendizagem ou constante de proporcionalidade \cite{braga2007}. Este princípio garante que padrões frequentemente apresentados em conjunto reforcem fortemente as conexões de rede, gerando o que Silva, Spatti e Flauzino \cite{silva2016} descrevem como "engramas" computacionais.

A capacidade de armazenamento deste tipo de memória é formalizada por meio de matrizes de correlação. Para um conjunto de padrões de entrada $X$, a matriz de pesos de memória $W$ é obtida pelo somatório dos produtos externos dos vetores de características \cite{abdi2006}. Todavia, a eficiência da recuperação depende crucialmente da ortogonalidade dos padrões armazenados; conforme aponta Braga, Carvalho e Ludermir \cite{braga2007}, a presença de padrões linearmente dependentes introduz interferências severas, reduzindo drasticamente a capacidade de recordação fiel da rede.

\subsection{Memória Auto-associativa com Aprendizagem de Widrow-Hoff}

Diante das limitações inerentes ao aprendizado estritamente hebbiano, particularmente a interferência (\textit{crosstalk}) entre padrões não-ortogonais, o desenvolvimento da neurocomputação exigiu o estabelecimento de regras de aprendizagem orientadas à redução ativa de erros. A introdução da regra de Widrow-Hoff marcou um avanço divisor de águas, estabelecendo a fundação para o treinamento supervisionado em matrizes associativas \cite{silva2016}.

\subsubsection{A Lógica da Correção Supervisionada}

A aprendizagem de Widrow-Hoff diferencia-se fundamentalmente da abordagem hebbiana ao introduzir um mecanismo de supervisão através de retroalimentação (\textit{feedback}). Abdi e Valentin \cite{abdi2006} enfatizam que, enquanto a regra de Hebb é não-supervisionada e guiada puramente pela coativação, a regra de Widrow-Hoff altera os pesos com base na discrepância computada entre a resposta observada da rede e a resposta previamente definida como alvo.

Em um contexto de memória auto-associativa, o alvo é o próprio vetor de entrada. O modelo iterativamente compara a saída produzida $\hat{x}$ com a entrada original $x$, gerando um vetor de erro $\varepsilon = x - \hat{x}$. De acordo com Silva, Spatti e Flauzino \cite{silva2016}, a atualização sináptica é então calculada para penalizar os pesos que mais contribuíram para a geração deste erro. Matematicamente, a matriz de pesos sofre correções sucessivas orientadas pelo gradiente do erro.

Esta formulação garante que a rede progressivamente "limpe" os ruídos e se ajuste para memorizar até mesmo conjuntos de dados que apresentam alto grau de correlação linear, superando a barreira da ortogonalidade estrita exigida pelas memórias de Hebb \cite{braga2007}. A introdução deste método minimizador de erro solidificou as bases analíticas para o advento do filtro linear adaptativo.

\subsection{Memória Linear Hetero-associativa}

A evolução natural da auto-associação foi a concepção de sistemas capazes de mapear um domínio de características para um contradomínio distinto, caracterizando o que se denomina memória hetero-associativa. Ao contrário do paradigma auto-associativo que busca a correção de padrões, a memória hetero-associativa foca primordialmente na classificação, tradução de padrões e predição de saídas sob condições de incerteza \cite{abdi1994}.

\subsubsection{Formalismo Matemático da Hetero-associação}

Conforme detalhado por Braga, Carvalho e Ludermir \cite{braga2007}, uma rede hetero-associativa opera sobre pares associados $(x_k, y_k)$, buscando sintetizar uma matriz de transformação linear $W$ que satisfaça a relação $y_k = W x_k$ para todos os $K$ pares do conjunto de treinamento. Neste caso, as dimensões dos vetores de entrada $x$ e saída $y$ não precisam ser idênticas, dotando a rede de extrema flexibilidade estrutural para atuar como um aproximador universal linear.

A matriz de correlação cruzada é construída pela agregação dos tensores de primeira ordem resultantes da combinação de entradas e saídas. Abdi \cite{abdi1994} demonstra que, em condições ideais de independência linear, a rede consegue reter um número de associações igual à dimensionalidade do espaço de entrada sem perdas. Entretanto, na prática, os dados do mundo real (como registros médicos) apresentam forte multicolinearidade.

Para lidar com isso, Silva, Spatti e Flauzino \cite{silva2016} destacam que a aplicação do aprendizado de Widrow-Hoff à memória hetero-associativa é imprescindível. Através do ajuste minimizador do erro médio quadrático, a matriz $W$ converge para uma aproximação pseudo-inversa de Moore-Penrose, garantindo a classificação otimizada mesmo em cenários ruidosos e densamente agrupados.

\section{Os Elementos Adaptativos e Arquiteturas Preditivas}

\subsection{ADALINE (Adaptive Linear Neuron)}

O modelo ADALINE (\textit{Adaptive Linear Neuron} ou \textit{Element}), concebido por Bernard Widrow e Marcian Hoff em 1960, representa uma das mais prolíficas arquiteturas baseadas no processamento linear. Historicamente, ele consolidou a aplicação prática das regras de correção de erro que haviam sido teorizadas nas memórias associativas, implementando fisicamente o conceito de descida do gradiente no aprendizado de máquina \cite{silva2016}.

\subsubsection{Combinadores Lineares e o Erro Quadrático Médio}

O processamento interno do ADALINE difere de modelos antecessores, como o Perceptron simples, pela natureza do sinal que é utilizado para o treinamento. De acordo com Braga, Carvalho e Ludermir \cite{braga2007}, enquanto o Perceptron quantiza sua saída antes de calcular o erro, o ADALINE calcula seu erro diretamente a partir da saída de seu combinador linear pré-ativação. A saída analógica do combinador é dada por $v = w^T x + b$, onde $w$ representa o vetor de pesos sinápticos, $x$ o vetor de atributos de entrada e $b$ o viés (polarização) da rede.

A inovação central teórica do ADALINE, conhecida como Algoritmo LMS (\textit{Least Mean Squares}), baseia-se na formulação da função de custo como o Erro Quadrático Médio. A função objetivo $J(w)$ a ser minimizada é estritamente convexa e descrita como o valor esperado do erro ao quadrado:
\begin{equation}
J(w) = E[(d - v)^2]
\end{equation}
sendo $d$ o rótulo verdadeiro ou sinal desejado \cite{abdi2006}.

Por se tratar de uma função cujas derivadas parciais formam um hiperparaboloide, a descida do gradiente encontra inequivocamente um único mínimo global. Silva, Spatti e Flauzino \cite{silva2016} apontam que esta característica garante estabilidade absoluta durante o treinamento, uma vez que, sob uma taxa de aprendizagem $\eta$ adequadamente pequena, a rede jamais divergirá do ponto de erro mínimo. Este rigor matemático fez do ADALINE a escolha padrão para sistemas de controle adaptativo e filtros de processamento de sinais durante décadas.

\subsection{ADALINE Logística e a Classificação Probabilística}

A formulação clássica do ADALINE opera eficientemente na previsão contínua, mas apresenta limitações na interpretação dos resultados de problemas estritamente classificatórios. Ao abordar o prognóstico médico, como a detecção de doenças cardíacas (foco principal deste estudo), respostas probabilísticas oferecem um grau de confiabilidade e interpretabilidade muito superior às classificações determinísticas binárias rígidas \cite{braga2007}. Surge assim o ADALINE Logístico, ou Regressão Logística formulada via processamento neural.

\subsubsection{A Função Sigmoide e o Mapeamento Suave}

A evolução metodológica reside na inserção de uma função de ativação não-linear sigmoidal na saída do combinador linear do ADALINE. Abdi e Valentin \cite{abdi2006} observam que a função logística $\sigma(v) = (1 + \exp(-v))^{-1}$ não apenas comprime a saída do modelo para o intervalo matemático de probabilidades válidas $(0, 1)$, mas também garante que as variações nos pesos sinápticos resultem em transições de fronteira suaves.

Esta suavidade traduz-se estatisticamente em uma estimação da probabilidade a posteriori de pertencimento à classe. Se o neurônio modela o risco cardíaco, a saída $y = \sigma(v)$ passa a ser interpretada como a probabilidade $P(d=1|x)$, indicando o grau de risco do paciente com base no vetor de dados $x$ apresentado à rede \cite{silva2016}.

\subsubsection{A Entropia Cruzada Binária (\textit{Log-Loss})}

Para treinar eficientemente o ADALINE Logístico, o Erro Quadrático Médio é preterido em favor da função de custo de Entropia Cruzada Binária. Fundamentada no Princípio da Máxima Verossimilhança da teoria estatística, esta função penaliza logaritmicamente predições incorretas que apresentam alta confiança \cite{braga2007}. A função de entropia cruzada para um conjunto de treinamento é dada por:
\begin{equation}
J(w) = - \frac{1}{N} \sum_{i=1}^{N} [d_i \log(y_i) + (1 - d_i) \log(1 - y_i)]
\end{equation}

Quando o algoritmo do gradiente descendente é aplicado à entropia cruzada, a regra de atualização dos pesos alcança uma elegância matemática notável. Conforme documentado pela Data Science Academy \cite{dsa2022}, a derivada parcial do custo anula os termos exponenciais da derivada sigmoidal, resultando em uma equação de atualização estruturalmente idêntica à Regra Delta clássica do ADALINE linear: $\Delta w = \eta (d - y) x$. 

No entanto, a semântica da atualização é completamente transformada: a rede ajusta seus pesos na exata proporção do desvio de probabilidade entre sua crença preditiva e a certeza absoluta do rótulo clínico real. Esta modelagem é um dos pilares práticos implementados nas análises metodológicas desta pesquisa \cite{silva2016}.

\section{Múltiplas Camadas e o Espaço Não Linear}

\subsection{Perceptron, Funções Lógicas e o Problema XOR}

A primeira geração de redes neurais, encabeçada pelo Perceptron de Frank Rosenblatt, prometeu revolucionar a inteligência artificial fornecendo uma máquina capaz de aprender de forma iterativa baseada em tentativas e erros \cite{abdi1994}. De fato, o Perceptron obteve sucesso esmagador ao resolver funções lógicas lineares fundamentais, tais como as portas lógicas E (AND) e OU (OR).

\subsubsection{A Restrição da Separabilidade Linear}

A arquitetura do Perceptron clássico é limitada a uma única camada de pesos sinápticos projetando-se diretamente para a camada de saída. O Teorema de Convergência do Perceptron estabelece que a rede invariavelmente encontrará a solução que classifica perfeitamente um conjunto de dados, sob uma condição estrita: os dados devem ser linearmente separáveis \cite{braga2007}. Isto significa que, no espaço n-dimensional dos dados, deve existir pelo menos um hiperplano capaz de atuar como fronteira discriminante isolando completamente as diferentes classes.

Entretanto, as expectativas a respeito da capacidade do Perceptron foram abruptamente refreadas pela análise rigorosa de Minsky e Papert no final da década de 1960. Eles demonstraram matematicamente que uma rede neural desprovida de camadas ocultas intermediárias é inerentemente incapaz de mapear o problema lógico do Ou-Exclusivo (XOR) \cite{abdi2006}. 

No problema XOR, as classes positivas e negativas situam-se em vértices opostos de um quadrilátero no plano euclidiano, configurando uma distribuição cruzada de pontos que nenhuma linha reta isolada consegue separar. Segundo a Data Science Academy \cite{dsa2022}, a repercussão da constatação dessa limitação estrutural causou um forte arrefecimento das pesquisas em redes neurais nas décadas seguintes. A solução para esta limitação topológica apenas surgiu com o desenvolvimento e o domínio matemático sobre as camadas ocultas não-lineares.

\subsection{Perceptron Multicamadas (MLP) e Funções Lógicas}

A superação das limitações inerentes aos modelos associativos de camada única consolidou-se com a estruturação do Perceptron Multicamadas (\textit{Multilayer Perceptron} - MLP). Ao incorporar camadas ocultas de unidades computacionais entre as entradas e saídas, o MLP viabilizou a modelagem de fronteiras de decisão altamente irregulares e não-lineares, superando definitivamente o problema lógico do XOR e abrindo caminho para o advento do \textit{Deep Learning} \cite{rand2006}. Este modelo constitui a arquitetura avançada de redes neurais analisada de forma comparativa na metodologia deste trabalho.

\subsubsection{O Teorema da Aproximação Universal}

O funcionamento de um MLP fundamenta-se na capacidade transformacional das camadas ocultas. Rand e Wilensky \cite{rand2006} explicam que cada neurônio oculto age processando uma combinação ponderada das entradas e aplicando uma função de ativação estritamente não-linear (como sigmoide, tangente hiperbólica ou ReLU). Essa etapa não-linear é imprescindível; sem ela, múltiplas matrizes de pesos poderiam ser algebricamente colapsadas em uma única matriz linear equivalente, impedindo o ganho real de complexidade.

A potência das camadas ocultas é formalizada pelo Teorema da Aproximação Universal \cite{braga2007}, que postula que um MLP com apenas uma camada intermediária finita, dotada de ativação não-linear, possui capacidade matemática teórica para aproximar com precisão arbitrária qualquer função contínua em domínios compactos. Assim, o MLP não tenta traçar uma linha reta através do problema XOR; ele, em vez disso, deforma matematicamente o espaço topológico de entrada, remapeando os pontos no espaço latente da camada oculta, no qual os padrões finalmente tornam-se linearmente separáveis para o neurônio da camada de saída \cite{dsa2022}.

\subsubsection{O Algoritmo de Retropropagação (Backpropagation)}

A arquitetura multicamadas exigiu o desenvolvimento de um método capaz de treinar neurônios que estão escondidos no meio da rede, os quais não possuem acesso direto ao erro observado no vetor final de saída. A solução para esse desafio é o algoritmo de \textit{Backpropagation} (Retropropagação do Erro), responsável por disseminar as derivadas de erro pelo interior da topologia profunda \cite{silva2016}.

O Backpropagation opera baseando-se na Regra da Cadeia do cálculo multivariável diferencial. Conforme abordado pela Data Science Academy \cite{dsa2022}, durante a fase adiante (\textit{forward pass}), o sinal é alimentado da entrada para a saída, gerando o erro de predição. Em seguida, na fase retrospectiva (\textit{backward pass}), o erro final da camada de saída $L$ é multiplicado pela derivada da função de ativação dos neurônios de saída, gerando um vetor de gradiente local (frequentemente denotado como delta, $\delta^{(L)}$).

O passo essencial do algoritmo consiste na retroalimentação deste gradiente delta para a camada anterior $l$. O erro associado a cada neurônio oculto é calculado iterativamente através da soma dos deltas dos neurônios subsequentes aos quais ele se conecta, ponderada pelas forças sinápticas dessas conexões \cite{abdi1994}. Finalmente, todos os pesos da arquitetura são submetidos a atualizações simultâneas proporcionais ao produto entre a saída ativada do neurônio originador e o delta recebido no neurônio destino, regidos por uma taxa de aprendizado finita.

Esta elegante engenharia estatística transformou os Perceptrons Multicamadas nas ferramentas de inteligência artificial mais adaptáveis e poderosas do mundo moderno.

\section{Redes Neurais Aplicadas na Área da Saúde}

O campo de interseção entre a inteligência computacional e a ciência médica clínica tem crescido em proporções exponenciais. A complexidade do diagnóstico e do prognóstico de condições de saúde, especialmente doenças sistêmicas agudas, como a insuficiência cardíaca crônica, frequentemente escapa à intuição médica pura ou às árvores de decisão lineares tradicionais. Onde as diretrizes médicas clássicas encontram limites analíticos ao processar dezenas de covariáveis biomédicas simultaneamente, as Redes Neurais Artificiais provaram ser excelentes ferramentas descritivas e prescritivas \cite{silva2016}.

A literatura médica moderna, segundo as pesquisas documentadas pela Data Science Academy \cite{dsa2022}, apoia-se firmemente em arquiteturas preditivas probabilísticas, como a regressão logística processada por ADALINEs, para sistemas que necessitam de alta interpretabilidade médica. Para estes modelos, é possível quantificar isoladamente a correlação de cada biomarcador (ex: taxas de glicose, níveis de colesterol LDL) no peso sináptico que acarreta a probabilidade de evento coronário. 

Simultaneamente, patologias em que interações de características obscurecem padrões univariados (onde o nível de um hormônio só eleva o risco cardíaco na presença elevada de um segundo marcador) exigem o poder não-linear das camadas ocultas. Para tais modelagens, o Perceptron Multicamadas (MLP) consolidou-se como padrão ouro \cite{braga2007}, sendo amplamente testado em processamento de \textit{Electronic Health Records} (EHR), detectando associações ocultas entre histórico genético, pressórico e estilos de vida. 

Ao adotarem estas abordagens de modelagem, os sistemas de saúde não buscam substituir o diagnóstico do cardiologista humano, mas fornecer um sistema de triagem algorítmica autônoma e incansável, que detecta tendências probabilísticas com alta precocidade estatística \cite{rand2006}. A implementação dos modelos Logísticos Lineares e Multicamadas abordada neste Trabalho de Conclusão de Curso segue diretamente essa premissa teórica investigativa aplicada aos dados cardiológicos brutos, que será explorada através dos métodos analíticos da próxima seção.
"""

path = r"c:\Users\Lucas\Documents\TCC - Previsao Insuficiencia Cardiaca\TCC - Parte Escrita\Modelo_de_dissertação___PROFMAT_UFVJM__1_\capitulos\fundamentacao-teorica.tex"
with open(path, "w", encoding="utf-8") as f:
    f.write(latex_content)

print(f"File updated with length: {len(latex_content)}")
