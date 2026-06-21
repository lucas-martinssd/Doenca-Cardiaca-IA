# TCC — Previsão de Insuficiência Cardíaca
**Instituição:** UFVJM | **Template:** PROFMAT | **Aluno:** Lucas

---

## CONTEXTO DO PROJETO

### Modelos Implementados (3 configurações: C1=500ep/0.01, C2=1000ep/0.001, C3=3000ep/0.0001)
- **ADALINE Logística NumPy**: GD em lote, sigmoide → C1:85,87% C2:86,23% C3:86,96% (Banco 1)
- **ADALINE Logística Sklearn** (SGDClassifier, log_loss): → C1:85,87% C2:86,23% C3:86,23% (Banco 1)
- **MLP NumPy**: Xavier init, ReLU oculta, sigmoide saída → C1:82,97% C2:86,23% **C3:87,68% / 15 FN** (Banco 1, melhor clínico)
- **MLP Sklearn** (MLPClassifier, SGD): → C1:87,32% **C2:88,04%** C3:82,61% (Banco 1, maior acurácia em C2)

### Datasets
- **Banco 1 (Heart Disease):** Age, Sex, ChestPainType (ATA/NAP/ASY/TA), RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope → HeartDisease (0/1). 918 pacientes, divisão 70/30 estratificada.
- **Banco 2 (Insuficiência Cardíaca):** age, creatine_phosphokinase, ejection_fraction, platelets, serum_creatinine, serum_sodium, time → DEATH_EVENT

### Pré-processamento
- Binarização por faixas (idade: jovem/adulto/idoso)
- One-Hot Encoding para variáveis categóricas (ChestPainType → 4 colunas)
- Mapeamento binário (Sex: M=1/F=0; ExerciseAngina: Y=1/N=0)
- MeuStandardScaler implementado manualmente (não necessário pois dados já tratados)
- Split estratificado manual (meuTrainTestSplit) com randomState=42

### Referência paralela
- **HeartCheck IA** (outro grupo): usou XGBoost + FastAPI + Angular → 87,32%

---

## REGRAS GERAIS PARA O CLAUDE CODE

- **Idioma:** Português brasileiro em todo texto LaTeX
- **Template:** Seguir ESTRITAMENTE o modelo PROFMAT/UFVJM (modelo-de-tcc.tex)
- **ABNT:** Citações com \cite{}, figuras com \caption{} e \label{}, tabelas formatadas
- **Nunca reescrever** seção inteira sem ser solicitado — editar apenas o trecho pedido
- **Sempre ler o arquivo .tex existente** antes de qualquer edição
- **Economizar tokens:** ler só o necessário, editar trechos, não recarregar arquivos já lidos

---

## ESTRUTURA DO PROJETO (atualizar se mudar)

```
TCC - Previsao Insuficiencia Cardiaca/
├── CLAUDE.md                          ← este arquivo
├── .claude/commands/                  ← comandos customizados
│   ├── pesquisar.md
│   ├── escrever.md
│   ├── revisar.md
│   └── tabela-resultados.md
├── TCC - Parte Escrita/
│   └── Modelo_de_dissertação__PROFMAT_UFVJM_1/
│       ├── modelo-de-tcc.tex          ← arquivo principal
│       └── capitulos/
│           ├── fundamentacao-teorica.tex
│           ├── metodologia.tex
│           ├── resultados.tex
│           └── conclusao.tex
├── codigo/                            ← código Python
│   ├── AdalineLogistica.py
│   ├── Multicamadas.py
│   └── generate_latex.py
└── referencias/                       ← PDFs e artigos
    └── 11-RedesNeurais-cppA2.pdf
```

---

## COMANDOS DISPONÍVEIS

| Comando | Quando usar |
|---|---|
| `/pesquisar` | Antes de escrever qualquer seção — analisa código e referências |
| `/escrever` | Escrever ou expandir seções no .tex |
| `/revisar` | Revisar trecho específico sem reescrever |
| `/tabela-resultados` | Gerar tabela LaTeX comparativa dos modelos |

---

## DADOS DE RESULTADOS (verificados em 20/05/2026 — execução reprodutível com seed=42)

### Banco 1 (276 pacientes no teste: 153 doentes / 123 saudáveis)
| Modelo | C1 (500ep/0,01) | C2 (1000ep/0,001) | C3 (3000ep/0,0001) | FN mín. |
|---|---|---|---|---|
| ADALINE NumPy | 85,87% / 28FN | 86,23% / 21FN | 86,96% / 20FN | 20 (C3) |
| ADALINE Sklearn | 85,87% / 21FN | 86,23% / 21FN | 86,23% / 21FN | 21 |
| MLP NumPy | 82,97% / 27FN | 86,23% / 16FN | **87,68% / 15FN** | **15 (C3)** |
| MLP Sklearn | 87,32% / 18FN | **88,04% / 20FN** | 82,61% / 35FN | 18 (C1) |

### Banco 2 (90 pacientes no teste: 29 óbitos / 61 sobreviventes)
| Modelo | C1 (500ep/0,01) | C2 (1000ep/0,001) | C3 (3000ep/0,0001) |
|---|---|---|---|
| ADALINE NumPy | 68,89% / 17FN | 72,22% / 17FN | 75,56% / 16FN |
| ADALINE Sklearn | 68,89% / 17FN | 68,89% / 17FN | 68,89% / 17FN |
| MLP NumPy | **78,89% / 9FN** | 75,56% / 14FN | 71,11% / 23FN |
| MLP Sklearn | 73,33% / 18FN | 68,89% / 17FN | 63,33% / **8FN** |

### Observação clínica importante
- **Banco 1**: MLP NumPy C3 = melhor clinicamente (15 FN, sensibilidade 90,20%). MLP Sklearn C2 = maior acurácia (88,04%).
- **Banco 2**: MLP NumPy C1 = melhor equilíbrio (78,89%, 9 FN). MLPs sofrem overfitting com mais épocas neste banco menor.

## CONFIGURAÇÕES DE SESSÃO
- Ao final de cada resposta, exiba: tokens usados, 
  tokens restantes estimados e custo acumulado da sessão