# /tabela-resultados — Gerador de Tabela LaTeX

Gere a tabela comparativa dos resultados do TCC diretamente no arquivo de resultados.tex.

## Dados a incluir (já conhecidos)

### Banco 1 — 1000 épocas
| Modelo | Acurácia | VP | VN | FP | FN |
|---|---|---|---|---|---|
| ADALINE sem biblioteca | 89,13% | 95 | 69 | 13 | 7 |
| ADALINE com biblioteca | 90,22% | 91 | 75 | 7 | 11 |
| MLP sem biblioteca | 89,13% | 95 | 69 | 13 | 7 |
| MLP com biblioteca | 85,33% | 90 | 67 | 15 | 12 |

### Banco 2 — 3000 épocas, taxa 0,001
| Modelo | Acurácia | VP | VN | FP | FN |
|---|---|---|---|---|---|
| ADALINE sem biblioteca | ~83% | 88 | 68 | 14 | 14 |
| MLP sem biblioteca | ~84% | 88 | 68 | 14 | 14 |
| ADALINE com biblioteca | ~88% | 91 | 75 | 7 | 11 |
| MLP com biblioteca | ~87% | 94 | 70 | 12 | 8 |

## Formato LaTeX a gerar

```latex
\begin{table}[H]
\centering
\caption{Comparativo de desempenho dos modelos — Banco 1 (1000 épocas)}
\label{tab:resultados-banco1}
\begin{tabular}{lcccccc}
\hline
\textbf{Modelo} & \textbf{Acurácia} & \textbf{VP} & \textbf{VN} & \textbf{FP} & \textbf{FN} \\
\hline
...
\hline
\end{tabular}
\fonte{Elaborado pelo autor.}
\end{table}
```

Gere uma tabela para cada banco de dados e insira no arquivo resultados.tex após a análise individual de cada modelo.