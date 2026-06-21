import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

COR_SB  = '#0064B4'
COR_CB  = '#FFA51E'
COR_SET = '#B03A2E'
COR_GRID = '#DDDDDD'

configs = ['C1', 'C2', 'C3']
x = np.arange(len(configs))
w = 0.35

adaline_sb = [85.87, 86.23, 86.96]
adaline_cb = [85.87, 86.23, 86.23]
mlp_sb     = [82.97, 86.23, 87.68]
mlp_cb     = [87.32, 88.04, 82.61]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), facecolor='white')

def build_panel(ax, sb, cb, titulo, subtitulo, arrow=False):
    bars_sb = ax.bar(x - w/2, sb, w, color=COR_SB, label='Sem Biblioteca', zorder=3)
    bars_cb = ax.bar(x + w/2, cb, w, color=COR_CB, label='Com Biblioteca', zorder=3)

    for bar, val in zip(bars_sb, sb):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.12,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=10)
    for bar, val in zip(bars_cb, cb):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.12,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=10)

    ax.set_ylim(80, 92)
    ax.set_yticks(range(80, 93, 2))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0f}%'))
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.yaxis.grid(True, color=COR_GRID, linewidth=0.8, zorder=0)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(titulo, fontsize=12.5, fontweight='bold', pad=4)
    ax.text(0.5, 1.01, subtitulo, transform=ax.transAxes,
            ha='center', va='bottom', fontsize=10, color='gray', style='italic')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.14),
              ncol=2, frameon=False, fontsize=10)

    if arrow:
        # seta bidirecional entre as barras de C3 (índice 2)
        x_sb = x[2] - w/2
        x_cb = x[2] + w/2
        y_arrow = max(sb[2], cb[2]) + 1.5
        ax.annotate('', xy=(x_cb, y_arrow), xytext=(x_sb, y_arrow),
                    arrowprops=dict(arrowstyle='<->', color=COR_SET, lw=1.8))
        ax.text((x_sb + x_cb)/2, y_arrow + 0.25, 'maior divergência',
                ha='center', va='bottom', fontsize=9.5, color=COR_SET)

build_panel(ax1, adaline_sb, adaline_cb,
            'ADALINE — Sem Biblioteca vs Com Biblioteca',
            'Resultados praticamente idênticos')

build_panel(ax2, mlp_sb, mlp_cb,
            'MLP — Sem Biblioteca vs Com Biblioteca',
            'Divergência maior em C3',
            arrow=True)

plt.tight_layout(pad=2.0)
plt.savefig('grafico_validacao.png', dpi=180, bbox_inches='tight',
            facecolor='white')
print("Imagem salva: grafico_validacao.png")
