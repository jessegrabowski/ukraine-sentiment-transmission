import matplotlib.pyplot as plt

def config_matplotlib():
    config = {'figure.figsize':(14,4),
              'figure.dpi':144,
              'figure.facecolor':'w',
              'axes.spines.top':False,
              'axes.spines.bottom':False,
              'axes.spines.left':False,
              'axes.spines.right':False,
              'axes.grid':True,
              'grid.linestyle':'--',
              'grid.linewidth':0.5}

    plt.rcParams.update(config)


def prepare_gridspec_figure(n_cols, n_plots):
    remainder = n_plots % n_cols
    has_remainder = remainder > 0
    n_rows = n_plots // n_cols + 1

    gs = plt.GridSpec(2 * n_rows, 2 * n_cols)
    plot_locs = []

    for i in range(n_rows - int(has_remainder)):
        for j in range(n_cols):
            plot_locs.append((slice(i * 2, (i + 1) * 2), slice(j * 2, (j + 1) * 2)))

    if has_remainder:
        last_row = slice((n_rows - 1) * 2, n_rows * 2)
        left_pad = int(n_cols - remainder)
        for j in range(remainder):
            col_slice = slice(left_pad + j * 2, left_pad + (j + 1) * 2)
            plot_locs.append((last_row, col_slice))

    return gs, plot_locs

