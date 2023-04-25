import numpy as np


def xtsum(df, variable):

    temp_df = (df.loc[:, variable]
               .to_frame()
               .assign(between=lambda x: (x.groupby(level=0)[variable]
                                          .transform(np.mean)))
               .assign(within=lambda x: (x[variable] - x.between +
                                         x[variable].mean())))

    stats = temp_df.describe()
    correct_between = (temp_df['between']
                       .droplevel(axis=0, level=1)
                       .reset_index()
                       .drop_duplicates()
                       .between)
    stats['between'] = correct_between.describe()

    return stats
