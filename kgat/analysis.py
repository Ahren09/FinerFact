import glob
import os
import os.path as osp
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

attr_names = ['pre', 'rec', 'f1', 'acc', 'auc']


def collect_best_results(metric="auc", dir=None, name=None):
    root_dir = f"C:\\Users\\yiqia\\Shared\\Sensitivity\\{dir}"
    # root_dir = "C:\\Users\\yiqia\\Shared\\8-23\\P_08-23_lr1e-05"

    results, kernel_li = [], []
    file_li = glob.glob(osp.join(root_dir, "*", "Eval_*.tsv"))
    criterion = re.compile(".*K[0-9]*.tsv$")
    file_li = list(filter(criterion.match, file_li))

    for file in file_li:
        kernel = int(file.split('_K')[2].split('.tsv')[0])
        kernel_li += [kernel]
    assert len(file_li) == len(kernel_li)
    # Should not have any duplicates
    assert len(kernel_li) == len(set(kernel_li))

    # ------------------------------------------
    # Below are new implementations based on LDA
    # ------------------------------------------

    kernel_li = []

    attr_data = [[] for _ in range(len(attr_names))]
    # data = [[]]*5 # This one sucks

    results_df_d = {}

    trend_d = dict(zip(attr_names, attr_data))

    for file in file_li:
        kernel = int(file.split('_K')[2].split('.tsv')[0])
        kernel_li += [kernel]
    print(f"Kernel sizes: {kernel_li}")

    kernel_li, file_li = zip(*sorted(zip(kernel_li, file_li), key=lambda x: x[0]))
    kernel_li, file_li = list(kernel_li), list(file_li)

    for file, kernel in zip(file_li, kernel_li):
        df = pd.read_csv(file, sep='\t')

        max_f1_index = df[metric].argmax()
        best_row = df.iloc[max_f1_index][attr_names]
        best_row = pd.DataFrame(best_row).T
        best_row['kernel'] = kernel
        results += [best_row]

        for attr_name in attr_names:
            trend_d[attr_name] += [pd.DataFrame(df[attr_name]).T]

    best_results_df = pd.concat(results)
    # Use the best epoch number as a new attribute
    best_results_df = best_results_df.rename_axis('best_epoch').reset_index().set_index("kernel")

    for attr_name in attr_names:
        results_df_d[attr_name] = pd.concat(trend_d[attr_name]).reset_index()
        results_df_d[attr_name].index = kernel_li
        results_df_d[attr_name].rename_axis("kernel", inplace=True)

    results_merged_df = pd.concat(results_df_d.values())

    # TODO
    # g = sns.relplot(
    #     data=results_merged_df,
    #     col="index",
    #     # hue="kernel",
    #     kind="line",
    #     palette=sns.color_palette("rocket_r", n_colors=11)
    # )

    # ------------------------------------------
    # Visualize results of each metric
    # ------------------------------------------

    # g = sns.relplot(
    #     data=results_df_d[attr_name].drop(['index'], axis=1).transpose(),
    #     # size="kernel",
    #     kind="line",
    #     palette=sns.color_palette("rocket_r", n_colors=len(results_df_d[attr_name]))
    # )
    # g.set(ylim=(0.7, 1.1))
    # plt.show()

    # ------------------------------------------
    # Visualize results with the best F-1 score
    # ------------------------------------------

    # Plot the lines on two facets
    plot_data = best_results_df.drop(['best_epoch'], axis=1)
    if not osp.exists("vis"):
        os.mkdir("vis")
    plot_data.to_csv(osp.join("vis", f"{name}.tsv"), sep='\t')


    return plot_data


def sensitivity_analysis():
    plot_data_P = collect_best_results(metric="f1", dir="P_lr5e-05",name="Politifact")
    plot_data_G = collect_best_results(metric="auc", dir="G_lr5e-05",name="Gossipcop")

    fig, ax = plt.subplots(1, 2, sharey=True)
    # fig, ax = plt.subplots(1, 2)
    ax[0] = sns.relplot(
        data=plot_data_P,
        kind="line",
        aspect=1.5,
        palette=sns.color_palette("rocket_r", n_colors=len(attr_names))
    )
    ax[0].set(ylim=(0.5, 1.0))
    plt.title("Politifact")
    plt.xticks(np.arange(0, 22, 2))

    ax[1] = sns.relplot(
        data=plot_data_G,
        kind="line",
        aspect=1.5,
        palette=sns.color_palette("rocket_r", n_colors=len(attr_names))
    )

    plt.title("Gossipcop")
    plt.xticks(np.arange(0, 22, 2))
    # g.xaxis.set_major_locator(ticker.MultipleLocator(5))


    plt.show()

    pd.options.display.float_format = '{:,.4f}'.format

    sns.relplot(
        data=best_results_df,
        x="kernel",
        hue="coherence", size="choice", col="align",
        kind="line", size_order=kernel_li, palette=palette,
        height=5, aspect=.75, facet_kws=dict(sharex=False),
    )

    plt.show()

    print()


def analyze_results(labs_tmp, preds_tmp, counters, filenames, epoch, args):
    preds_tmp = np.array(preds_tmp)
    labs_tmp = np.array(labs_tmp)
    tp = np.where((preds_tmp == labs_tmp) & (preds_tmp == 1))[0]
    tn = np.where((preds_tmp == labs_tmp) & (preds_tmp == 0))[0]
    fp = np.where((preds_tmp != labs_tmp) & (preds_tmp == 1))[0]
    fn = np.where((preds_tmp != labs_tmp) & (preds_tmp == 0))[0]

    filenames = np.array(filenames)
    for ids, counter in zip([tp, tn, fp, fn], counters):
        counter.update(filenames[ids])
    return counters


if __name__ == "__main__":
    sensitivity_analysis()
