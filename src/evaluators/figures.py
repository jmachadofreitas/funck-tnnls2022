""" Backbone for some figures """
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from .reports import BaselineDB, AlphaBetaTradeoffDB, BetaTradeoffDB, SSLTradeoffDB

# Choose the interpolation: how many intermediate images you want + 2 (for the start and end image)
n_interpolation = 9


def confusion_matrix(confusion_mat):
    f, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(confusion_mat, annot=True, linewidths=0.01, cmap="Oranges", linecolor="gray", fmt='.1f', ax=ax)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()


def interpolate_class(model, n_view=5):
    """ InfoGAN """
    model.eval()
    interpolation_noise = get_noise(n_view, z_dim, device=device).repeat(n_interpolation, 1)
    first_label = get_noise(1, c_dim).repeat(n_view, 1)[None, :]

    second_label = first_label.clone()
    first_label[:, :, 0] = -2
    second_label[:, :, 0] = 2

    # Calculate the interpolation vector between the two labels
    percent_second_label = torch.linspace(0, 1, n_interpolation)[:, None, None]
    interpolation_labels = first_label * (1 - percent_second_label) + second_label * percent_second_label
    interpolation_labels = interpolation_labels.view(-1, c_dim)

    # Combine the noise and the labels
    noise_and_labels = combine_vectors(interpolation_noise, interpolation_labels.to(device))
    fake = model(noise_and_labels)
    show_tensor_images(fake, num_images=n_interpolation * n_view, nrow=n_view, show=False)

    plt.figure(figsize=(8, 8))
    interpolate_class()
    _ = plt.axis('off')


def interpolate_class():
    """ InfoGAN """
    interpolation_noise = get_noise(1, z_dim, device=device).repeat(n_interpolation * n_interpolation, 1)
    first_label = get_noise(1, c_dim).repeat(n_interpolation * n_interpolation, 1)

    # Calculate the interpolation vector between the two labels
    first_label = torch.linspace(-2, 2, n_interpolation).repeat(n_interpolation)
    second_label = torch.linspace(-2, 2, n_interpolation).repeat_interleave(n_interpolation)
    interpolation_labels = torch.stack([first_label, second_label], dim=1)

    # Combine the noise and the labels
    noise_and_labels = combine_vectors(interpolation_noise, interpolation_labels.to(device))
    fake = gen(noise_and_labels)
    show_tensor_images(fake, num_images=n_interpolation * n_interpolation, nrow=n_interpolation, show=False)

    plt.figure(figsize=(8, 8))
    interpolate_class()
    _ = plt.axis('off')


class BaseFigure(object):

    def __init__(
            self,
            results_filepaths,
            baseline_filepaths=None,
            alpha=0.7,
            aspect=1.1,
            height=6
    ):
        self.results_filepaths = results_filepaths
        self.baseline_filepaths = baseline_filepaths
        self.palette = "icefire"  # "icefire" "vlag" "flare", "crest_r"
        # self.palette = "Spectral"  # sns.diverging_palette(220, 20, as_cmap=True)
        self.aspect = aspect  # 1
        self.height = height  # 8.66
        self.alpha = alpha
        self.markers = {
            "CPF-SI": "o", "CPFSI": "o",
            "CPF": "D", "IBSI": "p", "IB-SI": "D",
            "CFB": "s"
        }
        self.edgecolor = "none"
        self.bw_palette = sns.color_palette([
            "#111111", "#555555", "#999999",
            # "#55a868","#c44e52","#c44e52"
        ])
        self.style_order = ["CPF-SI", "CPFSI", "CPF", "IBSI", "CFB"]
        self.edgecolor = "none"
        self.linestyle_dotted = (1, (1, 1))  # ":"
        self.linestyle_densily_dashed = (2, (2, 1))  # "--"

    def print_empty(self, *dbs):
        for idx, df in enumerate(dbs):
            if df.empty:
                print(idx, df)

    def get_tables(self):
        pass

    def _add_metric_target(self, *dfs):
        for df in dfs:
            df["metric_target"] = df["metric"].str.cat(df["name"], sep="_")
        return dfs

    def _filter_seed(self, seed, *dfs):
        return [df.loc[df["seed"] == seed] for df in dfs]

    def _remove_stratified_metrics(self, *dfs):
        grouped_metrics = ["accuracies", "precisions", "recalls"]
        return [df.loc[~df["metric"].isin(grouped_metrics)].copy() for df in dfs]

    def prepare_tables(self):
        pass

    def get_coords(self, estimator=None):
        if estimator is None:
            x = ("y", self.prediction)
            y = ("0", self.reconstruction)
            s = ("s", self.prediction)
            hue = ("y", self.fairness)
        else:
            x = (estimator, "y", self.prediction)
            y = (estimator, "0", self.reconstruction)
            s = (estimator, "s", self.prediction)
            hue = (estimator, "y", self.fairness)
        return x, y, s, hue

    def get_labels(self, dataset=None):
        prediction = self.prediction.replace("_", " ").capitalize()
        reconstruction = self.reconstruction.replace("_", " ").upper()
        fairness = self.fairness.replace("_", " ").capitalize()

        if dataset is None:
            xlabel = f"{prediction}"
            ylabel = f"{reconstruction}"
            huelabel = f"{fairness}"
            return xlabel, ylabel, huelabel

        if dataset == "shifted_moons":
            xlabel = f"{prediction} (Moon)"
            ylabel = f"{reconstruction} (Coord)"
            slabel = f"{prediction} (Shift)"
            huelabel = f"{fairness} (Shift)"
        elif dataset == "german":
            xlabel = f"{prediction} (Credit Score)"
            ylabel = f"{reconstruction} (Duration)"
            slabel = f"{prediction} (Age)"
            huelabel = f"{fairness} (Age)"
        elif dataset == "compas":
            xlabel = f"{prediction} (Recidivism)"
            ylabel = f"{reconstruction} (# of Priors)"
            slabel = f"{prediction} (Ethnicity)"
            huelabel = f"{fairness} (Ethnicity)"
        elif dataset == "adult":
            xlabel = f"{prediction} (Income)"
            ylabel = f"{reconstruction} (Age)"
            slabel = f"{prediction} (Sex)"
            huelabel = f"{fairness} (Sex)"
        elif dataset == "adult":
            xlabel = f"{prediction} (Income)"
            ylabel = f"{reconstruction} (Age)"
            slabel = f"{prediction} (Sex)"
            huelabel = f"{fairness} (Sex)"
        elif dataset == "credit":
            xlabel = f"{prediction} (Credit Default)"
            ylabel = f"{reconstruction} (Credit Amount)"
            slabel = f"{prediction} (Sex)"
            huelabel = f"{fairness} (Sex)"
        elif dataset == "dutch":
            xlabel = f"{prediction} (Occupation)"
            ylabel = f"{reconstruction} (Age)"
            slabel = f"{prediction} (Sex)"
            huelabel = f"{fairness} (Sex)"
        else:
            xlabel = f"{prediction}"
            ylabel = f"{reconstruction}"
            slabel = f"{prediction}"
            huelabel = f"{fairness}"
        return xlabel, ylabel, slabel, huelabel

    def print_baselines(self, baseline):
        estimators = ["dummy", "lr", "rf"]
        for estimator in estimators:
            x, y, s, hue = self.get_coords(estimator)
            baseline_target = baseline[x].item()
            baseline_privacy = baseline[s].item()
            baseline_fairness = baseline[hue].item()
            print(f"Baseline ({estimator.upper()}) {self.prediction.capitalize()} (y): {baseline_target}")
            print(f"Baseline ({estimator.upper()}) {self.fairness.capitalize()} (y): {baseline_fairness}")
            print(f"Baseline ({estimator.upper()}) {self.prediction.capitalize()} (s): {baseline_privacy}")

    def figure(self):
        pass

    def _pivot_tables_args(self, base_index, aggregation):
        if "estimator" in aggregation:
            raise ValueError

        if aggregation is None or aggregation == "none":
            # No aggregation
            index = base_index + ["experiment_id", "replicate_id", "measure_id"]
        elif aggregation == "measure":
            # Aggregate by measure
            index = base_index + ["experiment_id", "replicate_id"]
        elif aggregation == "replicate":
            # Aggregate by replicate
            index = base_index + ["experiment_id", "measure_id"]
        elif aggregation == "all" or aggregation == "both":
            index = base_index
        else:
            index = aggregation if isinstance(aggregation, list) else [aggregation]

        values = "value"  # "mean", "median"
        columns = list()
        if "estimator" not in base_index:
            columns += ["estimator"]
        columns += ["name"]
        if "metric" not in base_index:
            columns += ["metric"]
        # print(columns)

        aggfunc = np.median  # np.mean

        index += ["model", "alpha", "beta"]
        index = list(set(index))  # Remove duplicates

        # print(values, index, columns, aggfunc)
        return values, index, columns, aggfunc


class AlphaBetaTradeoffFigure(BaseFigure):

    def __init__(
            self,
            results_filepaths,
            baseline_filepaths,
            prediction="accuracy",
            fairness="discrimination",
            reconstruction="mae",
            alpha=.7,
            aspect=1.1,
            height=6,
    ):
        """
            Args:
                estimator: "dummy", "lr", "rf"
                prediction: "accuracy", "precision", "recall", "auc"
                reconstruction: "mse", "rmse", "mae",
                fairness: "discrimination", "equalized_odds", "error_gap"
        """
        super().__init__(
            results_filepaths,
            baseline_filepaths=baseline_filepaths,
            alpha=alpha,
            aspect=aspect,
            height=height,
        )
        self.prediction = prediction
        self.fairness = fairness
        self.reconstruction = reconstruction

    def get_tables(
            self,
            latent_dim=None,
            aggregate=False
    ):
        baseline = list()
        for baseline_filepath in self.baseline_filepaths:
            b = BaselineDB(baseline_filepath)
            baseline_ = b.get_tables()
            baseline.append(baseline_)
        baseline = pd.concat(baseline)

        cpf, cfb, ibsi, ours = list(), list(), list(), list()
        for results_filepath in self.results_filepaths:
            ab = AlphaBetaTradeoffDB(results_filepath)
            cpf_, cfb_, ibsi_, ours_ = ab.get_tables(latent_dim=latent_dim, aggregate=aggregate)
            cpf.append(cpf_)
            cfb.append(cfb_)
            ibsi.append(ibsi_)
            ours.append(ours_)
        cpf = pd.concat(cpf)
        cfb = pd.concat(cfb)
        ibsi = pd.concat(ibsi)
        ours = pd.concat(ours)
        return baseline, cpf, cfb, ibsi, ours

    def prepare_tables(
            self,
            base_index=None,
            aggregation="none",
            seed=None,
            latent_dim=None,
    ):

        if base_index is None:  # Features used in FaceGrid
            base_index = list()

        baseline, cpf, cfb, ibsi, ours = self.get_tables(
            latent_dim=latent_dim,
            aggregate=False
        )

        if seed:
            cpf, cfb, ibsi, ours = self._filter_seed(seed, cpf, cfb, ibsi, ours)

        # Remove stratified metrics
        baseline, cpf, cfb, ibsi, ours = self._remove_stratified_metrics(baseline, cpf, cfb, ibsi, ours)

        # Pivot! Pivot!
        values, index, columns, aggfunc = self._pivot_tables_args(base_index, aggregation)
        if not baseline.empty:
            baseline = baseline.pivot_table(
                index=["dataset", "model"],
                columns=["estimator", "name", "metric"],
                values=values,
                aggfunc=aggfunc
            )

        if not cpf.empty:
            cpf = cpf.pivot_table(
                index=index,
                columns=columns,
                values=values,
                aggfunc=aggfunc
            )

        if not cfb.empty:
            cfb = cfb.pivot_table(
                index=index,
                columns=columns,
                values=values,
                aggfunc=aggfunc
            )

        if not ibsi.empty:
            # >>>>>>>>>>>>> TEMP <<<<<<<<<<<<<<<<<<<<<<<<<
            # ibsi = ibsi[ibsi["alpha"] == 1]  # !!!!!!
            ibsi = ibsi.pivot_table(
                index=index,
                columns=columns,
                values=values,
                aggfunc=aggfunc
            )

        if not ours.empty:
            ours = ours.pivot_table(
                index=index,
                columns=columns,
                values=values,
                aggfunc=aggfunc
            )

        return baseline, cpf, cfb, ibsi, ours

    def get_limit(self, data, coord, estimators=None, baseline=None):
        # data.sort_values(coord, ascending=False, inplace=True)
        if baseline is not None:
            keys = [(estimator, *coord) for estimator in estimators]
            lim = (
                min(baseline[keys].min(axis=1).item(), data[coord].min()),
                max(baseline[keys].max(axis=1).item(), data[coord].max())
            )
        else:
            lim = data[coord].min(), data[coord].max()
        diff = 0.05 * (lim[1] - lim[0])
        return max(0, lim[0] - diff), lim[1] + diff

    def get_limits(self, data, coord, estimators=None, baseline=None, dummy=False):
        limits = list()
        for estimator in estimators:
            tmp = data[data["estimator"] == estimator]
            if baseline is not None:
                if dummy:
                    key = (estimator, *coord)
                    dkey = ("dummy", *coord)
                    lim = (
                        max(0, min(baseline[key].min().item(), baseline[dkey].min().item(), tmp[coord].min())),
                        min(1, max(baseline[key].max().item(), baseline[dkey].max().item(), tmp[coord].max()))
                    )
                else:
                    key = (estimator, *coord)
                    lim = (
                        max(0, min(baseline[key].min().item(), tmp[coord].min())),
                        min(1, max(baseline[key].max().item(), tmp[coord].max()))
                    )
            else:
                lim = max(0, tmp[coord].min()), tmp[coord].max()
            diff = 0.1 * (lim[1] - lim[0])
            limit = max(-0.05, lim[0] - diff), lim[1] + diff
            # limit = max(-0.01, lim[0] - diff), lim[1] + diff
            limits.append(limit)
        return limits

    def get_predictor_limit(self, data, coord, predictors=None, baseline=None):
        # data.sort_values(coord, ascending=False, inplace=True)
        if baseline is not None:
            keys = [("lr", *coord) for _ in predictors]
            lim = (
                min(baseline[keys].min(axis=1).item(), data[coord].min()),
                max(baseline[keys].max(axis=1).item(), data[coord].max())
            )
        else:
            lim = data[coord].min(), data[coord].max()
        diff = 0.05 * (lim[1] - lim[0])
        return max(0, lim[0] - diff), min(1, lim[1] + diff)

    def get_predictor_limits(self, data, coord, predictors=None, baseline=None):
        limits = list()
        for predictor in predictors:
            tmp = data[data["estimator"] == predictor]
            if baseline is not None:
                key = ("lr", *coord)
                lim = (
                    max(0, min(baseline[key].min().item(), tmp[coord].min())),
                    max(0, baseline[key].max().item(), tmp[coord].max())
                )
            else:
                lim = max(0, tmp[coord].min()), tmp[coord].max()
            diff = 0.1 * (lim[1] - lim[0])
            limit = max(-0.05, lim[0] - diff), lim[1] + diff
            limits.append(limit)
        return limits

    def scatter(
            self,
            data,
            x,
            y,
            hue=None,
            size=None,
            s=150,
            row=None,
            col=None,
            row_order=None,
            col_order=None,
            xlabel=None,
            ylabel=None,
            huelabel=None,
            huelim=None,
            sharex=True,
            sharey=True,
            ascending=False,
            edgecolor=None,
            colorbar=True,
            height=None,
            aspect=None,
            margin_titles=True,
            overall_label=True
    ):
        height = height if height else self.height
        aspect = aspect if aspect else self.aspect
        edgecolor = edgecolor if edgecolor else self.edgecolor

        data.sort_values(hue, ascending=ascending, inplace=True)

        if huelim is None:
            huelim = data[hue].min(), data[hue].max()

        if colorbar:
            norm = plt.Normalize(*huelim)
            sm = plt.cm.ScalarMappable(cmap=self.palette, norm=norm)
            sm.set_array([])

        # FacetGrid
        g = sns.FacetGrid(
            data,
            row=row,
            col=col,
            row_order=row_order,
            col_order=col_order,
            margin_titles=margin_titles,
            height=height,
            aspect=aspect,
            legend_out=False,
            sharex=sharex,
            sharey=sharey
        )
        g.map_dataframe(
            sns.scatterplot,
            x=x,
            y=y,
            hue=hue,
            hue_norm=huelim,
            style="model",
            size=size,
            markers=self.markers,
            s=s,
            alpha=self.alpha,
            palette=self.palette,
            edgecolor=edgecolor,
        )

        # Colorbar
        if colorbar:
            dims = [1, 0.05, 0.015, 0.9]
            # if col == "Dataset" or col_order == ["IBSI"]:
            #     cax = g.figure.add_axes(dims)
            #     cbar = g.figure.colorbar(sm, cax=cax)
            #     cbar.ax.get_yaxis().labelpad = 50
            # elif row_order is not None and len(row_order) > 2:
            #     cax = g.figure.add_axes(dims)
            #     cbar = g.figure.colorbar(sm, cax=cax)
            #     cbar.ax.get_yaxis().labelpad = 40
            # else:
            if col_order is not None:
                if len(col_order) == 3:
                    dims = [1, 0.05, 0.015, 0.9]
                elif len(col_order) == 4:
                    dims = [1, 0.05, 0.012, 0.9]
            cax = g.figure.add_axes(dims)
            cbar = g.figure.colorbar(sm, cax=cax)
            cbar.ax.get_yaxis().labelpad = 40
        else:
            cbar = None

        # Labels
        if overall_label:
            g.set_axis_labels("", "")

            if ylabel:
                g.fig.text(  # overall ylabel
                    x=0,
                    y=0.5,  # .5
                    verticalalignment="center",  # make sure it's aligned at center vertically
                    s=ylabel,  # this is the text in the ylabel
                    # size=16,  # customize the fontsize if you will
                    rotation=90  # vertical text
                )

            if xlabel:
                g.fig.text(  # overall xlabel
                    x=0.5,  # .5
                    y=0,
                    horizontalalignment="center",  # make sure it's aligned at center horizontally
                    s=xlabel,  # this is the text in the xlabel
                    # size=16
                )
        else:
            g.set_axis_labels(xlabel, ylabel)

        if colorbar:
            cbar.ax.set_ylabel(huelabel, rotation=270)

        # Clean-up
        # g.despine(trim=True)

        return g, cbar

    def add_baselines(self, data, x, hue, g, cbar=None):
        axes0, axes1 = g.axes[0], g.axes[1]

        linestyle = self.linestyle_dotted
        xval = data[("lr", *x)].item() if len(x) == 2 else data[x].item()
        hueval = data[("lr", *hue)].item() if len(hue) == 2 else data[hue].item()
        for ax in axes0:
            ax.axvline(xval, c="gray", linestyle=linestyle, lw=3, zorder=1)
        cbar.ax.axhline(hueval, c="w", linestyle=linestyle, lw=3)

        linestyle = self.linestyle_densily_dashed
        xval = data[("rf", *x)].item() if len(x) == 2 else data[x].item()
        hueval = data[("rf", *hue)].item() if len(hue) == 2 else data[hue].item()
        for ax in axes1:
            ax.axvline(xval, c="gray", linestyle="--", lw=3, zorder=1)
        cbar.ax.axhline(hueval, c="w", linestyle=linestyle, lw=3)

        return g, cbar

    def add_baselines_v2(self, data, x, y, g, cbar=None, hue=None, dummy=False):
        axes0, axes1 = g.axes[0], g.axes[1]

        linestyle = self.linestyle_dotted if dummy else self.linestyle_densily_dashed
        estimator = "lr"
        xval = data[(estimator, *x)].item() if len(x) == 2 else data[x].item()
        yval = data[(estimator, *y)].item() if len(y) == 2 else data[y].item()
        for ax in axes0:
            ax.axvline(xval, c="gray", linestyle=linestyle, lw=3, zorder=1)
            ax.axhline(yval, c="gray", linestyle=linestyle, lw=3, zorder=1)
        if hue:
            hueval = data[(estimator, *hue)].item() if len(hue) == 2 else data[hue].item()
            cbar.ax.axhline(hueval, c="w", linestyle=linestyle, lw=3)

        linestyle = self.linestyle_dotted
        estimator = "rf"
        xval = data[(estimator, *x)].item() if len(x) == 2 else data[x].item()
        yval = data[(estimator, *y)].item() if len(y) == 2 else data[y].item()
        for ax in axes1:
            ax.axvline(xval, c="gray", linestyle=linestyle, lw=3, zorder=1)
            ax.axhline(yval, c="gray", linestyle=linestyle, lw=3, zorder=1)
        if hue:
            hueval = data[(estimator, *hue)].item() if len(hue) == 2 else data[hue].item()
            cbar.ax.axhline(hueval, c="w", linestyle=linestyle, lw=3)

        if dummy:
            linestyle = self.linestyle_densily_dashed
            estimator = "dummy"
            xval = data[(estimator, *x)].item() if len(x) == 2 else data[x].item()
            yval = data[(estimator, *y)].item() if len(y) == 2 else data[y].item()
            for ax in axes0:
                ax.axvline(xval, c="gray", linestyle=linestyle, lw=3, zorder=1)
                ax.axhline(yval, c="gray", linestyle=linestyle, lw=3, zorder=1)
            for ax in axes1:
                ax.axvline(xval, c="gray", linestyle=linestyle, lw=3, zorder=1)
                ax.axhline(yval, c="gray", linestyle=linestyle, lw=3, zorder=1)
            if hue:
                hueval = data[(estimator, *hue)].item() if len(hue) == 2 else data[hue].item()
                cbar.ax.axhline(hueval, c="w", linestyle=linestyle, lw=3)

    def add_rf_baselines(self, data, x, y, g):
        linestyle = self.linestyle_densily_dashed
        xval = data[("lr", *x)].item() if len(x) == 2 else data[x].item()
        yval = data[("lr", *y)].item() if len(y) == 2 else data[y].item()
        for axeslist in g.axes:
            for ax in axeslist:
                ax.axvline(xval, c="gray", linestyle=linestyle, lw=3, zorder=1)
                ax.axhline(yval, c="gray", linestyle=linestyle, lw=3, zorder=1)

    def add_estimator_baselines(self, data, x, y, g, cbar=None, estimator="lr", hue=None):
        linestyle = self.linestyle_densily_dashed
        xval = data[(estimator, *x)].item() if len(x) == 2 else data[x].item()
        yval = data[(estimator, *y)].item() if len(y) == 2 else data[y].item()
        for axeslist in g.axes:
            for ax in axeslist:
                ax.axvline(xval, c="gray", linestyle=linestyle, lw=3, zorder=1)
                ax.axhline(yval, c="gray", linestyle=linestyle, lw=3, zorder=1)
        if hue:
            hueval = data[(estimator, *hue)].item() if len(hue) == 2 else data[hue].item()
            cbar.ax.axhline(hueval, c="w", linestyle=linestyle, lw=3)


    def lowess(
            self,
            data,
            x,
            y,
            row=None,
            col=None,
            row_order=None,
            col_order=None,
            xlabel=None,
            ylabel=None,
            sharex=True,
            baseline=None,
    ):

        # Workaround
        data = pd.DataFrame(dict(
            x=data[x],
            y=data[y],
            beta=data["beta"],
            Estimator=data[row],
            Dataset=data[col],
            Model=data["model"]
        ))

        facet_kws = dict(legend_out=False, sharex=sharex, margin_titles=True)
        g = sns.lmplot(
            data=data,
            x="x",
            y="y",
            hue="Model",
            row="Estimator",
            col="Dataset",
            lowess=True,
            legend="brief",
            height=self.height,
            aspect=self.aspect,
            facet_kws=facet_kws
        )

        if x == "beta":
            g.set(xscale="log")

        # Labels
        g.set_axis_labels("", "")

        # overall ylabel
        g.fig.text(
            x=0, y=0.5,
            verticalalignment="center",  # make sure it's aligned at center vertically
            s=ylabel,  # this is the text in the ylabel
            # size=16,  # customize the fontsize if you will
            rotation=90
            )  # vertical text

        # overall xlabel
        g.fig.text(
            x=0.5, y=0,
            horizontalalignment="center",  # make sure it's aligned at center horizontally
            s=xlabel,  # this is the text in the xlabel
            # size=16
        )

        g.despine(trim=True)
        return g

    def add_lowess_baseline(self, baseline):
        """ TODO """
        if baseline is not None:
            if x != "beta":
                g.refline(x=baseline[x].item(), c="gray", linestyle="dashed", lw=3, zorder=1)
            g.refline(y=baseline[y].item(), c="gray", linestyle="dashed", lw=3, zorder=1)

    # def metrics(self, data, estimator, style, ylabel=None, baseline=None):
    #     # 2.2*10
    #     fig, axes = plt.subplots(1, 3, figsize=(1.8*8, 8), sharex=False, sharey=True)
    #     for idx, metric in enumerate(["discrimination", "equalized_odds", "error_gap"]):
    #         self.fairness = metric
    #         y, _, _, x = self.get_coords(estimator)
    #
    #         ax = axes[idx]
    #         # min_max = floor(10 * df[x].min()) / 10, ceil(10 * df[x].max()) / 10 + 1e-2
    #         g = sns.scatterplot(
    #             data=data,
    #             x=x,
    #             y=y,
    #             hue=style,
    #             style=style,
    #             # style_order=self.style_order,  # ["CPFSI", "CPF", "CFB"]
    #             markers=self.markers,
    #             s=150,
    #             # size="model",
    #             alpha=self.alpha,
    #             palette=self.bw_palette,
    #             # edgecolor="white",
    #             edgecolor=self.edgecolor,
    #             legend=False,
    #             ax=ax
    #         )
    #
    #         # Labels
    #         xlabel = x[-1].replace("_", " ").title()
    #         ax.set(xlabel=xlabel, ylabel=ylabel)
    #
    #         # # Reference Lines
    #         if baseline is not None:
    #             kwargs = dict(lw=3, zorder=1)
    #             # print(baseline[(estimator, *x[1:])].item(), baseline[(estimator, *y[1:])].item())
    #
    #             # g.axvline(baseline[("dummy", *x[1:])].item(), c="gray", linestyle="dotted", **kwargs)
    #             g.axvline(baseline[(estimator, *x[1:])].item(), c="gray", linestyle="dashed", **kwargs)
    #             # g.axhline(baseline[("dummy", *y[1:])].item(), c="gray", linestyle="dotted", **kwargs)
    #             g.axhline(baseline[(estimator, *y[1:])].item(), c="gray", linestyle="dashed", **kwargs)
    #
    #     # f = g.figure
    #     sns.despine(fig=fig, trim=False, offset=dict(bottom=20, left=20))
    #     fig.tight_layout()
    #     self.fairness = "discrimination"
    #     return fig


class BetaTradeoffFigure(BaseFigure):

    def __init__(
            self,
            results_filepaths,
            baseline_filepaths,
            prediction="accuracy",
            fairness="discrimination",
            # reconstruction="rmse"
            reconstruction="mae",
            aspect=1.1,
            height=6
    ):
        super().__init__(
            results_filepaths,
            baseline_filepaths=baseline_filepaths,
            aspect=aspect,
            height=height
        )
        self.prediction = prediction
        self.fairness = fairness
        self.reconstruction = reconstruction

    def get_tables(
            self,
            latent_dim=None,
            aggregate=False,
    ):
        baseline = list()

        for baseline_filepath in self.baseline_filepaths:
            b = BaselineDB(baseline_filepath)
            baseline_ = b.get_tables()
            baseline.append(baseline_)
        baseline = pd.concat(baseline)

        ibsi, cpf, ours = list(), list(), list()
        for results_filepath in self.results_filepaths:
            beta = BetaTradeoffDB(results_filepath)
            ibsi_, cpf_, ours_ = beta.get_tables(latent_dim=latent_dim, aggregate=aggregate)
            ibsi.append(ibsi_)
            cpf.append(cpf_)
            ours.append(ours_)
        ibsi = pd.concat(ibsi)
        cpf = pd.concat(cpf)
        ours = pd.concat(ours)
        return baseline, ibsi, cpf, ours

    def prepare_tables(
            self,
            base_index=None,
            aggregation="none",
            seed=None,
            latent_dim=None,
    ):

        if base_index is None:  # Features used in FaceGrid
            base_index = list()

        baseline, ibsi, cfb, ours = self.get_tables(
            latent_dim=latent_dim,
            aggregate=False
        )

        if seed:
            ibsi = ibsi.loc[ibsi["seed"] == seed]
            cfb = cfb.loc[cfb["seed"] == seed]
            ours = ours.loc[ours["seed"] == seed]

        # Remove stratified metrics
        baseline, ibsi, cfb, ours = self._remove_stratified_metrics(baseline, ibsi, cfb, ours)

        # Pivot! Pivot!
        values, index, columns, aggfunc = self._pivot_tables_args(base_index, aggregation)
        if not baseline.empty:
            baseline = baseline.pivot_table(
                index=["dataset", "model"],
                columns=["estimator", "name", "metric"],
                values=values,
                aggfunc=aggfunc
            )

        if not ibsi.empty:
            ibsi = ibsi.pivot_table(
                index=index,
                columns=columns,
                values=values,
                aggfunc=aggfunc
            )

        if not cfb.empty:
            cfb = cfb.pivot_table(
                index=index,
                columns=columns,
                values=values,
                aggfunc=aggfunc
            )

        if not ours.empty:
            ours = ours.pivot_table(
                index=index,
                columns=columns,
                values=values,
                aggfunc=aggfunc
            )

        return baseline, ibsi, cfb, ours

    def get_limit(self, data, coord, estimators=None, baseline=None):
        # data.sort_values(coord, ascending=False, inplace=True)
        if baseline is not None:
            keys = [(estimator, *coord) for estimator in estimators]
            lim = (
                min(baseline[keys].min(axis=1).item(), data[coord].min()),
                max(baseline[keys].max(axis=1).item(), data[coord].max())
            )
        else:
            lim = data[coord].min(), data[coord].max()
        diff = 0.05 * (lim[1] - lim[0])
        return max(0, lim[0] - diff), lim[1] + diff

    def get_limits(self, data, coord, estimators=None, baseline=None):
        limits = list()
        for estimator in estimators:
            tmp = data[data["estimator"] == estimator]
            if baseline is not None:
                key = (estimator, *coord)
                lim = (
                    max(0, min(baseline[key].min().item(), tmp[coord].min())),
                    max(0, baseline[key].max().item(), tmp[coord].max())
                )
            else:
                lim = max(0, tmp[coord].min()), tmp[coord].max()
            diff = 0.1 * (lim[1] - lim[0])
            limit = max(-0.001, lim[0] - diff), lim[1] + diff
            limits.append(limit)
        return limits

    def get_predictor_limit(self, data, coord, predictors=None, baseline=None):
        # data.sort_values(coord, ascending=False, inplace=True)
        if baseline is not None:
            keys = [("lr", *coord) for predictor in predictors]
            lim = (
                min(baseline[keys].min(axis=1).item(), data[coord].min()),
                max(baseline[keys].max(axis=1).item(), data[coord].max())
            )
        else:
            lim = data[coord].min(), data[coord].max()
        diff = 0.05 * (lim[1] - lim[0])
        return max(0, lim[0] - diff), lim[1] + diff

    def get_predictor_limits(self, data, coord, predictors=None, baseline=None):
        limits = list()
        for predictor in predictors:
            tmp = data[data["estimator"] == predictor]
            if baseline is not None:
                key = ("lr", *coord)
                lim = (
                    max(0, min(baseline[key].min().item(), tmp[coord].min())),
                    max(0, baseline[key].max().item(), tmp[coord].max())
                )
            else:
                lim = max(0, tmp[coord].min()), tmp[coord].max()
            diff = 0.1 * (lim[1] - lim[0])
            limit = max(-0.001, lim[0] - diff), lim[1] + diff
            limits.append(limit)
        return limits

    def scatter(
            self,
            data,
            x,
            y,
            hue,
            row=None,
            col=None,
            row_order=None,
            col_order=None,
            xlabel=None,
            ylabel=None,
            huelabel=None,
            huelim=None,
            sharex=True,
            sharey=True,
            ascending=True,
            trim=False,
    ):
        data.sort_values(hue, ascending=ascending, inplace=True)

        if huelim is None:
            huelim = data[hue].min(), data[hue].max()

        norm = plt.Normalize(*huelim)
        sm = plt.cm.ScalarMappable(cmap=self.palette, norm=norm)
        sm.set_array([])

        # FacetGrid
        g = sns.FacetGrid(
            data,
            row=row,
            col=col,
            row_order=row_order,
            col_order=col_order,
            margin_titles=True,
            height=self.height,
            aspect=self.aspect,
            legend_out=False,
            sharex=sharex,
            sharey=sharey
        )
        ax = g.map_dataframe(
            sns.scatterplot,
            x=x,
            y=y,
            hue=hue,
            hue_norm=huelim,
            style="model",
            markers=self.markers,
            s=150,
            alpha=self.alpha,
            palette=self.palette,
            edgecolor=self.edgecolor,
        )

        # Colorbar
        if col == "Dataset":
            cax = g.figure.add_axes([1, 0.05, 0.035, 0.9])
            cbar = g.figure.colorbar(sm, cax=cax)
            cbar.ax.get_yaxis().labelpad = 50
        else:
            cax = g.figure.add_axes([1, 0.1, 0.015, 0.85])
            cbar = g.figure.colorbar(sm, cax=cax)
            cbar.ax.get_yaxis().labelpad = 40

        # Labels
        g.set_axis_labels("", "")

        g.fig.text(  # overall ylabel
            x=0, y=0.5,
            verticalalignment="center",  # make sure it's aligned at center vertically
            s=ylabel,  # this is the text in the ylabel
            # size=16,  # customize the fontsize if you will
            rotation=90  # vertical text
        )

        g.fig.text(  # overall xlabel
            x=0.5, y=0,
            horizontalalignment="center",  # make sure it's aligned at center horizontally
            s=xlabel,  # this is the text in the xlabel
            # size=16
        )

        cbar.ax.set_ylabel(huelabel, rotation=270)

        # Clean-up
        if trim:
            g.despine(trim=True)

        return g, cbar

    def add_lr_baselines(self, data, x, y, g):
        linestyle = self.linestyle_dotted
        xval = data[("lr", *x)].item() if len(x) == 2 else data[x].item()
        yval = data[("lr", *y)].item() if len(y) == 2 else data[y].item()
        for axeslist in g.axes:
            for ax in axeslist:
                ax.axvline(xval, c="gray", linestyle=linestyle, lw=3, zorder=1)
                ax.axhline(yval, c="gray", linestyle=linestyle, lw=3, zorder=1)

    def lowess(
            self,
            data,
            x,
            y,
            row=None,
            col=None,
            row_order=None,
            col_order=None,
            xlabel=None,
            ylabel=None,
            sharex=True,
            baseline=None,
    ):
        # print(f"x={x}, y={y}")

        data = pd.DataFrame(dict(
            x=data[x],
            y=data[y],
            beta=data["beta"],
            Estimator=data[row],
            Dataset=data[col],
            Model=data["model"]
        ))

        facet_kws = dict(legend_out=False, sharex=sharex, margin_titles=True)
        g = sns.lmplot(
            data=data,
            x="x",
            y="y",
            hue="Model",
            row="Estimator",
            col="Dataset",
            lowess=True,
            legend="brief",
            height=self.height,
            aspect=self.aspect,
            facet_kws=facet_kws
        )

        if x == "beta":
            g.set(xscale="log")

        if baseline is not None:
            if x != "beta":
                g.refline(x=baseline[x].item(), c="gray", linestyle="dashed", lw=3, zorder=1)
            g.refline(y=baseline[y].item(), c="gray", linestyle="dashed", lw=3, zorder=1)

        # Labels
        g.set_axis_labels("", "")

        g.fig.text(  # overall ylabel
            x=0, y=0.5,
            verticalalignment="center",  # make sure it's aligned at center vertically
            s=ylabel,  # this is the text in the ylabel
            # size=16,  # customize the fontsize if you will
            rotation=90
            )  # vertical text


        g.fig.text(  # overall xlabel
            x=0.5, y=0,
            horizontalalignment="center",  # make sure it's aligned at center horizontally
            s=xlabel,  # this is the text in the xlabel
            # size=16
        )

        # Clean-up
        g.despine(trim=True)

        return g


class SSLTradeoffFigure(BaseFigure):

    def __init__(
            self,
            results_filepaths,
            baseline_filepaths,
            prediction="accuracy",
            fairness="discrimination",
            reconstruction="mae",
            aspect=1.1,
            height=6
    ):

        super().__init__(
            results_filepaths,
            baseline_filepaths=baseline_filepaths,
            aspect=aspect,
            height=height
        )
        self.prediction = prediction
        self.fairness = fairness
        self.reconstruction = reconstruction

    def get_tables(
            self,
            latent_dim=None,
            aggregate=False,
    ):
        baseline = list()
        for baseline_filepath in self.baseline_filepaths:
            b = BaselineDB(baseline_filepath)
            baseline_ = b.get_tables()
            baseline.append(baseline_)
        baseline = pd.concat(baseline)

        cpf, vfae = list(), list()
        ibsi, cpfsi = list(), list()
        semivfae, semiibsi, semicpfsi = list(), list(), list()
        for results_filepath in self.results_filepaths:
            ssl = SSLTradeoffDB(results_filepath)
            cpf_, vfae_, ibsi_, cpfsi_, semivfae_, semiibsi_, semicpfsi_ = ssl.get_tables(
                latent_dim=latent_dim,
                aggregate=aggregate
            )
            cpf.append(cpf_)
            vfae.append(vfae_)
            ibsi.append(ibsi_)
            cpfsi.append(cpfsi_)
            semivfae.append(semivfae_)
            semiibsi.append(semiibsi_)
            semicpfsi.append(semicpfsi_)

        cpf = pd.concat(cpf)
        vfae = pd.concat(vfae)
        ibsi = pd.concat(ibsi)
        cpfsi = pd.concat(cpfsi)
        semivfae = pd.concat(semivfae)
        semiibsi = pd.concat(semiibsi)
        semicpfsi = pd.concat(semicpfsi)

        return baseline, cpf, vfae, ibsi, cpfsi, semivfae, semiibsi, semicpfsi

    def prepare_tables(
            self,
            base_index=None,
            aggregation="none",
            seed=None,
            latent_dim=None,
    ):

        if base_index is None:  # Features used in FaceGrid
            base_index = list()

        baseline, vfae, cpf, ibsi, cpfsi, semivfae, semiibsi, semicpfsi = self.get_tables(
            latent_dim=latent_dim,
            aggregate=False
        )

        baseline, vfae, cpf, ibsi, cpfsi, semivfae, semiibsi, semicpfsi = self._add_metric_target(
            baseline, vfae, cpf, ibsi, cpfsi, semivfae, semiibsi, semicpfsi
        )

        print(
            "Original size: ",
            "len(Baseline) =", len(baseline),
            ", len(VFAE) =", len(vfae),
            ", len(CPF) =", len(cpf),
            ", len(IBSI) =", len(ibsi),
            ", len(CPFSI) =", len(cpfsi),
            ", len(SemiVFAE) =", len(semivfae),
            ", len(SemiIBSI) =", len(semiibsi),
            ", len(SemiCPFSI) =", len(semicpfsi)
        )

        if seed:
            vfae, cpf, ibsi, cpfsi, semivfae, semiibsi, semicpfsi = self._filter_seed(
                seed, vfae, cpf, ibsi, cpfsi, semivfae, semiibsi, semicpfsi
            )

        # Remove stratified metrics
        baseline, vfae, cpf, ibsi, cpfsi, semivfae, semiibsi, semicpfsi = self._remove_stratified_metrics(
                baseline, cpf, vfae, ibsi, cpfsi, semivfae, semiibsi, semicpfsi
            )

        values, index, columns, aggfunc = self._pivot_tables_args(base_index, aggregation)
        # print(f"values={values}, index={index}, columns={columns}")
        if not baseline.empty:
            baseline = baseline.pivot_table(
                index=["dataset", "model"],
                columns=["estimator", "name", "metric"],
                values=values,
                aggfunc=aggfunc
            )

        if not vfae.empty:
            vfae = vfae.pivot_table(
                index=index,
                columns=columns,
                values=values,
                aggfunc=aggfunc
            )

        if not cpf.empty:
            cpf = cpf.pivot_table(
                index=index,
                columns=columns,
                values=values,
                aggfunc=aggfunc
            )

        if not ibsi.empty:
            ibsi = ibsi.pivot_table(
                index=index,
                columns=columns,
                values=values,
                aggfunc=aggfunc
            )

        if not cpfsi.empty:
            cpfsi = cpfsi.pivot_table(
                index=index,
                columns=columns,
                values=values,
                aggfunc=aggfunc
            )

        if not semivfae.empty:
            print(semivfae.columns)
            semivfae = semivfae.pivot_table(
                index=index + ["skey"],
                columns=columns,
                values=values,
                aggfunc=aggfunc
            )

        if not semiibsi.empty:
            semiibsi = semiibsi.pivot_table(
                index=index + ["skey"],
                columns=columns,
                values=values,
                aggfunc=aggfunc
            )

        if not semicpfsi.empty:
            semicpfsi = semicpfsi.pivot_table(
                index=index + ["skey"],
                columns=columns,
                values=values,
                aggfunc=aggfunc
            )

        return baseline, vfae, cpf, ibsi, cpfsi, semivfae, semiibsi, semicpfsi

    def prepare_tables_2(
            self,
            base_index=None,
            aggregation="none",
            seed=None,
            latent_dim=None,
    ):

        if base_index is None:  # Features used in FaceGrid
            base_index = list()

        baseline, vfae, cpf, ibsi, cpfsi, semivfae, semiibsi, semicpfsi = self.get_tables(
            latent_dim=latent_dim,
            aggregate=False
        )

        baseline, vfae, cpf, ibsi, cpfsi, semivfae, semiibsi, semicpfsi = self._add_metric_target(
            baseline, vfae, cpf, ibsi, cpfsi, semivfae, semiibsi, semicpfsi
        )

        print(
            "Original size: ",
            "len(Baseline) =", len(baseline),
            ", len(VFAE) =", len(vfae),
            ", len(CPF) =", len(cpf),
            ", len(IBSI) =", len(ibsi),
            ", len(CPFSI) =", len(cpfsi),
            ", len(SemiVFAE) =", len(semivfae),
            ", len(SemiIBSI) =", len(semiibsi),
            ", len(SemiCPFSI) =", len(semicpfsi)
        )

        if seed:
            vfae, cpf, ibsi, cpfsi, semivfae, semiibsi, semicpfsi = self._filter_seed(
                seed, vfae, cpf, ibsi, cpfsi, semivfae, semiibsi, semicpfsi
            )

        # Remove stratified metrics
        baseline, vfae, cpf, ibsi, cpfsi, semivfae, semiibsi, semicpfsi = self._remove_stratified_metrics(
            baseline, cpf, vfae, ibsi, cpfsi, semivfae, semiibsi, semicpfsi
        )

        if "estimator" in aggregation:
            raise ValueError

        if aggregation is None or aggregation == "none":
            # No aggregation
            index = base_index + ["experiment_id", "replicate_id", "measure_id"]
        elif aggregation == "measure":
            # Aggregate by measure
            index = base_index + ["experiment_id", "replicate_id"]
        elif aggregation == "replicate":
            # Aggregate by replicate
            index = base_index + ["experiment_id", "measure_id"]
        elif aggregation == "all" or aggregation == "both":
            index = base_index
        else:
            index = aggregation if isinstance(aggregation, list) else [aggregation]

        values = "value"
        columns = list()
        if "estimator" not in base_index:
            columns += ["estimator"]
        # columns += ["metric_target"]

        aggfunc = np.median  # np.mean

        index += ["model", "alpha", "beta"]
        index = list(set(index))  # Remove duplicates        
        if not baseline.empty:
            baseline = baseline.pivot_table(
                index=["dataset", "model"],
                columns=["estimator", "metric_target"],
                values=values,
                aggfunc=aggfunc
            )

        if not vfae.empty:
            vfae = vfae.pivot_table(
                index=index,
                columns=columns,
                values=values,
                aggfunc=aggfunc
            )

        if not cpf.empty:
            cpf = cpf.pivot_table(
                index=index,
                columns=columns,
                values=values,
                aggfunc=aggfunc
            )

        if not ibsi.empty:
            ibsi = ibsi.pivot_table(
                index=index,
                columns=columns,
                values=values,
                aggfunc=aggfunc
            )

        if not cpfsi.empty:
            cpfsi = cpfsi.pivot_table(
                index=index,
                columns=columns,
                values=values,
                aggfunc=aggfunc
            )

        if not semivfae.empty:
            semivfae = semivfae.pivot_table(
                index=index,
                columns=columns,
                values=values,
                aggfunc=aggfunc
            )

        if not semiibsi.empty:
            semiibsi = semiibsi.pivot_table(
                index=index,
                columns=columns,
                values=values,
                aggfunc=aggfunc
            )

        if not semicpfsi.empty:
            semicpfsi = semicpfsi.pivot_table(
                index=index,
                columns=columns,
                values=values,
                aggfunc=aggfunc
            )

        return baseline, vfae, cpf, ibsi, cpfsi, semivfae, semiibsi, semicpfsi

    def lowess(
            self,
            data,
            x,
            y,
            row=None,
            col=None,
            row_order=None,
            col_order=None,
            xlabel=None,
            ylabel=None,
            sharex=True,
            sharey=True,
            height=None,
            aspect=None,
    ):
        height = height if height else self.height
        aspect = aspect if aspect else self.aspect

        # Workaround
        data = pd.DataFrame(dict(
            x=data[x],
            y=data[y],
            beta=data["beta"],
            betamodel=data["beta"].astype(str).str.cat(data["model"]),
            num_samples_per_class=data["num_samples_per_class"],
            Estimator=data[row],
            Metric=data[col],
            Model=data["model"]
        ))

        # plt.xscale("log", base=2)
        facet_kws = dict(legend_out=False, sharex=sharex, sharey=sharey, margin_titles=True)
        g = sns.lmplot(
            data=data,
            x="x",
            y="y",
            hue="Model",
            row="Estimator",
            col="Metric",
            row_order=row_order,
            lowess=True,
            legend="brief",
            height=height,
            aspect=aspect,
            facet_kws=facet_kws
        )

        if x == "beta" or x == "num_samples_per_class":
            [ax.set_xscale('log', base=2) for axlist in g.axes for ax in axlist]
            # g.set(xscale="log", base=2)
            # g.set_xscale('log', basex=2)

        # Labels
        g.set_axis_labels("", "")

        # overall ylabel
        g.fig.text(
            x=0, y=0.5,
            verticalalignment="center",  # make sure it's aligned at center vertically
            s=ylabel,  # this is the text in the ylabel
            # size=16,  # customize the fontsize if you will
            rotation=90
            )  # vertical text

        # overall xlabel
        g.fig.text(
            x=0.5, y=0,
            horizontalalignment="center",  # make sure it's aligned at center horizontally
            s=xlabel,  # this is the text in the xlabel
            # size=16
        )

        # g.despine(trim=True)
        return g

    def get_limits(self, data, rows, row_name, metrics, baseline=None):
        y = "y"
        limits = list()
        data = data[data[row_name].isin(rows)]
        for metric in metrics:
            tmp = data[data["metric"] == metric]
            if baseline is None:
                lim = max(0, tmp[y].dropna().min(skipna=True)), tmp[y].dropna().max(skipna=True)
            else:
                tmp2 = baseline[baseline["metric"] == metric]
                lim = (
                    max(0, min(tmp2[y].min().item(), tmp[y].min())),
                    max(0, tmp2[y].max().item(), tmp[y].max())
                )

            diff = 0.1 * (lim[1] - lim[0])
            limit = max(-0.05, lim[0] - diff), lim[1] + diff
            limits.append(limit)
        print(limits)
        return limits

    def boxplot(
            self,
            data,
            x,
            y,
            hue,
            row=None,
            col=None,
            row_order=None,
            col_order=None,
            hue_order=None,
            xlabel=None,
            ylabel=None,
            sharex=True,
            sharey=True,
            ascending=True,
            trim=False,
            legend_out=False,
            palette="muted"
    ):
        data.sort_values(hue, ascending=ascending, inplace=True)

        # FacetGrid
        g = sns.FacetGrid(
            data,
            row=row,
            col=col,
            row_order=row_order,
            col_order=col_order,
            margin_titles=True,
            height=self.height,
            aspect=self.aspect,
            legend_out=legend_out,
            sharex=sharex,
            sharey=sharey
        )
        g.map_dataframe(
            sns.boxplot,
            x=x,
            y=y,
            hue=hue,
            hue_order=hue_order,
            palette=palette
        )

        # Labels
        g.set_axis_labels("", "")

        g.fig.text(  # overall ylabel
            x=0, y=0.5,
            verticalalignment="center",  # make sure it's aligned at center vertically
            s=ylabel,  # this is the text in the ylabel
            # size=16,  # customize the fontsize if you will
            rotation=90  # vertical text
        )

        g.fig.text(  # overall xlabel
            x=0.5, y=0,
            horizontalalignment="center",  # make sure it's aligned at center horizontally
            s=xlabel,  # this is the text in the xlabel
            # size=16
        )

        # cbar.ax.set_ylabel(huelabel, rotation=270)

        # Clean-up
        if trim:
            g.despine(trim=True)

        return g

    def add_baselines(self, data, rows, cols, row_name, g, linestyle="--"):

        axes0 = g.axes[0]
        for idx, ax in enumerate(axes0):
            tmp = data[data[row_name] == rows[0]]
            yval = tmp[tmp["metric"] == cols[idx]]["y"].median()
            ax.axhline(yval, c="gray", linestyle=linestyle, lw=3, zorder=1)

        if len(g.axes) == 2:
            axes1 = g.axes[1]
            for idx, ax in enumerate(axes1):
                tmp = data[data[row_name] == rows[1]]
                yval = tmp[tmp["metric"] == cols[idx]]["y"].median()
                ax.axhline(yval, c="gray", linestyle=linestyle, lw=3, zorder=1)


        # def lowess(
    #         self,
    #         data,
    #         x,
    #         y,
    #         hue,
    #         row=None,
    #         col=None,
    #         row_order=None,
    #         col_order=None,
    #         xlabel=None,
    #         ylabel=None,
    #         huelabel=None,
    #         sharex=True
    # ):
    #
    #     if not data.empty:
    #         data.sort_values(hue, ascending=False, inplace=True)
    #         min_max = data[hue].min(), data[hue].max()
    #         norm = plt.Normalize(*min_max)
    #         sm = plt.cm.ScalarMappable(cmap=self.palette, norm=norm)
    #         sm.set_array([])
    #     else:
    #         raise ValueError("All Empty!")
    #
    #     # FacetGrid
    #     g = sns.FacetGrid(
    #         data,
    #         row=row,
    #         col=col,
    #         row_order=row_order,
    #         col_order=col_order,
    #         margin_titles=True,
    #         height=self.height,
    #         aspect=self.aspect,
    #         legend_out=False,
    #         sharex=True
    #     )
    #     ax = g.map_dataframe(
    #         sns.lineplot,
    #         x=x,
    #         y=y,
    #         hue=hue,
    #         hue_norm=min_max,
    #         style="model",
    #         markers=self.markers,
    #         s=150,
    #         alpha=self.alpha,  # .5
    #         palette=self.palette,
    #         edgecolor=self.edgecolor,
    #         # edgecolor="r"
    #     )
    #     g.set_titles(col_template="{col_name}")
    #
    #     # Colorbar
    #     if col == "Dataset":
    #         cax = g.figure.add_axes([1, 0.05, 0.035, 0.9])
    #         cbar = g.figure.colorbar(sm, cax=cax)
    #         cbar.ax.get_yaxis().labelpad = 50
    #     else:
    #         cax = g.figure.add_axes([1, 0.1, 0.015, 0.85])
    #         cbar = g.figure.colorbar(sm, cax=cax)
    #         cbar.ax.get_yaxis().labelpad = 40
    #
    #     # Labels
    #     ax.set(xlabel=xlabel, ylabel=ylabel)
    #     cbar.ax.set_ylabel(huelabel, rotation=270)
    #
    #     # Clean-up
    #     # g.despine(trim=True)
    #
    #     return g, cbar
