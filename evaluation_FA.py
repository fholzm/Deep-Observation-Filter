import os
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
import toml
import pandas as pd
from scipy import stats


def angle_between_vectors(v1, v2):
    """
    Compute the angle in degrees between two 3D vectors.

    Parameters:
    v1 (array-like): First 3D vector.
    v2 (array-like): Second 3D vector.

    Returns:
    float: Angle in degrees between the two vectors.
    """
    v1 = np.array(v1)
    v2 = np.array(v2)
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cos_theta = dot_product / (norm_v1 * norm_v2)
    angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)
    return angle_deg


def plotFrequencyDomainErrorPerPosition(fn, faxis, fig_params, config, mode="both"):
    fig = plt.figure(figsize=(fig_params["width"], fig_params["height"]))

    if mode == "both":
        cc = cycler(color=fig_params["colors"]) * cycler(linestyle=["-", "--"])

    else:
        cc = cycler("color", fig_params["colors"]) * cycler(
            "linestyle", ["-", "--", "-."]
        )

    plt.rc("axes", prop_cycle=cc)

    l = []

    labels = []
    for i in range(len(config["valid"]["eval_radius"])):
        labels.append(
            f"{config['valid']['eval_radius'][i][0]*100} cm $\\leq r <$ {config['valid']['eval_radius'][i][1]*100} cm"
        )

    data = np.load(os.path.join(fn_path, fn), allow_pickle=True)
    ERR = 10 * np.log10(data["est_error_psd"])
    rmse = 10 * np.log10(data["valid_rmse_per_epoch_samplewise"])
    metadata = data["metadata"].item()

    ERR_classes = [[] for _ in range(len(config["valid"]["eval_radius"]))]
    rmse_classes = [[] for _ in range(len(config["valid"]["eval_radius"]))]

    for i in range(len(ERR)):
        r = np.linalg.norm(metadata["rec_pos"][i])

        for j, radius in enumerate(config["valid"]["eval_radius"]):
            if r >= radius[0] and r < radius[1]:
                ERR_classes[j].append(ERR[i])
                rmse_classes[j].append(rmse[i])

    for i in range(len(ERR_classes)):
        if mode == "median":
            (line,) = plt.semilogx(
                faxis, np.median(np.array(ERR_classes[i]), axis=0), label=labels[i]
            )
            l.append(line)

        elif mode == "mean":
            (line,) = plt.semilogx(
                faxis, np.mean(np.array(ERR_classes[i]), axis=0), label=labels[i]
            )
            l.append(line)

        elif mode == "both":
            (line,) = plt.semilogx(
                faxis, np.median(np.array(ERR_classes[i]), axis=0), label=f"{labels[i]}"
            )
            plt.semilogx(faxis, np.mean(np.array(ERR_classes[i]), axis=0), "--")
            l.append(line)

        elif mode == "mean+SD":
            (line,) = plt.semilogx(
                faxis, np.mean(np.array(ERR_classes[i]), axis=0), label=labels[i]
            )
            plt.fill_between(
                faxis,
                np.mean(np.array(ERR_classes[i]), axis=0)
                - np.std(np.array(ERR_classes[i]), axis=0) / 2,
                np.mean(np.array(ERR_classes[i]), axis=0)
                + np.std(np.array(ERR_classes[i]), axis=0) / 2,
                alpha=0.3,
            )
            l.append(line)

    plt.grid(True, which="both")
    plt.xlabel(fig_params["xlabel"]), plt.ylabel(fig_params["ylabel"])
    plt.xlim(fig_params["xlim"]), plt.ylim(fig_params["ylim"])
    if len(labels) > 4 and mode != "mean+SD":
        lg = plt.legend(
            handles=l,
            ncol=2,
            title=fig_params["lgtitle"],
            title_fontproperties={"weight": "bold"},
            loc="upper left",
        )
    else:
        lg = plt.legend(
            handles=l,
            ncol=1,
            title=fig_params["lgtitle"],
            title_fontproperties={"weight": "bold"},
            loc="upper left",
        )

    if mode == "both":
        plt.gca().add_artist(lg)

        l2 = []
        (line,) = plt.plot([], [], "k-", label="Median")
        l2.append(line)
        (line,) = plt.plot([], [], "k--", label="Mean")
        l2.append(line)
        plt.legend(handles=l2, loc="lower right")

    if mode == "mean+SD":
        plt.gca().add_artist(lg)

        l2 = []
        (line,) = plt.plot([], [], "k-")
        l2.append(line)
        line = plt.hist([], alpha=0.3, color="black")[-1]
        l2.append(line)
        plt.legend(handles=l2, labels=["Mean", "$\\pm$ SD"], loc="lower right")

    plt.tight_layout()

    if fig_params["save"]:
        plt.savefig(fig_params["fn"])

    return rmse_classes


def plotFrequencyDomainErrorPerDirection(
    fn, mic_positions, surf_postitions, tolerance, faxis, fig_params, mode="both"
):
    fig = plt.figure(figsize=(fig_params["width"], fig_params["height"]))

    if mode == "both":
        cc = cycler(color=fig_params["colors"]) * cycler(linestyle=["-", "--"])

    else:
        cc = cycler("color", fig_params["colors"]) * cycler(
            "linestyle", ["-", "--", "-."]
        )

    plt.rc("axes", prop_cycle=cc)

    l = []

    labels = [f"Mic $\\pm$ {tolerance}°", f"Surface $\\pm$ {tolerance}°"]

    data = np.load(os.path.join(fn_path, fn), allow_pickle=True)
    ERR = 10 * np.log10(data["est_error_psd"])
    rmse = 10 * np.log10(data["valid_rmse_per_epoch_samplewise"])
    metadata = data["metadata"].item()

    ERR_classes = [[] for _ in range(2)]
    rmse_classes = [[] for _ in range(2)]

    for i in range(len(ERR)):
        r = np.linalg.norm(metadata["rec_pos"][i])

        for j in range(4):
            angle_mic = angle_between_vectors(metadata["rec_pos"][i], mic_positions[j])
            angle_surf = angle_between_vectors(
                metadata["rec_pos"][i], surf_postitions[j]
            )

            if angle_mic < tolerance:
                ERR_classes[0].append(ERR[i])
                rmse_classes[0].append(rmse[i])
                break

            if angle_surf < tolerance:
                ERR_classes[1].append(ERR[i])
                rmse_classes[1].append(rmse[i])
                break

    for i in range(len(ERR_classes)):
        if mode == "median":
            (line,) = plt.semilogx(
                faxis, np.median(np.array(ERR_classes[i]), axis=0), label=labels[i]
            )
            l.append(line)

        elif mode == "mean":
            (line,) = plt.semilogx(
                faxis, np.mean(np.array(ERR_classes[i]), axis=0), label=labels[i]
            )
            l.append(line)

        elif mode == "both":
            (line,) = plt.semilogx(
                faxis, np.median(np.array(ERR_classes[i]), axis=0), label=f"{labels[i]}"
            )
            plt.semilogx(faxis, np.mean(np.array(ERR_classes[i]), axis=0), "--")
            l.append(line)

        elif mode == "mean+SD":
            (line,) = plt.semilogx(
                faxis, np.mean(np.array(ERR_classes[i]), axis=0), label=labels[i]
            )
            plt.fill_between(
                faxis,
                np.mean(np.array(ERR_classes[i]), axis=0)
                - np.std(np.array(ERR_classes[i]), axis=0) / 2,
                np.mean(np.array(ERR_classes[i]), axis=0)
                + np.std(np.array(ERR_classes[i]), axis=0) / 2,
                alpha=0.3,
            )
            l.append(line)

    plt.grid(True, which="both")
    plt.xlabel(fig_params["xlabel"]), plt.ylabel(fig_params["ylabel"])
    plt.xlim(fig_params["xlim"]), plt.ylim(fig_params["ylim"])
    if len(labels) > 4 and mode != "mean+SD":
        lg = plt.legend(
            handles=l,
            ncol=2,
            title=fig_params["lgtitle"],
            title_fontproperties={"weight": "bold"},
            loc="upper left",
        )
    else:
        lg = plt.legend(
            handles=l,
            ncol=1,
            title=fig_params["lgtitle"],
            title_fontproperties={"weight": "bold"},
            loc="upper left",
        )

    if mode == "both":
        plt.gca().add_artist(lg)

        l2 = []
        (line,) = plt.plot([], [], "k-", label="Median")
        l2.append(line)
        (line,) = plt.plot([], [], "k--", label="Mean")
        l2.append(line)
        plt.legend(handles=l2, loc="lower right")

    if mode == "mean+SD":
        plt.gca().add_artist(lg)

        l2 = []
        (line,) = plt.plot([], [], "k-")
        l2.append(line)
        line = plt.hist([], alpha=0.3, color="black")[-1]
        l2.append(line)
        plt.legend(handles=l2, labels=["Mean", "$\\pm$ SD"], loc="lower right")

    plt.tight_layout()

    if fig_params["save"]:
        plt.savefig(fig_params["fn"])

    return rmse_classes


def plotFrequenyDomainError(fn, labels, faxis, fig_params, mode="both"):
    fig = plt.figure(figsize=(fig_params["width"], fig_params["height"]))

    if mode == "both":
        cc = cycler(color=fig_params["colors"]) * cycler(linestyle=["-", "--"])

    else:
        cc = cycler("color", fig_params["colors"]) + cycler(
            "linestyle", ["-", "--", "-."]
        )

    plt.rc("axes", prop_cycle=cc)

    l = []

    for i in range(len(fn)):
        data = np.load(os.path.join(fn_path, fn[i]))
        ERR = 10 * np.log10(data["est_error_psd"])

        if mode == "median":
            (line,) = plt.semilogx(faxis, np.median(ERR, axis=0), label=labels[i])
            l.append(line)

        elif mode == "mean":
            (line,) = plt.semilogx(faxis, np.mean(ERR, axis=0), label=labels[i])
            l.append(line)

        elif mode == "both":
            (line,) = plt.semilogx(faxis, np.median(ERR, axis=0), label=f"{labels[i]}")
            plt.semilogx(faxis, np.mean(ERR, axis=0), "--")
            l.append(line)

        elif mode == "mean+SD":
            (line,) = plt.semilogx(faxis, np.mean(ERR, axis=0), label=labels[i])
            plt.fill_between(
                faxis,
                np.mean(ERR, axis=0) - np.std(ERR, axis=0) / 2,
                np.mean(ERR, axis=0) + np.std(ERR, axis=0) / 2,
                alpha=0.3,
            )
            l.append(line)

    plt.grid(True, which="both")
    plt.xlabel(fig_params["xlabel"]), plt.ylabel(fig_params["ylabel"])
    plt.xlim(fig_params["xlim"]), plt.ylim(fig_params["ylim"])
    if len(labels) > 4 and mode != "mean+SD":
        lg = plt.legend(
            handles=l,
            ncol=2,
            title=fig_params["lgtitle"],
            title_fontproperties={"weight": "bold"},
            loc="lower right",
        )
    else:
        lg = plt.legend(
            handles=l,
            ncol=1,
            title=fig_params["lgtitle"],
            title_fontproperties={"weight": "bold"},
            loc="lower right",
        )

    if mode == "both":
        plt.gca().add_artist(lg)

        l2 = []
        (line,) = plt.plot([], [], "k-", label="Median")
        l2.append(line)
        (line,) = plt.plot([], [], "k--", label="Mean")
        l2.append(line)
        plt.legend(handles=l2, loc="lower right")

    if mode == "mean+SD":
        plt.gca().add_artist(lg)

        l2 = []
        (line,) = plt.plot([], [], "k-")
        l2.append(line)
        line = plt.hist([], alpha=0.3, color="black")[-1]
        l2.append(line)
        plt.legend(handles=l2, labels=["Mean", "\\pm$ SD"], loc="lower right")

    plt.tight_layout()

    if fig_params["save"]:
        plt.savefig(fig_params["fn"])


# %% Plotting stuff
csv_save = True
plt_save = True
plt_colors = ["tab:red", "tab:green", "tab:blue"]

fig_path = "figures/paper"
plt_oversample = 1
plt_width = 3.2 * plt_oversample
plt_fontsize = 10 * plt_oversample
plt_legend_fontsize = 8 * plt_oversample
linewidth = 1.5 * plt_oversample

if not os.path.exists(fig_path):
    os.makedirs(fig_path)

plt.rcParams.update(
    {
        "text.latex.preamble": r"\usepackage{times}\usepackage{mathptmx}",
        "font.size": plt_fontsize,
        "text.usetex": True,
        "font.serif": ["Times New Roman"],
        "font.family": "serif",
        "lines.linewidth": linewidth,
        "legend.fontsize": plt_legend_fontsize,
        "legend.title_fontsize": plt_legend_fontsize,
    }
)

# %% Paths and parameters
fn_path = "export"

fn = [
    "FA_varpos_singlesource_epoch_1000.npz",
    "FA_varpos_singlesource_valfixedpos_epoch_1000.npz",
    "FA_varpos_singlesource_fixedpos_epoch_1000.npz",
]
lb = [
    "train+val",
    "train only",
    "none",
]

config = toml.load("configs/FA_varpos_singlesource.toml")

nFFT = 1024
fs = 16000

angular_tolerance = 10

faxis = np.linspace(0, fs / 2, nFFT // 2 + 1)

# %% Calculate mean and median RMSE
for j in range(len(fn)):
    data = np.load(os.path.join(fn_path, fn[j]))
    RMSE = 10 * np.log10(data["valid_rmse_per_epoch_samplewise"])
    print(
        f"Median RMSE {lb[j]}: {np.median(RMSE):.2f} dB; mean: {np.mean(RMSE):.2f} dB; std: {np.std(RMSE):.2f} dB"
    )

print("\n")


# %% Experiment 1: Dependence on position data
fig_params = {
    "width": plt_width,
    "height": plt_width / 1.8,
    "xlim": [50, 8000],
    "ylim": [-65, 2],
    "xlabel": "Frequency / Hz",
    "ylabel": "Estimation error / dB",
    "lgtitle": "\\textbf{Position data}",
    "save": plt_save,
    "fn": os.path.join(fig_path, "FA_metrics_broadband.pdf"),
    "colors": plt_colors,
}

plotFrequenyDomainError(fn, lb, faxis, fig_params, mode="mean")


# %% Experiment 2: distance from centre
fig_params = {
    "width": plt_width,
    "height": plt_width / 1.8,
    "xlim": [50, 8000],
    "ylim": [-55, 2],
    "xlabel": "Frequency / Hz",
    "ylabel": "Estimation error / dB",
    "lgtitle": "\\textbf{Distance from centre}",
    "save": plt_save,
    "fn": os.path.join(fig_path, "FA_metrics_per_recposition.pdf"),
    "colors": plt_colors,
}
rmse_positionwise = plotFrequencyDomainErrorPerPosition(
    fn[0], faxis, fig_params, config, mode="mean"
)

for j in range(len(rmse_positionwise)):
    print(
        f"Median RMSE {config['valid']['eval_radius'][j][0]*100} cm < r < {config['valid']['eval_radius'][j][1]*100} cm: {np.median(rmse_positionwise[j]):.2f} dB; mean: {np.mean(rmse_positionwise[j]):.2f} dB; std: {np.std(rmse_positionwise[j]):.2f} dB"
    )
print("\n")


# %% Experiment 3: direction of source
mic_positions = (
    np.array(config["remote_mic"]["position"]) * config["remote_mic"]["scale"]
)
surf_normvector = mic_positions * -1

fig_params = {
    "width": plt_width * 0.7,
    "height": plt_width * 0.7,
    "save": plt_save,
    "fn": os.path.join(fig_path, "FA_tetrahedron.pdf"),
    "colors": plt_colors,
}

fig_params = {
    "width": plt_width,
    "height": plt_width / 1.8,
    "xlim": [50, 8000],
    "ylim": [-55, 2],
    "xlabel": "Frequency / Hz",
    "ylabel": "Estimation error / dB",
    "lgtitle": "\\textbf{Primary direction}",
    "save": plt_save,
    "fn": os.path.join(fig_path, "FA_metrics_per_direction.pdf"),
    "colors": plt_colors,
}
rmse_directionwise = plotFrequencyDomainErrorPerDirection(
    fn[0],
    mic_positions,
    surf_normvector,
    angular_tolerance,
    faxis,
    fig_params,
    mode="mean",
)

for j, case in enumerate(["mic", "surface"]):
    print(
        f"Median RMSE {case}: {np.median(rmse_directionwise[j]):.2f} dB; mean: {np.mean(rmse_directionwise[j]):.2f} dB; std: {np.std(rmse_directionwise[j]):.2f} dB"
    )

# %% Export csv of data
rmse_tmp = []

for i in range(len(fn)):
    data = np.load(os.path.join(fn_path, fn[i]))
    rmse_tmp.append(10 * np.log10(data["valid_rmse_per_epoch_samplewise"]))

rmse_tmp = np.array(rmse_tmp)

d = {
    "Experiment": lb,
    "Median": np.median(rmse_tmp, axis=1),
    "Mean": np.mean(rmse_tmp, axis=1),
    "SD": np.std(rmse_tmp, axis=1),
}
df = pd.DataFrame(d)

if csv_save:
    df.to_csv("export/FA_overall_metrics.csv", index=False, float_format="%.3f")


# Metrics per position
med_per_pos = (np.median(posclass) for posclass in rmse_positionwise)
mean_per_pos = (np.mean(posclass) for posclass in rmse_positionwise)
std_per_pos = (np.std(posclass) for posclass in rmse_positionwise)

range_idx = []
for i in range(len(config["valid"]["eval_radius"])):
    range_idx.append(
        "$r \\in \\left["
        + str(config["valid"]["eval_radius"][i][0] * 100)
        + ","
        + str(config["valid"]["eval_radius"][i][1] * 100)
        + "\\right) \\si{\\centi\\metre}$"
    )

d = {
    "index": range(len(config["valid"]["eval_radius"])),
    "range": range_idx,
    "Median": med_per_pos,
    "Mean": mean_per_pos,
    "SD": std_per_pos,
}
df = pd.DataFrame(d)

if csv_save:
    df.to_csv("export/FA_pos_metrics.csv", index=False, float_format="%.3f")


# Metrics per direction
med_per_dir = (np.median(dirclass) for dirclass in rmse_directionwise)
mean_per_dir = (np.mean(dirclass) for dirclass in rmse_directionwise)
std_per_dir = (np.std(dirclass) for dirclass in rmse_directionwise)

d = {
    "direction_idx": ["mic", "surface"],
    "Median": med_per_dir,
    "Mean": mean_per_dir,
    "SD": std_per_dir,
}
df = pd.DataFrame(d)

if csv_save:
    df.to_csv("export/FA_dir_metrics.csv", index=False, float_format="%.3f")


# %% Statistics
# Metrics per primary direction
p_dir = stats.ttest_ind(rmse_directionwise[0], rmse_directionwise[1]).pvalue

print(f"RMSE mic vs. surface: p-value (ind t-test): {p_dir:.3f}")

d = {
    "Experiment": ["mic vs. surface"],
    "p-value": p_dir,
}
df = pd.DataFrame(d)

if csv_save:
    df.to_csv("export/FA_dir_statistics.csv", index=False, float_format="%.3f")

plt.show()
