import os
import toml
import torch, torchaudio
import io
import numpy as np
import sys
import tqdm
import matplotlib.pyplot as plt
import argparse
from typing import Union
import PIL.Image
import wandb

from torch.utils import data
from ptflops import get_model_complexity_info
from utils.dataset import DirectionalNoiseDatasetPrerendered
from utils import models
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor
from utils.metrics import NMSE
from utils.transforms import OverlapSave


def write_loss_summary(
    tag: str, tb_writer: SummaryWriter, wandb_run, loss: float, step: int
):
    tb_writer.add_scalar(tag, loss, step)
    wandb_run.log({tag: loss}, step=step)
    tb_writer.close()


def write_losshist_summary(
    tag: str, tb_writer: SummaryWriter, wandb_run, loss_hist: np.array, step: int
):
    tb_writer.add_histogram(tag, loss_hist, step)
    wandb_run.log({tag: wandb.Histogram(loss_hist)}, step=step)
    tb_writer.close()


def write_pyplotfigure_summary(
    tag: str, tb_writer: SummaryWriter, wandb_run, plot_buf: io.BytesIO, step: int
):
    image = PIL.Image.open(plot_buf)
    wandb_run.log({tag: wandb.Image(image)}, step=step)

    image = ToTensor()(image)

    tb_writer.add_image(tag, image, step)
    tb_writer.close()


def write_audio_summary(
    tag: str,
    tb_writer: SummaryWriter,
    wandb_run,
    vm_target: torch.Tensor,
    vm_estimated: torch.Tensor,
    error: torch.Tensor,
    step: int,
    fs: int,
):

    full_tag_target = tag + "/target"
    full_tag_est = tag + "/estimated"
    full_tag_error = tag + "/error"

    tb_writer.add_audio(full_tag_target, vm_target, step, sample_rate=fs)
    tb_writer.add_audio(full_tag_est, vm_estimated, step, sample_rate=fs)
    tb_writer.add_audio(full_tag_error, error, step, sample_rate=fs)
    tb_writer.close()

    wandb_run.log(
        {
            full_tag_target: wandb.Audio(vm_target, sample_rate=fs),
            full_tag_est: wandb.Audio(vm_estimated, sample_rate=fs),
            full_tag_error: wandb.Audio(error, sample_rate=fs),
        },
        step=step,
    )


def write_checkpoint(
    obj: Union[torch.optim.Optimizer, torch.nn.Module],
    name: str,
    dir: str,
    epoch: int,
    extension="ckpt",
):

    filename = name + str(epoch) + "." + extension
    cp_name = os.path.join(dir, filename)
    torch.save(obj.state_dict(), cp_name)
    print("Checkpoint '" + filename + "' for epoch " + str(epoch) + " has been stored.")


def load_checkpoint(dirname: str, file_list: str, extension: str):
    # get latest checkpoint
    epochs = [i.split("_", -1)[-1] for i in file_list]
    epochs = [int(i.split(".", -1)[0]) for i in epochs]
    latest_epoch = max(epochs)
    latest_substring = "_" + str(latest_epoch) + extension
    latest_ckpts = [latest_substring in d for d in file_list]
    temp = np.array(file_list)
    latest_ckpt_files = temp[latest_ckpts]

    try:
        assert len(latest_ckpt_files) == 2
    except AssertionError:
        sys.exit(
            "there exist either too many checkpoint-files or one checkpoint-file is missing!"
        )

    model_idx = np.array(["model" in f for f in latest_ckpt_files])
    latest_model_ckpt = latest_ckpt_files[model_idx][0]
    latest_opt_ckpt = latest_ckpt_files[np.invert(model_idx)][0]

    model_state_dict = torch.load(
        os.path.join(dirname, latest_model_ckpt), map_location="cpu"
    )
    opt_state_dict = torch.load(
        os.path.join(dirname, latest_opt_ckpt), map_location="cpu"
    )

    print(
        "checkpoints '"
        + latest_model_ckpt
        + "' and '"
        + latest_opt_ckpt
        + "' for epoch "
        + str(latest_epoch)
        + " have been loaded!"
    )
    return model_state_dict, opt_state_dict, latest_epoch


def plot_spec_tb(
    mtx: np.array,
    epochs: np.array = None,
    f_axis: np.array = None,
    vmin: float = -20,
    vmax: float = 10,
    dB: bool = True,
    title: str = None,
):
    nBins = mtx.shape[0]
    nEpochs = mtx.shape[1]

    if epochs is None:
        epochs = np.arange(0, nEpochs)

    if f_axis is None:
        f_axis = np.arange(0, nBins)
        f_axis_label = "Frequency"
    else:
        f_axis_label = "Frequency / Hz"

    if dB:
        mtx = 10 * np.log10(mtx)

    xmin = epochs[0]
    xmax = epochs[-1]
    extent = xmin, xmax, f_axis[0], f_axis[-1]

    plt.figure()
    plt.imshow(mtx, extent=extent, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    plt.xlabel("Epochs")
    plt.ylabel(f_axis_label)
    if title is not None:
        plt.title(title)

    cbar = plt.colorbar()
    if dB:
        cbar.set_label("Relative error / dB")
    else:
        cbar.set_label("Relative error")
    plt.axis("auto")

    buf = io.BytesIO()
    plt.savefig(buf, format="jpeg")
    buf.seek(0)

    return buf


def plot_spec_singlebatch(
    psd: np.array,
    f_axis: np.array,
    vmin: float = None,
    vmax: float = None,
    log_faxis: bool = True,
    dB: bool = True,
    title: str = None,
):
    nBins = psd.shape[0]

    if dB:
        psd = 10 * np.log10(np.real(np.abs(psd)))

    if f_axis is None:
        f_axis = np.arange(0, nBins)
        f_axis_label = "Frequency"
    else:
        f_axis_label = "Frequency / Hz"

    if vmin is None:
        vmin = np.min(psd) * 0.9
    if vmax is None:
        vmax = np.max(psd) * 1.1

    plt.figure()

    if log_faxis:
        plt.semilogx(f_axis, psd)
    else:
        plt.plot(f_axis, psd)

    plt.ylim([vmin, vmax])
    plt.grid(which="both")
    plt.xlabel(f_axis_label)

    if dB:
        plt.ylabel("Frequency / Hz")
    else:
        plt.ylabel("Frequency")

    if title is not None:
        plt.title(title)
    plt.axis("auto")

    buf = io.BytesIO()
    plt.savefig(buf, format="jpeg")
    plt.close()
    buf.seek(0)

    return buf


# %% Get config file provided as argument
parser = argparse.ArgumentParser()
parser.add_argument(
    "-c", "--config", nargs="?", const=1, type=str, default="config.toml"
)

args = parser.parse_args()
config = toml.load(args.config)

os.environ["CUDA_VISIBLE_DEVICES"] = str(config["cuda_visible_devices"])

# %% Load heaivily used config parameters
NFFT = config["nfft"]
HOPSIZE = config["model"]["hopsize"]
FS = config["samplerate"]
EPOCHS = config["train"]["max_epochs"]
VALID_ONLY = config["valid"]["only"]
CKPT_PATH = os.path.join(config["checkpoint_path"], config["filename"])

# Calculate derived parameters
N_REM_MICS = len(config["remote_mic"]["position"])
aperture = 2 * np.sqrt(2) * config["remote_mic"]["scale"]
max_lag = int(np.ceil(np.max(aperture) / config["data"]["c"] * FS))

# %% Setup tensorboard and wandb
writer = SummaryWriter(
    os.path.join(config["tensorboard_path"], config["filename"]),
    filename_suffix=".tlog",
)
os.makedirs(CKPT_PATH, exist_ok=True)

wandb.login()
run = wandb.init(
    project="FA_DeepObservationFilter",
    name=config["filename"],
    config=config,
    mode="disabled",
)

# %% Set random seed and devices
np.random.seed(config["train"]["seed"])

if torch.cuda.is_available():
    train_device = torch.device("cuda")
    valid_device = torch.device("cuda")
    print("CUDA is available!")

elif torch.backends.mps.is_available():
    train_device = torch.device("mps")
    valid_device = torch.device("mps")
    print("MPS is available!")
else:
    train_device = torch.device("cpu")
    valid_device = torch.device("cpu")
    print("CUDA and MPS are not available!")


# %% Initialize model, optimizer and loss
model_to_train = getattr(models, config["model"]["name"])(config)
model_to_train.to(train_device)

input_fun = lambda inp_shape: {
    "x": torch.FloatTensor(torch.empty(inp_shape)).to(train_device),
    "virt_coords": torch.FloatTensor(torch.empty(3)).to(train_device),
}

macs, params = get_model_complexity_info(
    model_to_train,
    (N_REM_MICS, NFFT // 2 + 1, config["train"]["frames_per_cc"]),
    input_constructor=input_fun,
    as_strings=True,
    backend="pytorch",
    print_per_layer_stat=True,
    verbose=True,
)

print("{:<30}  {:<8}".format("Computational complexity: ", macs))
print("{:<30}  {:<8}".format("Number of parameters: ", params))


if config["train"]["optimizer"].lower() == "adam":
    optimizer = torch.optim.Adam(
        model_to_train.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
    )
elif config["train"]["optimizer"].lower() == "adamw":
    optimizer = torch.optim.AdamW(
        model_to_train.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
    )
else:
    sys.exit("Optimizer not supported!")

loss_fn = torch.nn.MSELoss()
NMSE = NMSE(return_dB=False, per_sample=True)


# %% Load checkpoint if available
ckpt_file_list = os.listdir(CKPT_PATH)
if len(ckpt_file_list) >= 2:
    # if at least two checkpoint files (model and optimizer) exist => load checkpoints
    latest_model_ckpt, latest_opt_ckpt, start_epoch = load_checkpoint(
        CKPT_PATH, ckpt_file_list, ".ckpt"
    )
    start_epoch += 1
    model_to_train.load_state_dict(latest_model_ckpt)
    optimizer.load_state_dict(latest_opt_ckpt)
    # if training is restarted with new lr => set lr again so it is not overwritten by checkpoint
    optimizer.param_groups[0]["lr"] = config["train"]["lr"]
else:
    # no checkpoint files are found => create new ones at given directory path
    print("No checkpoints found! Training starts with epoch #0!")
    start_epoch = 0


# %% initialize dataloader
train_ds = DirectionalNoiseDatasetPrerendered(
    config, config["data"]["directory"] + "/train/"
)
valid_ds = DirectionalNoiseDatasetPrerendered(
    config, config["data"]["directory"] + "/valid/"
)

train_dl = data.DataLoader(
    train_ds,
    batch_size=config["train"]["batch_size"],
    num_workers=config["train"]["num_process"],
    shuffle=True,
    drop_last=True,
)

valid_dl = data.DataLoader(
    valid_ds,
    batch_size=config["valid"]["batch_size"],
    num_workers=config["valid"]["num_process"],
    shuffle=False,
    drop_last=True,
)


model_to_train.train()

train_loss_per_batch = []
train_loss_per_epoch = []

valid_loss_per_epoch = []
valid_NMSE_per_epoch = []
valid_best_NMSE_per_epoch = []
valid_worst_NMSE_per_epoch = []
valid_median_NMSE_per_epoch = []

valid_epoch_ctr = []
valid_mean_error_fd_per_epoch = []
valid_median_error_fd_per_epoch = []

if VALID_ONLY:
    valid_epoch = start_epoch - 1
else:
    valid_epoch = start_epoch

N_log = config["N_log_epochs"]
ckpt_counter = N_log
log_counter = N_log

window = torch.hann_window(NFFT).to(train_device)
OS = OverlapSave(nfft=NFFT, hopsize=HOPSIZE, complex_input=False)

# %% Add one epoch if only validation is performed to trick max epoch setting
if VALID_ONLY:
    EPOCHS += 1

# %% Training loop
for epoch in tqdm.tqdm(range(start_epoch, EPOCHS), desc="epochs", position=0):
    train_loss_per_batch = []

    torch.cuda.empty_cache()

    for batch, (rm, vm, metadata) in enumerate(train_dl):
        # Skip training if only validation is performed
        if VALID_ONLY == True:
            break

        model_to_train.train()

        # Get signals, already STFT-transformed
        rm = rm.to(train_device)
        vm = vm.to(train_device)

        # Fourier transform for GCC and overlap-save filtering
        rm_fd = OS(rm, to_fd=True, reset=True)
        rm_fd_gcc = OS.transform_for_gcc(rm, reset=True)

        # Use either accuate position or center of virtual microphone area
        if config["train"]["fixed_position"]:
            virtual_pos = (
                torch.tensor(config["virtual_mic"]["position"])
                .to(train_device)
                .repeat(config["train"]["batch_size"], 1)
            )
        else:
            virtual_pos = metadata["rec_pos"].type(torch.FloatTensor).to(train_device)

        # Calculate required count of optimization steps per sample
        n_optimizations = int(
            np.ceil(
                (rm_fd.shape[-1] - config["train"]["frames_per_cc"])
                / config["train"]["optimization_interval"]
            )
        )

        # Get initial beamformer weights
        o_coefficients = model_to_train(
            rm_fd_gcc[..., 0 : config["train"]["frames_per_cc"]],
            virtual_pos,
            reset_corr=True,
        )

        # Perform signal estimation for each segment
        for segment in range(n_optimizations):
            # Get start and end frame for current segment
            start_frame = (
                segment * config["train"]["optimization_interval"]
                + config["train"]["frames_per_cc"]
            )
            end_frame = np.min(
                [
                    start_frame + config["train"]["optimization_interval"],
                    rm_fd.shape[-1],
                ]
            )

            # Perform filter operation
            vm_est_fd = torch.sum(
                rm_fd[..., start_frame:end_frame] * o_coefficients.unsqueeze(-1), 1
            )
            vm_estimated = OS(vm_est_fd, to_fd=False)

            # Get target signal corresponding to current frame
            start_sample = start_frame * HOPSIZE
            end_sample = (end_frame) * HOPSIZE
            if end_sample > vm.shape[2]:
                end_sample = vm.shape[2]

            vm_target = torch.squeeze(vm[..., start_sample:end_sample])

            # Calculate loss
            if config["train"]["normalized_loss"]:
                rms = torch.sqrt(torch.mean(vm_target**2, dim=-1)).unsqueeze(-1)
                loss = loss_fn(vm_estimated / rms, vm_target / rms)
            else:
                loss = loss_fn(vm_estimated, vm_target)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Store loss for current batch
            train_loss_per_batch.append(loss.detach().cpu().numpy())

            # Calculate filter coefficients for next segment
            start_frame_corr = np.max(end_frame - config["train"]["frames_per_cc"], 0)
            o_coefficients = model_to_train(
                rm_fd_gcc[..., start_frame_corr:end_frame], virtual_pos
            )

    ckpt_counter -= 1
    log_counter -= 1

    if not (VALID_ONLY):
        if log_counter == 0 or epoch == 0 or epoch == EPOCHS - 1:
            train_loss_per_epoch.append(np.mean(train_loss_per_batch))

            write_loss_summary(
                "training loss (MSE)",
                writer,
                run,
                train_loss_per_epoch[-1],
                epoch,
            )

            torch.cuda.empty_cache()

        # save checkpoint every N_log_ckpt epochs
        if ckpt_counter == 0 or epoch == 0 or epoch == EPOCHS - 1:
            write_checkpoint(
                model_to_train, "DeepObservationfilter_model_", CKPT_PATH, epoch, "ckpt"
            )
            write_checkpoint(
                optimizer, "DeepObservationfilter_optimizer_", CKPT_PATH, epoch, "ckpt"
            )

    # Validation
    if log_counter == 0 or VALID_ONLY or epoch == 0 or epoch == EPOCHS - 1:
        valid_epoch_ctr.append(valid_epoch)

        with torch.no_grad():
            valid_loss_per_batch = []

            torch.manual_seed(config["valid"]["seed"] + 2)
            N_batches = len(valid_dl)

            # Initialize lists for samplewise metrics
            metadata_samplewise = []
            valid_NMSE_per_epoch_samplewise = torch.zeros(
                (N_batches, config["valid"]["batch_size"])
            )
            est_error_psd = np.zeros(
                (
                    N_batches,
                    config["valid"]["batch_size"],
                    config["valid"]["psd_nfft"] // 2 + 1,
                ),
            )

            # Debug random audio sample per validation
            if config["valid"]["debug_audio"]:
                batch_idx = np.random.randint(0, N_batches)
                sample_idx = np.random.randint(0, config["valid"]["batch_size"])

            model_to_train.eval()

            for batch, (rm, vm, metadata) in enumerate(valid_dl):
                metadata_samplewise.append(metadata)

                rm = rm.to(valid_device)
                vm = vm.to(valid_device)

                # Fourier transform for GCC and overlap-save filtering
                rm_fd = OS(rm, to_fd=True, reset=True)
                rm_fd_gcc = OS.transform_for_gcc(rm, reset=True)

                # Use either accuate position or center of virtual microphone area
                if config["valid"]["fixed_position"]:
                    virtual_pos = (
                        torch.tensor(config["virtual_mic"]["position"])
                        .to(valid_device)
                        .repeat(config["valid"]["batch_size"], 1)
                    )
                else:
                    virtual_pos = (
                        metadata["rec_pos"].type(torch.FloatTensor).to(valid_device)
                    )

                # Initialize estimated signal buffer
                vm_est_fd_batch = torch.zeros(
                    (rm_fd.shape[0], rm_fd.shape[2], rm_fd.shape[3]),
                    dtype=torch.complex64,
                    requires_grad=False,
                    device=valid_device,
                )

                # Calculate required count of optimization steps per sample
                n_optimizations = int(
                    np.ceil(
                        (rm_fd.shape[-1] - config["valid"]["frames_per_cc"])
                        / config["valid"]["inference_interval"]
                    )
                )

                # Get initial filter coefficients
                o_coefficients = model_to_train(
                    rm_fd_gcc[..., 0 : config["valid"]["frames_per_cc"]],
                    virtual_pos,
                    reset_corr=True,
                )

                # Perform signal estimation for each segment
                for segment in range(n_optimizations):
                    # Get start and end frame for current segment
                    start_frame = (
                        segment * config["valid"]["inference_interval"]
                        + config["valid"]["frames_per_cc"]
                    )
                    end_frame = np.min(
                        [
                            start_frame + config["valid"]["inference_interval"],
                            rm_fd.shape[-1],
                        ]
                    )

                    # Perform filter operation
                    vm_est_fd = torch.sum(
                        rm_fd[..., start_frame:end_frame]
                        * o_coefficients.unsqueeze(-1),
                        1,
                    )
                    vm_estimated = OS(vm_est_fd, to_fd=False)
                    vm_est_fd_batch[..., start_frame:end_frame] = vm_est_fd

                    # Get target signal corresponding to current frame
                    start_sample = start_frame * HOPSIZE
                    end_sample = (end_frame) * HOPSIZE
                    if end_sample > vm.shape[2]:
                        end_sample = vm.shape[2]

                    vm_target = torch.squeeze(vm[..., start_sample:end_sample])

                    # Calculate loss
                    loss = loss_fn(vm_estimated, vm_target)

                    valid_loss_per_batch.append(loss.detach().cpu().numpy())

                    # Calculate correlation matrix for next segment
                    start_frame_corr = np.max(
                        end_frame - config["valid"]["frames_per_cc"], 0
                    )

                    # Calculate filter coefficients for next segment
                    o_coefficients = model_to_train(
                        rm_fd_gcc[..., start_frame_corr:end_frame], virtual_pos
                    )

                start_frame = config["valid"]["frames_per_cc"]
                start_sample = start_frame * HOPSIZE
                vm_target = torch.squeeze(vm[..., start_sample:])

                vm_estimated = OS(vm_est_fd_batch[..., start_frame:], to_fd=False)

                max_vm_length = torch.min(
                    torch.tensor([vm_target.shape[-1], vm_estimated.shape[-1]])
                )

                vm_target = vm_target[..., :max_vm_length]
                vm_estimated = vm_estimated[..., :max_vm_length]

                # Calculate NMSE for each sample in batch
                valid_NMSE_per_epoch_samplewise[batch, :] = NMSE(
                    vm_target, vm_estimated
                )

                # Calculate PSD using torchaudio
                win_psd = torch.hann_window(config["valid"]["psd_nfft"]).to(
                    valid_device
                )
                vm_target_stft_for_psd = torch.unsqueeze(
                    torch.stft(
                        vm_target,
                        n_fft=config["valid"]["psd_nfft"],
                        hop_length=config["valid"]["psd_nfft"] // 2,
                        window=win_psd,
                        onesided=True,
                        return_complex=True,
                    ),
                    1,
                )
                difference_stft_for_psd = torch.unsqueeze(
                    torch.stft(
                        vm_target - vm_estimated,
                        n_fft=config["valid"]["psd_nfft"],
                        hop_length=config["valid"]["psd_nfft"] // 2,
                        window=win_psd,
                        onesided=True,
                        return_complex=True,
                    ),
                    1,
                )

                f_axis = np.linspace(0, FS // 2, config["valid"]["psd_nfft"] // 2 + 1)
                est_error_diff = torchaudio.functional.psd(difference_stft_for_psd)[
                    ..., 0, 0
                ].real
                psd_target = torchaudio.functional.psd(vm_target_stft_for_psd)[
                    ..., 0, 0
                ].real

                est_error_psd[batch] = (est_error_diff / psd_target).cpu().numpy()

                # Debug audio sample
                if config["valid"]["debug_audio"] and batch == batch_idx:
                    write_audio_summary(
                        "validation audio",
                        writer,
                        run,
                        vm_target[sample_idx].cpu(),
                        vm_estimated[sample_idx].cpu(),
                        (vm_target[sample_idx] - vm_estimated[sample_idx]).cpu(),
                        step=epoch,
                        fs=FS,
                    )

            # Save metrics for each sample in batch
            if config["valid"]["export_metrics"]:
                fn = os.path.join(
                    config["valid"]["export_path"],
                    config["filename"] + f"_epoch_{epoch}.npz",
                )

                # Convert metadata dict content to numpy array
                tmp_dict = {}

                for k in metadata_samplewise[0]:
                    tmp_dict[k] = (
                        torch.cat([d[k] for d in metadata_samplewise]).cpu().numpy()
                    )

                np.savez(
                    fn,
                    est_error_psd=est_error_psd.reshape(
                        est_error_psd.shape[0] * est_error_psd.shape[1], -1
                    ),
                    valid_NMSE_per_epoch_samplewise=valid_NMSE_per_epoch_samplewise.flatten()
                    .cpu()
                    .numpy(),
                    metadata=tmp_dict,
                )

            valid_NMSE_per_epoch_samplewise = 10 * torch.log10(
                valid_NMSE_per_epoch_samplewise
            )

            # Calculate mean and median NMSE for each sample in batch and show in tensorboard/wandb
            valid_NMSE_per_epoch.append(
                torch.mean(valid_NMSE_per_epoch_samplewise).cpu().numpy()
            )
            valid_best_NMSE_per_epoch.append(
                torch.min(valid_NMSE_per_epoch_samplewise).cpu().numpy()
            )
            valid_worst_NMSE_per_epoch.append(
                torch.max(valid_NMSE_per_epoch_samplewise).cpu().numpy()
            )
            valid_median_NMSE_per_epoch.append(
                torch.median(valid_NMSE_per_epoch_samplewise).cpu().numpy()
            )
            valid_mean_error_fd_per_epoch.append(np.mean(est_error_psd, axis=(0, 1)))
            valid_median_error_fd_per_epoch.append(
                np.median(est_error_psd, axis=(0, 1))
            )
            im_to_plot = plot_spec_tb(
                np.array(valid_mean_error_fd_per_epoch).T,
                epochs=valid_epoch_ctr,
                f_axis=f_axis,
                vmin=-20,
                vmax=10,
                dB=True,
                title="mean validation error",
            )
            write_pyplotfigure_summary(
                "mean validation error spectrogram", writer, run, im_to_plot, epoch
            )
            im_to_plot = plot_spec_singlebatch(
                valid_mean_error_fd_per_epoch[-1],
                f_axis=f_axis,
                vmin=-30,
                vmax=15,
                log_faxis=True,
                dB=True,
                title="mean validation error",
            )
            write_pyplotfigure_summary(
                "mean validation error", writer, run, im_to_plot, epoch
            )
            im_to_plot = plot_spec_tb(
                np.array(valid_median_error_fd_per_epoch).T,
                epochs=valid_epoch_ctr,
                f_axis=f_axis,
                vmin=-20,
                vmax=10,
                dB=True,
                title="median validation error",
            )
            write_pyplotfigure_summary(
                "median validation error spectrogram", writer, run, im_to_plot, epoch
            )
            im_to_plot = plot_spec_singlebatch(
                valid_median_error_fd_per_epoch[-1],
                f_axis=f_axis,
                vmin=-30,
                vmax=15,
                log_faxis=True,
                dB=True,
                title="median validation error",
            )
            write_pyplotfigure_summary(
                "median validation error", writer, run, im_to_plot, epoch
            )

            valid_loss_per_epoch.append(np.mean(valid_loss_per_batch))

            write_loss_summary(
                "validation loss (MSE)", writer, run, valid_loss_per_epoch[-1], epoch
            )

            write_loss_summary(
                "mean validation NMSE in dB",
                writer,
                run,
                valid_NMSE_per_epoch[-1],
                epoch,
            )
            write_loss_summary(
                "median validation NMSE in dB",
                writer,
                run,
                valid_median_NMSE_per_epoch[-1],
                epoch,
            )
            write_loss_summary(
                "minimum validation NMSE in dB",
                writer,
                run,
                valid_worst_NMSE_per_epoch[-1],
                epoch,
            )
            write_loss_summary(
                "maximum validation NMSE in dB",
                writer,
                run,
                valid_best_NMSE_per_epoch[-1],
                epoch,
            )
            write_losshist_summary(
                "validation NMSE in dB",
                writer,
                run,
                valid_NMSE_per_epoch_samplewise.cpu().numpy(),
                epoch,
            )

    if log_counter == 0:
        log_counter = N_log
    if ckpt_counter == 0:
        ckpt_counter = N_log

    valid_epoch += 1

    if VALID_ONLY:
        break

wandb.finish()
print("Training/validation finished!")
