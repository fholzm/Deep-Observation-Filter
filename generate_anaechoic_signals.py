import numpy as np
import soundfile as sf
import pyroomacoustics as pra
import colorednoise as cn
from tqdm import tqdm
import toml
import os
from joblib import Parallel, delayed
import pandas as pd


def sph2cart(az: float, el: float, r: float):
    """Convert spherical coordinates to cartesian coordinates.

    Parameters
    ----------
    az : float
        Azimuth angle in radians.
    el : float
        Elevation angle in radians.
    r : float
        Radius.

    Returns
    -------
    tuple
        Cartesian coordinates (x,y,z).
    """
    x = r * np.cos(az) * np.cos(el)
    y = r * np.sin(az) * np.cos(el)
    z = r * np.sin(el)
    return x, y, z


def renderSignals(config, remote_positions, subdir, idx):
    # Randomize number of primary sources
    if config["data"]["nsrcrange"][0] == config["data"]["nsrcrange"][1]:
        n_src = config["data"]["nsrcrange"][0]
    else:
        n_src = np.random.randint(
            config["data"]["nsrcrange"][0], config["data"]["nsrcrange"][1]
        )

    # Randomize the primary signal's level
    if config["data"]["gainrange"][0] == config["data"]["gainrange"][1]:
        gain = []
        for j in range(n_src):
            gain.append(config["data"]["gainrange"][0])
    else:
        gain = 10 ** (
            np.random.uniform(
                config["data"]["gainrange"][0], config["data"]["gainrange"][1], n_src
            )
            / 20
        )

    # Randomize the noise exponent
    if (
        config["data"]["noise_exponentrange"][0]
        == config["data"]["noise_exponentrange"][1]
    ):
        noise_exponent = []
        for j in range(n_src):
            noise_exponent.append(config["data"]["noise_exponentrange"][0])
    else:
        noise_exponent = np.random.uniform(
            config["data"]["noise_exponentrange"][0],
            config["data"]["noise_exponentrange"][1],
            n_src,
        )

    # Generate the primary signals
    sig_in = np.empty((n_src, config["data"]["length"] * config["samplerate"] + 1000))
    for j in range(n_src):
        sig_in[j, :] = cn.powerlaw_psd_gaussian(
            noise_exponent[j], config["samplerate"] * config["data"]["length"] + 1000
        )

    # Randomize the primary source positions
    src_sph = np.random.uniform([-np.pi, -np.pi / 2], [np.pi, np.pi / 2], (n_src, 2))
    src_pos = np.array(
        sph2cart(src_sph[:, 0], src_sph[:, 1], config["source"]["distance"])
    )

    # Randomize the virtual microphone position
    if subdir == "/train/":
        radius_range = config["virtual_mic"]["radiusrange"]["train"]
    elif subdir == "/valid/":
        radius_range = config["virtual_mic"]["radiusrange"]["valid"]
    else:
        radius_range = config["virtual_mic"]["radiusrange"]

    virtual_pos_sph = np.random.uniform(
        [-np.pi, -np.pi / 2, radius_range[0]],
        [np.pi, np.pi / 2, radius_range[1]],
        3,
    )

    virtual_position = np.array(config["virtual_mic"]["position"]).T + np.array(
        sph2cart(virtual_pos_sph[0], virtual_pos_sph[1], virtual_pos_sph[2])
    )

    # Create the room model and place the microphones and sources
    room = pra.AnechoicRoom(fs=config["samplerate"])
    room.add_microphone_array(remote_positions)
    room.add_microphone(virtual_position)

    for j in range(n_src):
        room.add_source(src_pos[:, j], signal=sig_in[j, :] * gain[j])

    # Simulate the room
    room.set_sound_speed(config["data"]["c"])
    room.simulate()

    # Cut the signals to the desired length
    len_diff = (
        room.mic_array.signals.shape[1]
        - config["data"]["length"] * config["samplerate"]
    )
    sig = room.mic_array.signals[:, len_diff // 2 : -len_diff // 2]

    # Write to directory
    sf.write(
        config["data"]["directory"] + subdir + f"scene_{idx}.wav",
        sig.T,
        FS,
        "PCM_24",
    )

    # Return metadata
    return pd.DataFrame(
        {
            "src_pos": [src_pos],
            "src_az": [src_sph[:, 0]],
            "src_el": [src_sph[:, 1]],
            "src_tilt": [noise_exponent],
            "rec_pos": [virtual_position],
            "gain": [gain],
        },
        index=[idx],
    )


# Load the configuration file
config = toml.load("configs/FA_datagen_varpos_sph.toml")
np.random.seed(config["data"]["seed"])

# Create the directories
os.makedirs(config["data"]["directory"], exist_ok=True)
os.makedirs(config["data"]["directory"] + "/train", exist_ok=True)
os.makedirs(config["data"]["directory"] + "/valid", exist_ok=True)

# Define the positions of the remote microphones
remote_positions = (
    np.array(config["remote_mic"]["position"]) * config["remote_mic"]["scale"]
).T

# Define the most important parameters
FS = config["samplerate"]
N_SCENES = config["data"]["nscenes"]
SRC_DISTANCE = config["source"]["distance"]

N_SCENES_TRAIN = int(N_SCENES * config["data"]["train_split"])
N_SCENES_VALID = N_SCENES - N_SCENES_TRAIN

# Generate the signals for the training set
metadata = Parallel(n_jobs=10)(
    delayed(renderSignals)(config, remote_positions, "/train/", idx)
    for idx in tqdm(range(N_SCENES_TRAIN))
)
metadata = pd.concat(metadata).sort_index()
metadata.to_pickle(config["data"]["directory"] + "/train/metadata.pkl")

# Generate the signals for the validation set
metadata = Parallel(n_jobs=10)(
    delayed(renderSignals)(config, remote_positions, "/valid/", idx)
    for idx in tqdm(range(N_SCENES_VALID))
)
metadata = pd.concat(metadata).sort_index()
metadata.to_pickle(config["data"]["directory"] + "/valid/metadata.pkl")
