import os
import time

import librosa
import numpy as np
import pandas as pd

ROOT = "data/noise_free_data"  
LOCATIONS = ["CAL", "BRY"]  
OUT_DIR = "features_out"  
os.makedirs(OUT_DIR, exist_ok=True)


def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, sr=None)

    spectral_flux = librosa.onset.onset_detect(
        onset_envelope=librosa.onset.onset_strength(y=audio, sr=sample_rate),
        sr=sample_rate,
    )
    spectral_flux_mean = float(np.mean(spectral_flux) if spectral_flux.size else 0.0)

    rmse_mean = float(np.mean(librosa.feature.rms(y=audio)))
    spectral_bandwidth_mean = float(
        np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate))
    )
    spectral_flatness_mean = float(np.mean(librosa.feature.spectral_flatness(y=audio)))

    spectral_contrast_mean = np.mean(
        librosa.feature.spectral_contrast(y=audio, sr=sample_rate), axis=1
    )
    poly_features_mean = np.mean(
        librosa.feature.poly_features(y=audio, sr=sample_rate), axis=1
    )
    tonnetz_mean = np.mean(librosa.feature.tonnetz(y=audio, sr=sample_rate), axis=1)
    mfcc_mean = np.mean(
        librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13), axis=1
    )

    zcr_mean = float(np.mean(librosa.feature.zero_crossing_rate(y=audio)))
    tempo, _ = librosa.beat.beat_track(y=audio, sr=sample_rate)

    return np.hstack(
        [
            spectral_flux_mean,
            rmse_mean,
            spectral_bandwidth_mean,
            spectral_flatness_mean,
            spectral_contrast_mean,
            poly_features_mean,
            tonnetz_mean,
            mfcc_mean,
            zcr_mean,
            float(tempo),
        ]
    )


def feature_columns():
    return (
        ["file_name"]
        + [
            "spectral_flux_mean",
            "rmse_mean",
            "spectral_bandwidth_mean",
            "spectral_flatness_mean",
        ]
        + [f"spectral_contrast_{i}" for i in range(7)]
        + [f"poly_features_{i}" for i in range(2)]
        + [f"tonnetz_{i}" for i in range(6)]
        + [f"mfcc_{i}" for i in range(13)]
        + ["zcr_mean", "tempo", "Location"]
    )


def process_location(code):
    input_folder = os.path.join(ROOT, code, "MP3")
    if not os.path.isdir(input_folder):
        print(f"[{code}] Missing folder: {input_folder}")
        return None

    rows = []
    for file_name in os.listdir(input_folder):
        if not file_name.lower().endswith(".mp3"):
            continue
        fpath = os.path.join(input_folder, file_name)
        try:
            feats = extract_features(fpath)
            rows.append([file_name] + list(feats) + [code])
        except Exception as e:
            print(f"[{code}] Error {file_name}: {e}")

    if not rows:
        print(f"[{code}] No MP3s processed.")
        return None

    df = pd.DataFrame(rows, columns=feature_columns())
    out_csv = os.path.join(OUT_DIR, f"{code}_features_df.csv")
    df.to_csv(out_csv, index=False)
    print(f"[{code}] Saved -> {out_csv} ({len(df)} rows)")
    return out_csv


if __name__ == "__main__":
    t0 = time.time()


    csv_paths = []
    for code in LOCATIONS:
        p = process_location(code)
        if p:
            csv_paths.append(p)

  
    if csv_paths:
        dfs = [pd.read_csv(p) for p in csv_paths]
        final = pd.concat(dfs, ignore_index=True)
        final_path = os.path.join(OUT_DIR, "final_features.csv")
        final.to_csv(final_path, index=False)
        print(f"[ALL] Final merged -> {final_path} ({len(final)} rows)")

        # delete per-location CSVs after successful merge
        for p in csv_paths:
            try:
                os.remove(p)
                print(f"[CLEANUP] Deleted {p}")
            except OSError as e:
                print(f"[CLEANUP] Couldn't delete {p}: {e}")
    else:
        print("[ALL] Nothing to merge.")

    print(f"Done in {time.time() - t0:.1f}s")
