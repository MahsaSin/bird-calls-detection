import json
import os
from typing import Dict, List, Optional

import joblib
import librosa
import numpy as np
import pandas as pd


class Predictor:
    """
    Class wrapper for audio feature extraction and multilabel prediction.
    """

    # ---------- Construction ----------
    def __init__(self, bundle_path: str = "artifacts/metadata.json"):
        """
        Load model metadata and the trained model.
        """
        self.bundle_path = bundle_path
        self.bundle = self._load_bundle(bundle_path)

        self.feature_cols: List[str] = self.bundle["feature_columns"]
        self.target_cols: List[str] = self.bundle["target_columns"]
        self.loc_map: Dict[str, int] = self.bundle.get("location_mapping", {})
        self.model_path: str = self.bundle["model_path"]

        self.model = joblib.load(self.model_path)

   
    def predict_file(
        self, audio_path: str, location: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Predict tags for a single audio file. Returns a DataFrame with target columns + 'labels'.
        """
        feats = self.extract_features(audio_path)
        row = [os.path.basename(audio_path)] + list(feats)
        df = pd.DataFrame([row], columns=self.feature_columns())

        df = self._ensure_location(df, self.feature_cols, self.loc_map, location)

        X = df[self.feature_cols].astype(float)
        y_pred = self.model.predict(X)

        pred_df = pd.DataFrame(y_pred, columns=self.target_cols)
        pred_df["labels"] = [
            ";".join([lab for lab, v in r.items() if int(v) == 1])
            for _, r in pred_df.iterrows()
        ]
        return pred_df

    # ---------- Static/Helper Methods ----------
    @staticmethod
    def extract_features(file_path: str) -> np.ndarray:
        """
        Compute features for one audio file (mirrors original function).
        """
        audio, sample_rate = librosa.load(file_path, sr=None)

        spectral_flux = librosa.onset.onset_detect(
            onset_envelope=librosa.onset.onset_strength(y=audio, sr=sample_rate),
            sr=sample_rate,
        )
        spectral_flux_mean = float(
            np.mean(spectral_flux) if spectral_flux.size else 0.0
        )

        rmse_mean = float(np.mean(librosa.feature.rms(y=audio)))
        spectral_bandwidth_mean = float(
            np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate))
        )
        spectral_flatness_mean = float(
            np.mean(librosa.feature.spectral_flatness(y=audio))
        )

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

    @staticmethod
    def feature_columns() -> List[str]:
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
            + ["zcr_mean", "tempo"]
        )

    @staticmethod
    def _load_bundle(bundle_path: str) -> dict:
        with open(bundle_path, "r") as f:
            return json.load(f)

    @staticmethod
    def _ensure_location(
        df: pd.DataFrame,
        feature_cols: List[str],
        loc_map: Dict[str, int],
        user_loc: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Add numeric 'Location' column if the model expects it.
        """
        if "Location" in feature_cols and "Location" not in df.columns:
            if not user_loc:
                raise ValueError(
                    "This model expects a 'Location' feature. "
                    "Provide location (e.g., CAL, BRY)."
                )
            if user_loc not in loc_map:
                raise ValueError(
                    f"Unknown location '{user_loc}'. Known: {sorted(loc_map.keys())}"
                )
            df = df.copy()
            df["Location"] = loc_map[user_loc]
        return df


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    audio_file = r"CAL1_20220605_055000.mp3"
    predictor = Predictor(bundle_path="artifacts/metadata.json")
    preds = predictor.predict_file(audio_file, location="CAL")
    print(preds)
