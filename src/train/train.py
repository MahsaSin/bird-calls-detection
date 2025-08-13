import json
from pathlib import Path

import joblib
import pandas as pd
from dotenv import load_dotenv
from omegaconf import OmegaConf
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

load_dotenv()


def train(conf_path="train/config.yaml"):
    conf = OmegaConf.load(conf_path)
    print(conf)

    cfg_dir = Path(conf_path).resolve().parent
    data_path = Path(conf.data_path)
    if not data_path.is_absolute():
        data_path = (cfg_dir / data_path).resolve()
    print("Reading:", data_path)

    df = pd.read_csv(data_path)

    # Location mapping
    default_loc_map = {"BRY": 0, "CAL": 1}
    loc_map = dict(getattr(conf, "location_mapping", default_loc_map))
    df["Location"] = df["Location"].map(loc_map)

    # Features / targets
    if hasattr(conf, "feature_columns") and conf.feature_columns:
        feature_columns = list(conf.feature_columns)
    else:
        feature_columns = (
            [
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

    target_columns = list(
        getattr(
            conf,
            "targets",
            [
                "AMRO",
                "BHCO",
                "CHSW",
                "EUST",
                "GRCA",
                "HOSP",
                "HOWR",
                "NOCA",
                "RBGU",
                "RWBL",
            ],
        )
    )

    X = df[feature_columns].astype(float)
    Y = df[target_columns].fillna(0).astype(int)  

    # SVM params
    C = float(getattr(conf.svm, "C", 1.9093325625779412))
    kernel = str(getattr(conf.svm, "kernel", "rbf"))
    gamma = str(getattr(conf.svm, "gamma", "scale"))
    random_state = int(getattr(conf.svm, "random_state", 42))

    
    base_svc = make_pipeline(
        StandardScaler(),
        SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            class_weight="balanced",
            random_state=random_state,
        ),
    )
    ovr = OneVsRestClassifier(base_svc)
    ovr.fit(X, Y)

    # Save
    Path("models").mkdir(parents=True, exist_ok=True)
    joblib.dump(ovr, "models/svm_ovr.joblib")

    bundle = {
        "feature_columns": feature_columns,
        "target_columns": target_columns,
        "location_mapping": loc_map,
        "model_path": "models/svm_ovr.joblib",
    }
    Path("artifacts").mkdir(parents=True, exist_ok=True)
    with open("artifacts/metadata.json", "w") as f:
        json.dump(bundle, f)

    print("Saved one model -> models/svm_ovr.joblib")
    print("Metadata -> artifacts/metadata.json")


if __name__ == "__main__":
    train()
