from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from inference.predictor import Predictor

app = FastAPI()
predictor = Predictor()


@app.post("/predict")
async def predict(audio_path: str):
    try:
        location = "CAL"
        audio_file = Path(audio_path)
        if not audio_file.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")

        preds = predictor.predict_file(audio_path=str(audio_file), location=location)

        print("\nPrediction DataFrame:\n", preds, "\n")

        return JSONResponse(content={"predictions": preds.to_dict(orient="records")})
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")
