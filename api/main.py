from pathlib import Path
from fastapi.responses import JSONResponse
from fastapi import FastAPI, UploadFile, File

from inference.predict import get_xray_prediction

app = FastAPI()

@app.post("/predict")
def get_classification(file: UploadFile = File(...)):
    image_path = None
    try:
        image_path = Path(f"./temp/{file.filename}")
        image_path.parent.mkdir(parents=True, exist_ok=True)
        with open(image_path, "wb") as f:
            f.write(file.file.read())
        result = get_xray_prediction(image_path)
        return JSONResponse(
            status_code=200,
            content=result
        )
    except Exception as e:
        print(f"Got an exception: {e}")
        return JSONResponse(
            status_code=500,
            content="Endpoint is currently not working"
        )
    finally:
        if image_path and image_path.exists():
            image_path.unlink()