from fastapi import APIRouter, HTTPException, UploadFile, File, Form , Query
from pydantic import BaseModel
from fastapi.responses import FileResponse
import os
import pandas as pd

router = APIRouter()

ruta_actual = os.getcwd()

@router.get("/base")
def fun_ruta_actual():
    return {"mensaje": "Probando el route, todo OK"}

@router.get("/ruta-actual")
def fun_ruta_actual():
    return {f"mensaje: {ruta_actual}"} #/usr/src/app

# Función para subir un archivo CSV
@router.post("/upload-csv/")
async def upload_csv(file: UploadFile = File(...)):
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")

    # Guardar el archivo en el servidor (opcional)
    file_location = f"{ruta_actual}/{file.filename}"
    with open(file_location, "wb") as f:
        f.write(await file.read())

    # Leer el archivo CSV usando pandas
    try:
        df = pd.read_csv(file_location)
        # Puedes realizar operaciones con el DataFrame df si es necesario
        return {"filename": file.filename, "columns": df.columns.tolist(), "rows": len(df)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando el archivo: {e}")

    # Eliminar el archivo después de procesarlo (opcional)
    os.remove(file_location)