from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from . import mis_rutas

app = FastAPI()

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(mis_rutas.router, prefix="/app-casas")

@app.get('/')
def health():
    return {
        "mensaje": "Bienvenido a mi Proyecto de DataPath. Soy Alan Turing"
    }