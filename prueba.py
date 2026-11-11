import os
import secrets
import time
from typing import Literal, get_args

import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from google import genai
from google.genai import types
from pydantic import BaseModel

load_dotenv()
client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"],
    http_options=types.HttpOptions(timeout=12000),  # 12 s por intento (mínimo permitido: 10 s)
)
MODELOS = ["gemini-3.8-flash", "gemini-3.7-flash"]
TOKEN = os.environ.get("TOKEN_ROBOT", "")
MOSTRAR_TIEMPO = True  # pon False cuando conectes el ESP32

app = FastAPI()

Accion = Literal[
    "saludar",
    "levantar_brazo_izq",
    "levantar_brazo_der",
    "levantar_ambos_brazos",
    "bajar_brazos",
    "girar_cabeza_izq",
    "girar_cabeza_der",
    "centrar_cabeza",
    "asentir",
    "negar",
    "reposo",
    "detener",
]
COMANDOS_VALIDOS = set(get_args(Accion))

SISTEMA = (
    "Eres el controlador de un robot con cabeza, torso y dos brazos. No tiene "
    "piernas, pies ni dedos. Traduce la instrucción del usuario a UNA acción. "
    "Acciones: saludar (agitar un brazo), levantar_brazo_izq, "
    "levantar_brazo_der, levantar_ambos_brazos, bajar_brazos, "
    "girar_cabeza_izq, girar_cabeza_der, centrar_cabeza, asentir (decir sí "
    "con la cabeza), negar (decir no con la cabeza), reposo (brazos abajo y "
    "cabeza al centro), detener (parar el movimiento actual). Izquierda y "
    "derecha son las del robot. Si piden algo que el robot no puede hacer "
    "(caminar, agarrar objetos, etc.) o la instrucción es ambigua o "
    "insegura, responde detener."
    "Tu lógica de decisión es tener en cuenta que tienes dos brazos con servomotores de "
    "180 grados, una cabeza con servomotor de 180 grados que solo permite girar a izquierda y derecha,"
    "y un torso."
)


class Peticion(BaseModel):
    instruccion: str


class Comando(BaseModel):
    accion: Accion


def verificar(x_token: str = Header(default="")):
    if not TOKEN or not secrets.compare_digest(x_token, TOKEN):
        raise HTTPException(status_code=401, detail="Token inválido")


@app.middleware("http")
async def medir_tiempo(request: Request, call_next):
    t0 = time.time()
    response = await call_next(request)
    dt = time.time() - t0
    response.headers["X-Tiempo-Respuesta"] = f"{dt:.2f}s"
    print(f"[TOTAL] {request.method} {request.url.path} -> "
          f"{response.status_code} en {dt:.2f}s")
    return response


def agente(instruccion: str) -> dict:
    for modelo in MODELOS:
        print(f"[1] Probando {modelo}...")
        t0 = time.time()
        try:
            r = client.models.generate_content(
                model=modelo,
                contents=instruccion,
                config=types.GenerateContentConfig(
                    system_instruction=SISTEMA,
                    response_mime_type="application/json",
                    response_schema=Comando,
                ),
            )
            print(f"[2] {modelo} respondió en {time.time() - t0:.1f}s")
            print("[3] Texto crudo:", r.text)
            return r.parsed.model_dump()
        except Exception as e:
            print(f"[ERROR] {modelo} a los {time.time() - t0:.1f}s:", repr(e)[:200])
    print("[FALLO] Ningún modelo respondió")
    return {"accion": "detener"}


@app.post("/comando", dependencies=[Depends(verificar)])
def comando(p: Peticion):
    print("[0] Llegó la petición:", p.instruccion)
    t0 = time.time()
    respuesta = agente(p.instruccion)
    if respuesta["accion"] not in COMANDOS_VALIDOS:
        respuesta = {"accion": "detener"}
    if MOSTRAR_TIEMPO:
        respuesta["tiempo_s"] = round(time.time() - t0, 2)
    print("[4] Respondiendo:", respuesta)
    return respuesta


if __name__ == "__main__":
    # 0.0.0.0 = escuchar en toda la red, no solo en este equipo
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)