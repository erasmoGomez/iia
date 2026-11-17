import os
from pathlib import Path

from datetime import datetime, timedelta

import pandas as pd
from dotenv import load_dotenv
from google.adk.agents import Agent

# Carga las credenciales del archivo .env (está en esta misma carpeta)
load_dotenv(Path(__file__).parent / ".env")

# Lee los valores del .env
proveedor = os.getenv("PROVEEDOR", "gemini")  # gemini, openai u ollama
clave_openai = os.getenv("OPEN_API_KEY")
clave_ollama = os.getenv("OLLAMA_API_KEY")

if proveedor == "openai":
    from google.adk.models.lite_llm import LiteLlm

    modelo = LiteLlm(model="openai/gpt-5.6-luna", api_key=clave_openai)
elif proveedor == "ollama":
    from google.adk.models.lite_llm import LiteLlm

    # Ollama usa un formato compatible con OpenAI,
    # por eso el modelo empieza con "openai/"
    modelo = LiteLlm(
        model="openai/gemma4:31b",
        api_base="https://ollama.com/v1",  # dirección de la nube de Ollama
        api_key=clave_ollama,
    )
else:
    # Gemini lee la clave GEMINI_API_KEY que cargó load_dotenv
    modelo = "gemini-flash-lite-latest"

# Carpeta donde están los archivos CSV (junto a este archivo)
carpeta = Path(__file__).parent / "datos"


# Herramientas: funciones normales de Python.
# Cada una lee el CSV que necesita al empezar y, si cambia algo, lo guarda al terminar.
def scooters_disponibles() -> dict:
    """Lista los scooters que están disponibles para alquilar."""
    scooters = pd.read_csv(carpeta / "scooters.csv")
    scooters_libres = scooters[scooters["estado"] == "disponible"]
    return {
        "cantidad": len(scooters_libres),
        "scooters": scooters_libres.to_dict(orient="records"),
    }


def buscar_cliente(nombre: str) -> dict:
    """Busca clientes por nombre o parte del nombre. Ejemplo: 'Ana'."""
    clientes = pd.read_csv(carpeta / "clientes.csv")
    clientes_encontrados = clientes[clientes["nombre"].str.contains(nombre, case=False)]
    return {"clientes": clientes_encontrados.to_dict(orient="records")}


def alquileres_de_cliente(id_cliente: str) -> dict:
    """Devuelve los alquileres de un cliente. Ejemplo de id: 'C001'."""
    alquileres = pd.read_csv(carpeta / "alquileres.csv")
    alquileres_del_cliente = alquileres[alquileres["id_cliente"] == id_cliente]
    return {"alquileres": alquileres_del_cliente.to_dict(orient="records")}


def detalle_alquiler(id_alquiler: str) -> dict:
    """Muestra el detalle de un alquiler por su código. Ejemplo: 'A021'."""
    alquileres = pd.read_csv(carpeta / "alquileres.csv")
    scooters = pd.read_csv(carpeta / "scooters.csv")
    pagos = pd.read_csv(carpeta / "pagos.csv")

    alquiler = alquileres[alquileres["id_alquiler"] == id_alquiler]
    if alquiler.empty:
        return {"error": "No existe ese alquiler"}

    # El monto a pagar son las horas por la tarifa del scooter
    scooter = scooters[scooters["id_scooter"] == alquiler.iloc[0]["id_scooter"]]
    monto_a_pagar = int(alquiler.iloc[0]["horas"] * scooter.iloc[0]["tarifa_hora_soles"])

    return {
        "alquiler": alquiler.to_dict(orient="records"),
        "monto_a_pagar_soles": monto_a_pagar,
        "pagos": pagos[pagos["id_alquiler"] == id_alquiler].to_dict(orient="records"),
    }


def resumen_ingresos() -> dict:
    """Calcula el total pagado y el total pendiente de cobro."""
    pagos = pd.read_csv(carpeta / "pagos.csv")
    total_pagado = pagos[pagos["estado"] == "pagado"]["monto_soles"].sum()
    total_pendiente = pagos[pagos["estado"] == "pendiente"]["monto_soles"].sum()
    return {"total_pagado": int(total_pagado), "total_pendiente": int(total_pendiente)}


def ingresos_por_metodo() -> dict:
    """Suma lo pagado agrupado por método de pago (útil para graficar)."""
    pagos = pd.read_csv(carpeta / "pagos.csv")
    pagos_realizados = pagos[pagos["estado"] == "pagado"]
    total_por_metodo = pagos_realizados.groupby("metodo")["monto_soles"].sum()
    return {
        "etiquetas": total_por_metodo.index.tolist(),
        "valores": total_por_metodo.tolist(),
    }


def horas_por_scooter() -> dict:
    """Suma las horas alquiladas por cada scooter (útil para graficar)."""
    alquileres = pd.read_csv(carpeta / "alquileres.csv")
    total_horas = alquileres.groupby("id_scooter")["horas"].sum()
    return {
        "etiquetas": total_horas.index.tolist(),
        "valores": total_horas.tolist(),
    }


def crear_grafico(
    titulo: str, tipo: str, etiquetas: list[str], valores: list[float]
) -> dict:
    """Muestra un gráfico al usuario. Úsala cuando pidan graficar algo.

    Args:
        titulo: título del gráfico.
        tipo: 'bar' (barras), 'line' (líneas) o 'pie' (torta).
        etiquetas: nombres de cada categoría.
        valores: número de cada categoría, en el mismo orden.
    """
    return {
        "titulo": titulo,
        "tipo": tipo,
        "etiquetas": etiquetas,
        "valores": valores,
    }


def registrar_alquiler(id_cliente: str, id_scooter: str, horas: int) -> dict:
    """Registra un alquiler nuevo.
    Ejemplo: id_cliente 'C001', id_scooter 'S003', horas 2."""
    # 1. Leer la tabla
    alquileres = pd.read_csv(carpeta / "alquileres.csv")

    # 2. Armar la fila nueva (la fecha de fin es la de inicio más las horas)
    fecha_inicio = datetime.now()
    fecha_fin = fecha_inicio + timedelta(hours=horas)
    alquiler_nuevo = {
        "id_alquiler": "A0" + str(len(alquileres) + 1),   # por ejemplo A021
        "id_cliente": id_cliente,
        "id_scooter": id_scooter,
        "fecha_inicio": str(fecha_inicio)[:16],
        "fecha_fin": str(fecha_fin)[:16],
        "horas": horas,
    }

    # 3. Pegarla al final de la tabla y guardar
    alquileres = pd.concat([alquileres, pd.DataFrame([alquiler_nuevo])])
    alquileres.to_csv(carpeta / "alquileres.csv", index=False)
    return {"mensaje": "Alquiler registrado", "id_alquiler": alquiler_nuevo["id_alquiler"]}


def registrar_pago(id_alquiler: str, monto: int, metodo: str) -> dict:
    """Registra el pago de un alquiler. Métodos: Yape, Plin, Tarjeta o Efectivo.
    Ejemplo: id_alquiler 'A021', monto 12, metodo 'Yape'."""
    # 1. Leer la tabla y quitar el pago anterior de este alquiler (si había uno pendiente)
    pagos = pd.read_csv(carpeta / "pagos.csv")
    pagos = pagos[pagos["id_alquiler"] != id_alquiler]

    # 2. Armar la fila nueva
    pago_nuevo = {
        "id_pago": "P" + id_alquiler[1:],   # el pago de A021 es P021
        "id_alquiler": id_alquiler,
        "monto_soles": monto,
        "metodo": metodo,
        "fecha_pago": str(datetime.now())[:10],
        "estado": "pagado",
    }

    # 3. Pegarla al final de la tabla y guardar
    pagos = pd.concat([pagos, pd.DataFrame([pago_nuevo])])
    pagos.to_csv(carpeta / "pagos.csv", index=False)
    return {"mensaje": "Pago registrado"}


# Manual de comportamiento del agente, dividido en secciones con reglas cortas
INSTRUCCION = """
ROL
Eres un agente de una empresa de alquiler de scooters.
Respondes en español, con frases cortas, sin listas ni símbolos,
porque tu respuesta puede leerse en voz alta.

REGLAS GENERALES
1. Usa siempre las herramientas para consultar los datos.
2. No inventes información.
3. Los montos están en soles (S/).

CLIENTES
- Nunca le pidas el id al usuario: él te da el nombre.
- Con el nombre, usa buscar_cliente para obtener el id y úsalo en las demás herramientas.

ALQUILERES
- Para registrar un alquiler necesitas: cliente, scooter y horas.
  Pregunta lo que falte.
- Antes de registrar, comprueba con scooters_disponibles que el scooter esté libre.

PAGOS
- Para ver el monto a pagar o si un alquiler ya está pagado, usa detalle_alquiler.
- Métodos de pago válidos: Yape, Plin, Tarjeta y Efectivo.
  Escríbelos exactamente así.

CONFIRMACIÓN
- Antes de registrar un alquiler o un pago, repite los datos y pregunta: ¿Confirmas?
- Registra solo si el usuario responde que sí.

GRÁFICOS
- Obtén los datos con una herramienta y luego llama a crear_grafico.
"""


root_agent = Agent(
    model=modelo,  # el modelo de lenguaje
    name="agente_scooters",
    instruction=INSTRUCCION,  # el manual de arriba
    tools=[
        scooters_disponibles,
        buscar_cliente,
        alquileres_de_cliente,
        detalle_alquiler,
        resumen_ingresos,
        ingresos_por_metodo,
        horas_por_scooter,
        crear_grafico,
        registrar_alquiler,
        registrar_pago,
    ],
)
