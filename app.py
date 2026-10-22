"""Inpainting con Stable Diffusion 1.5: interfaz Gradio en / y API HTTP en /api."""

import io
import os
import threading

import gradio as gr
import numpy as np
import torch
import uvicorn
from diffusers import StableDiffusionInpaintPipeline
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image, ImageDraw, ImageFilter
from starlette.concurrency import run_in_threadpool

MODEL_ID = os.environ.get("MODEL_ID", "Nguyentruong031203/inpainting_1.5_stable_diffusion")
LADO_MAX = int(os.environ.get("LADO_MAX", "512"))
NEGATIVO_DEFECTO = "blurry, low quality, deformed, watermark, text"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print(f"Cargando {MODEL_ID} en {device}...", flush=True)
pipe = StableDiffusionInpaintPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device)
pipe.set_progress_bar_config(disable=False)
print("Modelo listo.", flush=True)

# El pipeline no es thread-safe: la interfaz y la API se turnan la GPU
gpu = threading.Lock()


def tamano_trabajo(w, h):
    """Escala para que el lado largo sea LADO_MAX, en múltiplos de 8."""
    escala = LADO_MAX / max(w, h)
    return max(8, round(w * escala / 8) * 8), max(8, round(h * escala / 8) * 8)


def inpaint(original, mascara, prompt, negativo, pasos, guia, fuerza, semilla, difuminado, conservar_fuera):
    """Regenera la zona blanca de `mascara` en `original`. Devuelve (imagen, máscara usada, semilla, tamaño)."""
    original = original.convert("RGB")
    mascara = mascara.convert("L").resize(original.size)
    if mascara.getbbox() is None:
        raise ValueError("La máscara está vacía: marca la zona que quieres cambiar.")
    if difuminado > 0:
        mascara = mascara.filter(ImageFilter.GaussianBlur(difuminado))

    w, h = tamano_trabajo(*original.size)
    semilla = int(semilla)
    if semilla < 0:
        semilla = int(torch.randint(0, 2**31 - 1, (1,)).item())

    with gpu:
        resultado = pipe(
            prompt=prompt,
            negative_prompt=negativo or None,
            image=original.resize((w, h), Image.LANCZOS),
            mask_image=mascara.resize((w, h), Image.LANCZOS),
            width=w,
            height=h,
            num_inference_steps=int(pasos),
            guidance_scale=float(guia),
            strength=float(fuerza),
            generator=torch.Generator(device=device).manual_seed(semilla),
        ).images[0]

    resultado = resultado.resize(original.size, Image.LANCZOS)
    if conservar_fuera:
        # Fuera de la máscara se queda la imagen original, sin la pérdida del VAE
        resultado = Image.composite(resultado, original, mascara)
    return resultado, mascara, semilla, (w, h)


# --- Interfaz Gradio ---------------------------------------------------------

def mascara_de_capas(editor):
    """Une el alfa de todas las capas pintadas: blanco = zona a regenerar."""
    fondo = editor["background"]
    mascara = np.zeros((fondo.height, fondo.width), dtype=bool)
    for capa in editor.get("layers") or []:
        if capa is None:
            continue
        alfa = np.array(capa.convert("RGBA").resize(fondo.size))[:, :, 3]
        mascara |= alfa > 0
    return Image.fromarray((mascara * 255).astype(np.uint8), "L")


def generar(editor, prompt, negativo, pasos, guia, fuerza, semilla, difuminado, conservar_fuera):
    if editor is None or editor.get("background") is None:
        raise gr.Error("Sube una imagen primero.")
    if not prompt.strip():
        raise gr.Error("Escribe un prompt.")
    mascara = mascara_de_capas(editor)
    if mascara.getbbox() is None:
        raise gr.Error("Pinta con el pincel la zona que quieres cambiar.")

    resultado, mascara, semilla, (w, h) = inpaint(
        editor["background"], mascara, prompt, negativo, pasos, guia, fuerza, semilla, difuminado, conservar_fuera
    )
    return resultado, mascara, f"Semilla: {semilla} · trabajo a {w}×{h}"


with gr.Blocks(title="Inpainting SD 1.5") as demo:
    gr.Markdown(
        "## Inpainting SD 1.5\n"
        "Sube una imagen, **pinta en blanco la zona a cambiar** y describe en el prompt "
        "lo que quieres que aparezca ahí (mejor en inglés)."
    )
    with gr.Row():
        with gr.Column():
            editor = gr.ImageEditor(
                label="Imagen y máscara",
                type="pil",
                sources=["upload", "clipboard"],
                brush=gr.Brush(default_size=40, colors=["#FFFFFF"], color_mode="fixed"),
                layers=False,
                height=520,
            )
            prompt = gr.Textbox(label="Prompt", lines=2, placeholder="a red ceramic vase with flowers")
            negativo = gr.Textbox(label="Prompt negativo", lines=1, value=NEGATIVO_DEFECTO)
            with gr.Accordion("Ajustes", open=False):
                pasos = gr.Slider(10, 80, value=30, step=1, label="Pasos")
                guia = gr.Slider(1, 15, value=7.5, step=0.5, label="Guidance scale")
                fuerza = gr.Slider(0.1, 1.0, value=1.0, step=0.05, label="Fuerza (1 = ignora lo que había debajo)")
                semilla = gr.Number(value=-1, precision=0, label="Semilla (-1 = aleatoria)")
                difuminado = gr.Slider(0, 30, value=4, step=1, label="Difuminado del borde de la máscara (px)")
                conservar_fuera = gr.Checkbox(value=True, label="Conservar intacta la imagen fuera de la máscara")
            boton = gr.Button("Generar", variant="primary")
        with gr.Column():
            salida = gr.Image(label="Resultado", type="pil", format="png")
            info = gr.Markdown()
            mascara_vista = gr.Image(label="Máscara usada", type="pil", height=200)

    boton.click(
        generar,
        inputs=[editor, prompt, negativo, pasos, guia, fuerza, semilla, difuminado, conservar_fuera],
        outputs=[salida, mascara_vista, info],
    )


# --- API HTTP (la usa Claude Code con curl) ----------------------------------

api = FastAPI()


@api.get("/api/health")
def health():
    return {"ok": True, "model": MODEL_ID, "device": device, "lado_max": LADO_MAX}


def mascara_de_cajas(texto, tamano):
    """'x1,y1,x2,y2;x1,y1,x2,y2' en píxeles de la imagen original → máscara blanca en esas cajas."""
    mascara = Image.new("L", tamano, 0)
    dibujo = ImageDraw.Draw(mascara)
    for trozo in filter(None, (t.strip() for t in texto.split(";"))):
        try:
            x1, y1, x2, y2 = (float(v) for v in trozo.split(","))
        except ValueError:
            raise HTTPException(400, f"bbox mal formada: '{trozo}' (se espera x1,y1,x2,y2)")
        dibujo.rectangle([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)], fill=255)
    return mascara


@api.post("/api/inpaint")
async def api_inpaint(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    mask: UploadFile | None = File(None),
    bbox: str = Form(""),
    negative_prompt: str = Form(NEGATIVO_DEFECTO),
    steps: int = Form(30),
    guidance: float = Form(7.5),
    strength: float = Form(1.0),
    seed: int = Form(-1),
    feather: int = Form(4),
    keep_outside: bool = Form(True),
):
    try:
        original = Image.open(io.BytesIO(await image.read()))
        original.load()
    except Exception:
        raise HTTPException(400, "No se puede leer 'image'.")

    if mask is not None and mask.filename:
        mascara = Image.open(io.BytesIO(await mask.read())).convert("L")
    elif bbox.strip():
        mascara = mascara_de_cajas(bbox, original.size)
    else:
        raise HTTPException(400, "Falta la zona: manda 'mask' (PNG, blanco = cambiar) o 'bbox' (x1,y1,x2,y2).")

    try:
        resultado, _, semilla, (w, h) = await run_in_threadpool(
            inpaint, original, mascara, prompt, negative_prompt, steps, guidance, strength, seed, feather, keep_outside
        )
    except ValueError as e:
        raise HTTPException(400, str(e))

    buffer = io.BytesIO()
    resultado.save(buffer, format="PNG")
    return Response(buffer.getvalue(), media_type="image/png", headers={"X-Seed": str(semilla), "X-Work-Size": f"{w}x{h}"})


demo.queue(max_size=8)
app = gr.mount_gradio_app(api, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
