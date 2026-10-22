# inpainting-local

Inpainting con Stable Diffusion 1.5 en local, en Docker y con GPU:

- **Interfaz** (Gradio) en <http://localhost:7860>: subes una imagen, pintas la zona y escribes el prompt.
- **API** en `POST /api/inpaint`, para que Claude Code edite imágenes con la skill de [`skill/SKILL.md`](skill/SKILL.md).

Modelo por defecto: [`Nguyentruong031203/inpainting_1.5_stable_diffusion`](https://huggingface.co/Nguyentruong031203/inpainting_1.5_stable_diffusion), un fine-tune de SD 1.5 sin licencia declarada. Para qué sirve y para qué no está en la skill. Se cambia con `MODEL_ID` en `docker-compose.yml`; por ejemplo, al oficial `stable-diffusion-v1-5/stable-diffusion-inpainting`.

## Encender y apagar

```powershell
docker compose up -d      # la primera vez construye la imagen y descarga ~4,4 GB de pesos
docker compose stop       # libera la GPU
docker compose logs -f
```

Los pesos se quedan en el volumen `inpainting-hf-cache`. La imagen usa PyTorch con CUDA 12.8, que hace falta para las RTX de la serie 50 (Blackwell).

## API

```bash
curl -s -o salida.png http://localhost:7860/api/inpaint \
  -F image=@foto.png \
  -F bbox="120,340,380,600" \
  -F prompt="a wooden chair, photorealistic"
```

| Campo | Por defecto | |
|---|---|---|
| `image` | — | Obligatorio |
| `prompt` | — | Obligatorio, en inglés |
| `mask` / `bbox` | — | Uno de los dos. `mask`: PNG con la zona a cambiar en blanco. `bbox`: `x1,y1,x2,y2` en píxeles, varias separadas por `;` |
| `negative_prompt` | `blurry, low quality, …` | |
| `steps` | 30 | |
| `guidance` | 7.5 | |
| `strength` | 1.0 | Con valores más bajos respeta más lo que había debajo |
| `seed` | -1 | Aleatoria; la usada vuelve en la cabecera `X-Seed` |
| `feather` | 4 | Difuminado del borde de la máscara, en px |
| `keep_outside` | true | Fuera de la máscara se queda la imagen original intacta |

`GET /api/health` dice si el modelo está cargado.

## Skill de Claude Code

La copia en uso está en `~/.claude/skills/inpainting-local/SKILL.md`. Si se cambia una de las dos, hay que copiarla a la otra.
