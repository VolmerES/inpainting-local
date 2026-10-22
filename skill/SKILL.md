---
name: inpainting-local
description: Modifica una zona de una imagen (quitar, cambiar o añadir algo en una parte concreta) con el Stable Diffusion 1.5 de inpainting que corre en local en Docker (contenedor inpainting-sd15, GPU). Usar cuando el usuario pida editar, retocar, borrar o reemplazar algo dentro de una imagen que tenga en disco. No sirve para generar imágenes desde cero ni para cambiar el estilo de la imagen entera.
---

# Inpainting local (SD 1.5 en Docker)

Proyecto: `C:\Users\volme\Developer-2\inpainting-local` (Gradio en `/`, API en `/api`).
Modelo: `Nguyentruong031203/inpainting_1.5_stable_diffusion`, fine-tune de SD 1.5
sin documentar ni licencia (la ruta de entrenamiento dice `insert_bbox`).
~5 s por imagen a 30 pasos, 3,7 GB de VRAM.

## Para qué vale (medido el 2026-10-22, 9 casos × 2 semillas)

| Uso | Resultado | Cómo |
|---|---|---|
| **Quitar** un objeto o persona aislado con fondo sencillo (césped, banco, pared) | ✅ Muy bueno, sin rastro | Máscara ajustada al objeto + algo de margen; prompt que describa el fondo que debe quedar |
| **Sustituir** un objeto o persona por otro (perro → gato, mujer → hombre) | ✅ Bueno: respeta luz, escala y perspectiva | Máscara con la silueta del objeto; prompt del objeto nuevo y su contexto |
| **Insertar** algo en una zona vacía | ✅/⚠️ Funciona en ~1 de cada 2 semillas | `bbox` del tamaño que debe ocupar; probar 3-4 semillas y quedarse con la buena |
| Quitar en una **zona grande** (varias personas, media imagen) | ⚠️ Tiende a rellenar con otro objeto (maceta, bolso) en vez de dejarlo vacío | Varias semillas, prompt muy explícito ("empty …, nothing on it"), o dividir en dos pasadas |
| **Caras** pequeñas | ⚠️ Pasables de lejos, flojas de cerca | Solo si la cara ocupa poco en la imagen |
| **Texto** (rótulos, carteles) | ❌ Letras sin sentido | No usarlo: decírselo al usuario |

**No sirve para**: generar desde cero, cambiar el estilo de toda la imagen,
retocar caras en primer plano, texto, ni imágenes donde la zona tenga mucho
detalle fino en alta resolución (trabaja a 512 px). En esos casos decirle al
usuario que este modelo no llega en vez de entregarle un mal resultado.

## 1. Asegurarse de que está encendido

```bash
curl -s http://localhost:7860/api/health
```

Si no responde, encenderlo y esperar a que cargue el modelo (unos 20-60 s; la
primera vez descarga ~4,4 GB):

```bash
docker compose -f C:/Users/volme/Developer-2/inpainting-local/docker-compose.yml up -d
until curl -s http://localhost:7860/api/health >/dev/null; do sleep 3; done
```

Usar `until` con el tool Monitor o en un solo Bash; no hacer polling a mano.
Al acabar, si lo encendí yo, preguntar al usuario si lo apago
(`docker compose ... stop`), porque ocupa VRAM.

## 2. Decidir la zona

- Mirar la imagen con Read y sacar su tamaño real:
  `python -c "from PIL import Image; print(Image.open(r'RUTA').size)"`.
  Read puede mostrarla reescalada: las coordenadas van **en píxeles de la
  imagen original**.
- Zona rectangular → `bbox=x1,y1,x2,y2` (varias cajas separadas por `;`).
  Dejar algo de margen alrededor del objeto (10-15 %).
- Zona irregular → generar una máscara PNG del mismo tamaño (blanco = cambiar,
  negro = dejar) con PIL (`ImageDraw.ellipse`, `polygon`…) en el scratchpad, y
  mandarla como `mask`.

## 3. Llamar a la API

```bash
curl -s -f -D - -o "SALIDA.png" http://localhost:7860/api/inpaint \
  -F image=@"ENTRADA.png" \
  -F bbox="120,340,380,600" \
  -F prompt="a wooden chair, photorealistic, natural light" \
  -F steps=30 -F seed=-1
```

Campos: `image` (obligatorio), `prompt` (obligatorio, **en inglés**), `mask` o
`bbox` (uno de los dos), `negative_prompt`, `steps` (30), `guidance` (7.5),
`strength` (1.0 = ignora lo que había debajo; 0.6-0.8 = respeta la forma y los
colores), `seed` (-1 aleatoria), `feather` (4 px de difuminado del borde),
`keep_outside` (true: fuera de la máscara se queda la original intacta).
La respuesta es el PNG; la cabecera `X-Seed` dice la semilla usada.

- **Quitar algo**: prompt que describa el fondo que debe quedar
  ("empty wooden floor", "clear blue sky"), no el objeto que se quita.
- **Nunca sobrescribir la imagen original**: escribir el resultado con otro
  nombre (`foto_editada.png`) o en el scratchpad, salvo que el usuario diga lo
  contrario.

## 4. Revisar y repetir

Ver el resultado con Read. Si no convence, cambiar semilla, ampliar la caja,
ajustar el prompt o bajar `strength`, y probar 2-3 veces antes de enseñarlo.
Decirle al usuario qué semilla y prompt se usaron.

Límites: SD 1.5 trabaja a 512 px en el lado largo (`LADO_MAX`) y el resultado
se reescala al tamaño original, así que la zona regenerada pierde detalle en
imágenes grandes. El safety checker del modelo devuelve la zona en negro si
detecta contenido NSFW.
