# Guia de Usuario - Synthetic Dataset Generator

## Tabla de Contenidos

1. [Introduccion](#1-introduccion)
2. [Requisitos del Sistema](#2-requisitos-del-sistema)
3. [Arranque de la Aplicacion](#3-arranque-de-la-aplicacion)
4. [Dashboard](#4-dashboard)
5. [Gestor de Dominios](#5-gestor-de-dominios)
6. [Flujo de Trabajo Principal (Workflow)](#6-flujo-de-trabajo-principal-workflow)
   - [Paso 1: Analisis del Dataset](#paso-1-analisis-del-dataset)
   - [Paso 2: Configuracion del Pipeline](#paso-2-configuracion-del-pipeline)
   - [Paso 3: Seleccion de Fuentes](#paso-3-seleccion-de-fuentes)
   - [Paso 4: Generacion](#paso-4-generacion)
   - [Paso 4.5: Validacion Domain Gap](#paso-45-validacion-domain-gap-opcional)
   - [Paso 5: Exportacion](#paso-5-exportacion)
   - [Paso 6: Combinar Datasets](#paso-6-combinar-datasets)
   - [Paso 7: Division Train/Val/Test](#paso-7-division-trainvaltest)
7. [Herramientas](#7-herramientas)
   - [Monitor de Jobs](#71-monitor-de-jobs)
   - [Estado de Servicios](#72-estado-de-servicios)
   - [Segmentacion SAM](#73-segmentacion-sam)
   - [Extraccion de Objetos](#74-extraccion-de-objetos)
   - [Auto-Etiquetado](#75-auto-etiquetado)
   - [Gestor de Etiquetas](#76-gestor-de-etiquetas)
   - [Tamanos de Objetos](#77-tamanos-de-objetos)
   - [Post-Procesamiento y Balanceo](#78-post-procesamiento-y-balanceo-de-clases)
   - [Domain Gap](#79-domain-gap)
8. [Flujos de Trabajo Recomendados](#8-flujos-de-trabajo-recomendados)

---

## 1. Introduccion

El **Synthetic Dataset Generator** es una aplicacion completa para crear, gestionar y aumentar datasets sinteticos para tareas de vision por computador y machine learning. Permite generar imagenes sinteticas con anotaciones COCO a partir de objetos extraidos e imagenes de fondo, aplicando efectos realistas para minimizar el domain gap entre datos sinteticos y reales.

### Arquitectura

La aplicacion utiliza una arquitectura de microservicios:

| Servicio | Puerto | Funcion |
|----------|--------|---------|
| **Gateway** | 8000 | Router principal, orquestacion de servicios |
| **Depth** | 8001 | Estimacion de profundidad para colocacion realista |
| **Segmentation** | 8002 | Segmentacion con SAM3 (Segment Anything Model) |
| **Effects** | 8003 | Efectos visuales, correccion de color, iluminacion |
| **Augmentor** | 8004 | Pipeline principal de generacion sintetica |
| **Domain Gap** | 8005 | Analisis de domain gap, metricas FID/KID |

El frontend es una aplicacion Vue 3 con TypeScript que se conecta al Gateway.

---

## 2. Requisitos del Sistema

- **Docker** y **Docker Compose** instalados
- **GPU NVIDIA** con drivers CUDA (recomendado, necesario para SAM3, Depth, Domain Gap)
- Al menos **16 GB de RAM** (32 GB recomendado)
- Al menos **8 GB de VRAM** en GPU (para todos los servicios GPU)
- Navegador web moderno (Chrome, Firefox, Edge)

---

## 3. Arranque de la Aplicacion

### Iniciar todos los servicios

```bash
docker compose -f docker-compose.microservices.yml up -d
```

### Verificar que los servicios estan corriendo

```bash
docker compose -f docker-compose.microservices.yml ps
```

### Acceder a la interfaz web

Abre tu navegador y navega a: **http://localhost:5173**

### Detener los servicios

```bash
docker compose -f docker-compose.microservices.yml down
```

---

## 4. Dashboard

Al abrir la aplicacion, veras el **Dashboard** con una vista general del sistema:

- Metricas principales del proyecto
- Estado rapido de los servicios
- Acceso directo a las secciones principales

Desde el sidebar izquierdo puedes navegar a cualquier seccion de la aplicacion.

---

## 5. Gestor de Dominios

**Ruta:** Sidebar > Domain Manager

El gestor de dominios permite configurar perfiles especificos para diferentes entornos de generacion (underwater, urbano, aereo, etc.).

### 5.1 Ver Dominios Disponibles

1. Haz clic en **Domain Manager** en el sidebar
2. Veras los dominios disponibles como tarjetas con:
   - Nombre e icono del dominio
   - Numero de regiones, objetos, efectos y presets configurados
3. Haz clic en un dominio para ver su detalle

### 5.2 Crear un Dominio Personalizado

1. Haz clic en el boton **Create Domain**
2. Completa los campos:
   - **ID**: Identificador unico (ej: `industrial_floor`)
   - **Name**: Nombre visible (ej: "Industrial Floor")
   - **Description**: Descripcion del dominio
   - **Icon**: Icono del dominio
3. Guarda el dominio

### 5.3 Importar/Exportar Dominios

- **Exportar**: En la vista de detalle del dominio, haz clic en **Export** para descargar el JSON
- **Importar**: En el listado de dominios, haz clic en **Import Domain** y pega el JSON

### 5.4 Detalle del Dominio

Al abrir un dominio, veras varias pestanas:

#### Regiones
- Lista de regiones de la escena (ej: fondo marino, superficie, columna de agua)
- Cada region tiene un color, prompts SAM3 para deteccion automatica, y nombre visible

#### Objetos
- Tipos de objetos con tamanos reales, densidad, propiedades fisicas
- Palabras clave para deteccion automatica

#### Efectos
- Efectos especificos del dominio (ej: causticas para underwater)
- Parametros configurables en formato JSON

#### Fisica
- Tipo de fisica (underwater, aerea, terrestre)
- Densidad del medio, direccion de gravedad
- Umbrales de flotacion/hundimiento

#### Matriz de Compatibilidad
- Mapa de calor que muestra que objetos son compatibles con que regiones
- Valores de 0% (incompatible) a 100% (totalmente compatible)
- Codificado por colores: verde (alto), amarillo (medio), naranja (bajo), gris (ninguno)

### 5.5 Activar un Dominio

1. En el listado de dominios, haz clic en **Activate** en el dominio deseado
2. El dominio activo se usara como configuracion base en el workflow de generacion

---

## 6. Flujo de Trabajo Principal (Workflow)

El workflow principal consta de 7 pasos secuenciales. Cada paso debe completarse antes de avanzar al siguiente. Puedes volver a pasos anteriores en cualquier momento.

### Paso 1: Analisis del Dataset

**Ruta:** Sidebar > Workflow > 1. Analysis

**Objetivo:** Cargar y analizar un dataset COCO existente que servira como base para la generacion sintetica.

#### Instrucciones:

1. **Opcion A - Subir archivo JSON:**
   - Arrastra un archivo `.json` en formato COCO al area de carga
   - O haz clic en el area para seleccionar el archivo

2. **Opcion B - Seleccionar dataset existente:**
   - Usa el selector de archivos para navegar a un dataset COCO previamente generado

3. **Revisar resultados del analisis:**
   - **Total de imagenes**: Numero de imagenes en el dataset
   - **Total de anotaciones**: Numero de bounding boxes/segmentaciones
   - **Categorias**: Numero de clases de objetos
   - **Media de anotaciones por imagen**: Distribucion promedio
   - **Distribucion por categoria**: Grafico de barras mostrando cuantas anotaciones tiene cada clase

4. Haz clic en **Next** para avanzar al paso 2

> **Nota:** Si no tienes un dataset COCO, puedes crear uno usando la herramienta de **Auto-Etiquetado** (seccion 7.5).

---

### Paso 2: Configuracion del Pipeline

**Ruta:** Sidebar > Workflow > 2. Configure

**Objetivo:** Configurar todos los efectos de aumentacion y parametros de colocacion de objetos.

#### 2.1 Presets Rapidos

En la parte superior, selecciona un preset predefinido para cargar una configuracion optimizada:

| Preset | Uso |
|--------|-----|
| **Underwater Marine** | Datasets marinos con causticas, tinte de agua, correccion de color |
| **Urban Street** | Escenas urbanas, trafico, senales |
| **Minimal** | Augmentacion ligera para variaciones sutiles |
| **Aggressive** | Augmentacion intensa para maxima variedad |

#### 2.2 Pestana: Basic Augmentation

Configura efectos de augmentacion basicos. Cada efecto tiene un switch para activar/desactivar:

| Efecto | Parametros | Descripcion |
|--------|------------|-------------|
| **Blur** | Radio min/max (0-10px) | Desenfoque gaussiano |
| **Noise** | Intensidad min/max (0-0.2) | Ruido aleatorio |
| **Brightness** | Factor min/max (0.5-1.5) | Ajuste de brillo |
| **Contrast** | Factor min/max (0.5-1.5) | Ajuste de contraste |
| **Rotation** | Angulo min/max (-180 a 180) | Rotacion aleatoria |
| **Scale** | Factor min/max (0.5-2.0) | Escalado aleatorio |
| **Horizontal Flip** | Probabilidad (0-1) | Volteo horizontal |
| **Vertical Flip** | Probabilidad (0-1) | Volteo vertical |

#### 2.3 Pestana: Underwater & Realism

Efectos avanzados de realismo:

| Efecto | Parametros | Descripcion |
|--------|------------|-------------|
| **Color Correction** | Intensidad (0-1) | Adapta colores del objeto al fondo |
| **Blur Matching** | Fuerza (0-1) | Iguala el nivel de desenfoque con el fondo |
| **Lighting** | Tipo (ambient/spot/gradient), intensidad | Efectos de iluminacion |
| **Underwater Tint** | Intensidad, color RGB, claridad del agua | Tinte de agua para dominios marinos |
| **Drop Shadows** | Opacidad (0-1), desenfoque (0-100px) | Sombras realistas |
| **Caustics** | Intensidad (0-1), deterministico toggle | Patrones de refraccion de luz |
| **Motion Blur** | Probabilidad (0-1), kernel (3-31px) | Simulacion de movimiento |
| **Perspective Warp** | Magnitud (0-0.3) | Distorsion de perspectiva |

#### 2.4 Pestana: Blending & Edges

| Parametro | Opciones | Descripcion |
|-----------|----------|-------------|
| **Blending Method** | Alpha / Poisson / Laplacian | Metodo de fusion de objeto con fondo |
| **Binary Alpha** | On/Off | Canal alpha binario |
| **Alpha Feather Radius** | 0-10px | Suavizado del borde alpha |
| **Edge Smoothing** | 0-20px | Suavizado de bordes del objeto |
| **Max Blur Budget** | 0-50 | Presupuesto maximo de desenfoque acumulado |

#### 2.5 Pestana: Object Placement

| Parametro | Rango | Descripcion |
|-----------|-------|-------------|
| **Objects per Image** | Min 1, Max 20 | Rango de objetos por imagen |
| **Min Size Ratio** | 1-20% | Tamano minimo relativo a la imagen |
| **Absolute Min Size** | 5-50px | Tamano minimo en pixeles |
| **Min Area Ratio** | 0.1-10% | Area minima relativa a la imagen |
| **Max Area Ratio** | 10-80% | Area maxima relativa a la imagen |
| **Overlap Threshold** | 0-50% | Solapamiento maximo permitido entre objetos |

5. Cuando estes satisfecho con la configuracion, haz clic en **Next**

---

### Paso 3: Seleccion de Fuentes

**Ruta:** Sidebar > Workflow > 3. Sources

**Objetivo:** Seleccionar las imagenes de fondo y el directorio de salida.

#### 3.1 Directorio de Fondos

1. Haz clic en **Browse** junto a "Background Images Directory"
2. Navega por el explorador de directorios hasta encontrar tu carpeta de fondos
3. Selecciona la carpeta que contiene las imagenes de fondo (JPG, PNG, JPEG)
4. **(Opcional)** Haz clic en **Preview** para ver una cuadricula de las imagenes de fondo
   - Navega entre paginas si hay muchas imagenes (8 por pagina)
   - Verifica que las imagenes son adecuadas

> **Nota:** Si no seleccionas un directorio de fondos, se usaran fondos de color solido.

#### 3.2 Directorio de Salida

1. Haz clic en **Browse** junto a "Output Directory"
2. Selecciona o crea la carpeta donde se guardara el dataset generado

3. Haz clic en **Next** para avanzar

---

### Paso 4: Generacion

**Ruta:** Sidebar > Workflow > 4. Generation

**Objetivo:** Configurar objetivos de generacion y ejecutar el job de generacion sintetica.

#### 4.1 Pestana: Targets (Objetivos)

1. Para cada categoria del dataset, ajusta el numero de imagenes a generar:
   - Usa el **slider** individual por categoria (0-500)
   - O usa los botones rapidos de cantidad: **50**, **100**, **200**, **500**
2. Observa el **contador total** de imagenes en la parte superior

#### 4.2 Pestana: Generation Options

| Opcion | Descripcion |
|--------|-------------|
| **Use Depth Estimation** | Activa estimacion de profundidad para colocacion realista en la escena |
| **Use Segmentation** | Activa SAM3 para segmentacion precisa |
| **Depth-Aware Placement** | Coloca objetos segun capas de profundidad (primer plano, medio, fondo) |

#### 4.3 Pestana: Validation

Configura validaciones de calidad para las imagenes generadas:

- **Identity Validation**: Verifica que el objeto mantiene su identidad visual
  - Max Color Shift: Desviacion maxima de color (10-100)
  - Min Sharpness Ratio: Nitidez minima (0-1)
  - Min Contrast Ratio: Contraste minimo (0-1)
- **Quality Validation**: Verifica calidad perceptual
  - Min Perceptual Quality (0-1)
  - Min Anomaly Score (0-1)
- **Physics Validation**: Valida coherencia fisica
- **Reject Invalid**: Rechaza automaticamente imagenes que no pasen validacion

#### 4.4 Pestana: Processing

| Parametro | Descripcion |
|-----------|-------------|
| **Parallel Processing** | Activa procesamiento concurrente |
| **Concurrent Limit** | Numero de jobs simultaneos (1-16) |
| **VRAM Threshold** | Umbral de memoria GPU (30-95%) |
| **Debug Mode** | Guarda informacion de depuracion del pipeline |

#### 4.5 Pestana: Lighting

| Parametro | Descripcion |
|-----------|-------------|
| **Max Light Sources** | Numero maximo de fuentes de luz detectadas (1-10) |
| **Intensity Threshold** | Umbral de intensidad para deteccion (0.1-1) |
| **Estimate HDR** | Estimacion de rango dinamico alto |
| **Water Attenuation** | Atenuacion por profundidad del agua |
| **Depth Category** | Shallow / Mid / Deep |

#### 4.6 Pestana: Metadata

Informacion del dataset exportado:

- **Name**: Nombre del dataset
- **Version**: Version (ej: "1.0")
- **Description**: Descripcion del dataset
- **Contributor**: Autor/organizacion
- **License**: Nombre y URL de la licencia

#### 4.7 Iniciar Generacion

1. Haz clic en **Generate** para iniciar el job
2. Observa el progreso en tiempo real:
   - **Barra de progreso** con porcentaje
   - **Imagenes generadas** / total
   - **Anotaciones creadas**
   - **Duracion** en segundos
3. Puedes **cancelar** el job en cualquier momento
4. Al finalizar, veras un resumen con las metricas de calidad (si validacion estaba activa)

---

### Paso 4.5: Validacion Domain Gap (Opcional)

**Aparece automaticamente** al completar la generacion (paso 4).

**Objetivo:** Evaluar si las imagenes sinteticas generadas son visualmente similares a imagenes reales de referencia.

#### Instrucciones:

1. El panel de **Domain Gap Validation** aparecera colapsado tras completar la generacion
2. Haz clic para expandirlo
3. Si no tienes sets de referencia, ve a **Tools > Domain Gap** para subir imagenes reales primero
4. **Selecciona un Reference Set** del dropdown
5. Haz clic en **Analyze**
6. Revisa los resultados:

| Metrica | Descripcion |
|---------|-------------|
| **Gap Score** | Puntuacion global 0-100 (0 = identico, 100 = maximo gap) |
| **FID Score** | Frechet Inception Distance (menor = mejor) |
| **KID Score** | Kernel Inception Distance (menor = mejor) |
| **Gap Level** | Low / Medium / High / Critical |

7. Si el gap es **bajo** (verde, <30): Tus datos sinteticos son adecuados. Continua a Export.
8. Si el gap es **moderado** (amarillo, 30-60): Considera aplicar las sugerencias.
9. Si el gap es **alto** (rojo, >60): Se recomienda ajustar parametros y regenerar.

#### Aplicar Sugerencias

Si hay sugerencias de parametros:
1. Revisa cada sugerencia (parametro, valor actual vs sugerido, razon)
2. Haz clic en **Apply All** para aplicar automaticamente todas las sugerencias
3. Seras redirigido a la pantalla de Configure con los parametros actualizados
4. Regenera el dataset con los nuevos parametros

#### Saltar

Haz clic en **Skip** si no deseas validar el domain gap y quieres continuar directamente a Export.

---

### Paso 5: Exportacion

**Ruta:** Sidebar > Workflow > 5. Export

**Objetivo:** Exportar el dataset generado al formato requerido por tu framework de ML.

#### Instrucciones:

1. **Selecciona el dataset**: Elige el archivo COCO JSON a exportar
2. **Selecciona el directorio de imagenes**: Carpeta con las imagenes generadas
3. **Selecciona el directorio de salida**: Donde guardar el dataset exportado
4. **Elige el formato de exportacion:**

| Formato | Descripcion | Archivos generados |
|---------|-------------|-------------------|
| **YOLO (Ultralytics)** | Coordenadas normalizadas en TXT | `.txt` por imagen + `data.yaml` |
| **Pascal VOC (XML)** | Anotaciones XML por imagen | `.xml` por imagen |
| **COCO JSON** | Formato COCO estandar | `annotations.json` |

5. **(Opcional)** Marca **Include Images** para copiar las imagenes junto a las anotaciones
6. Haz clic en **Export**
7. Verifica el resumen:
   - Imagenes exportadas
   - Anotaciones exportadas
   - Ruta de salida

---

### Paso 6: Combinar Datasets

**Ruta:** Sidebar > Workflow > 6. Combine

**Objetivo:** Fusionar multiples datasets en uno solo.

#### Instrucciones:

1. Veras una lista de datasets disponibles
2. **Selecciona 2 o mas datasets** marcando las casillas de verificacion
3. Selecciona el **directorio de salida** para el dataset combinado
4. Configura opciones:
   - **Merge Categories**: Fusiona categorias con el mismo nombre (recomendado)
   - **Deduplicate Images**: Elimina imagenes duplicadas por nombre de archivo
5. Haz clic en **Combine**
6. Revisa el resumen: total de imagenes, anotaciones y categorias del dataset combinado

> **Caso de uso tipico:** Combinar varias rondas de generacion sintetica o mezclar datos sinteticos con datos reales.

---

### Paso 7: Division Train/Val/Test

**Ruta:** Sidebar > Workflow > 7. Splits

**Objetivo:** Dividir el dataset final en conjuntos de entrenamiento, validacion y test.

#### Modo 1: Division por Ratios

1. Selecciona el modo **Train/Val/Test Split**
2. Ajusta los **porcentajes** con los sliders:
   - **Train**: Porcentaje para entrenamiento (tipico: 70-80%)
   - **Validation**: Porcentaje para validacion (tipico: 10-15%)
   - **Test**: Porcentaje para test (tipico: 10-15%)
3. La barra visual muestra la proporcion de cada split en colores
4. Los tres valores deben sumar **100%**
5. Marca **Stratified Split** para mantener la distribucion de clases en cada split (recomendado)
6. Opcionalmente ajusta la **semilla aleatoria** (default: 42) para reproducibilidad
7. Selecciona el **directorio de salida**
8. Haz clic en **Split**

#### Modo 2: K-Fold Cross Validation

1. Selecciona el modo **K-Fold**
2. Configura el **numero de folds K** (2-10)
3. Selecciona el **fold de validacion** haciendo clic en el boton correspondiente
4. Marca **Stratified** si deseas mantener la distribucion de clases
5. Haz clic en **Split**
6. Se generaran K archivos JSON (fold_1.json, fold_2.json, ...) en el directorio de salida

---

## 7. Herramientas

### 7.1 Monitor de Jobs

**Ruta:** Sidebar > Tools > Job Monitor

Monitorea y gestiona todos los jobs en ejecucion y completados.

#### Funcionalidades:

1. **Filtrar jobs** por:
   - **Estado**: All / Pending / Running / Completed / Failed
   - **Fuente**: All / Generation / Labeling / Extraction / SAM3

2. **Ver detalles de un job**: Haz clic en un job para ver:
   - ID del job
   - Tipo y fuente
   - Estado con barra de progreso (para jobs en ejecucion)
   - Mensajes de error (si fallo)
   - Logs del job

3. **Acciones disponibles**:
   - **Cancel**: Detener un job en ejecucion
   - **Resume**: Reanudar un job interrumpido (solo augmentor)
   - **Retry**: Reintentar un job fallido (solo augmentor)
   - **Regenerate**: Regenerar con los mismos parametros (solo completados)
   - **Delete**: Eliminar un job (solo no activos)

> La lista se actualiza automaticamente cada 5 segundos.

---

### 7.2 Estado de Servicios

**Ruta:** Sidebar > Tools > Service Status

Muestra el estado de salud de todos los microservicios del backend.

#### Informacion por servicio:

| Campo | Descripcion |
|-------|-------------|
| **Nombre** | Identificador del servicio |
| **Puerto** | Puerto de red |
| **Estado** | Healthy (verde) / Degraded (amarillo) / Unhealthy (rojo) |
| **Latencia** | Tiempo de respuesta en milisegundos |
| **URL** | Direccion del servicio |

- El banner superior muestra el estado general del sistema
- Haz clic en **Refresh** para actualizar manualmente (auto-refresco cada 30s)

---

### 7.3 Segmentacion SAM

**Ruta:** Sidebar > Tools > SAM Segmentation

Segmenta objetos en imagenes usando SAM3 (Segment Anything Model).

#### Modo 1: Segmentacion por Texto

1. Selecciona una **imagen** usando el explorador de archivos
2. Escribe un **prompt de texto** describiendo el objeto a segmentar (ej: "fish", "car", "bottle")
3. Haz clic en **Segment**
4. Resultados:
   - Numero de objetos encontrados
   - Puntuacion de confianza por cada deteccion (0-100%)
   - Barras de progreso visuales

#### Modo 2: Conversion de Dataset

1. Selecciona el **COCO JSON** de origen
2. Selecciona el **directorio de imagenes**
3. Selecciona el **directorio de salida**
4. Configura parametros:
   - **Min Area Threshold**: Area minima de objetos (10-1000px)
   - **Confidence Threshold**: Confianza minima (10-100%)
5. Haz clic en **Convert**
6. Monitorea el progreso del job

---

### 7.4 Extraccion de Objetos

**Ruta:** Sidebar > Tools > Object Extraction

Extrae objetos individuales de un dataset para usarlos como fuente en la generacion sintetica.

#### Instrucciones:

1. Selecciona la ruta al **COCO JSON** del dataset fuente
2. Selecciona el **directorio de imagenes** del dataset
3. Selecciona el **directorio de salida** donde guardar los recortes
4. Configura parametros:

| Parametro | Descripcion |
|-----------|-------------|
| **Min Object Size** | Tamano minimo en pixeles (8-256px) |
| **Include Masks** | Guardar mascaras de segmentacion junto a los recortes |
| **Use SAM3** | Usar SAM3 para generar mascaras precisas (mas lento pero mejor calidad) |
| **Deduplicate** | Omitir objetos visualmente similares |
| **Padding** | Pixeles extra alrededor del bounding box (0-50px) |

5. Haz clic en **Extract**
6. Monitorea el progreso en el Job Monitor

**Resultado:** Una carpeta con imagenes recortadas de cada objeto, organizadas por categoria, listas para usar en la generacion sintetica.

---

### 7.5 Auto-Etiquetado

**Ruta:** Sidebar > Tools > Auto Labeling

Etiqueta automaticamente imagenes usando GroundingDINO y SAM3.

#### Modo 1: Nuevo Etiquetado

1. **(Opcional)** Selecciona una **plantilla rapida** para cargar clases predefinidas:
   - Marine Life, Vehicles, People, Urban, Nature, Food, Fashion

2. Agrega **directorios de imagenes** (puedes anadir multiples):
   - Haz clic en **Add Directory**
   - Navega y selecciona cada carpeta

3. Agrega las **clases a detectar**:
   - Escribe el nombre de cada clase (ej: "fish", "car")
   - Haz clic en **Add** o presiona Enter
   - O usa una plantilla para cargar multiples clases de golpe

4. Selecciona el **directorio de salida** para las anotaciones

5. Configura parametros:

| Parametro | Descripcion |
|-----------|-------------|
| **Min Confidence** | Umbral de deteccion (10-90%) |
| **Task Type** | Detection (boxes) / Segmentation (masks) / Both |
| **Use SAM3** | Activar para mascaras de alta calidad |
| **Box Threshold** | Umbral de deteccion de cajas (0.1-0.9) |
| **Text Threshold** | Umbral de coincidencia de texto (0.1-0.9) |

6. Haz clic en **Start Labeling**

7. Monitorea en tiempo real:
   - Progreso en porcentaje
   - Confianza media de las detecciones
   - Imagenes con/sin detecciones
   - Previsualizacion de las ultimas 5 imagenes procesadas

#### Modo 2: Re-etiquetado

1. Selecciona los **directorios de imagenes**
2. Selecciona el modo:
   - **Add**: Anadir nuevas anotaciones a las existentes
   - **Replace**: Reemplazar todas las anotaciones
   - **Improve Segmentation**: Mejorar las mascaras existentes con SAM3
3. Carga las **anotaciones existentes** (COCO JSON)
4. Opcionalmente agrega nuevas clases
5. Haz clic en **Start Relabeling**

**Resultado:** Un archivo COCO JSON con todas las anotaciones generadas automaticamente.

> **Tip:** Despues de auto-etiquetar, usa la herramienta de **Extraccion de Objetos** para recortar los objetos detectados y usarlos en la generacion sintetica.

---

### 7.6 Gestor de Etiquetas

**Ruta:** Sidebar > Tools > Label Manager

Visualiza y gestiona las etiquetas/categorias de tus datasets.

#### Instrucciones:

1. Selecciona un **dataset** del dropdown
2. Veras la lista de categorias con el numero de anotaciones de cada una
3. **Acciones disponibles**:
   - **Renombrar**: Haz clic en el icono de edicion, escribe el nuevo nombre, guarda
   - **Eliminar**: Haz clic en el icono de basura, confirma (elimina la categoria y TODAS sus anotaciones)
   - **Refrescar**: Recarga la lista de categorias

---

### 7.7 Tamanos de Objetos

**Ruta:** Sidebar > Tools > Object Sizes

Configura los tamanos relativos de los objetos y analiza dimensiones del dataset.

#### Seccion 1: Configuracion de Tamanos

1. **(Opcional)** Carga un **preset rapido**:
   - Marine Life, Vehicles, Common Objects
2. Agrega clases manualmente:
   - Escribe el nombre de la clase
   - Ajusta el slider de tamano (1-100% del tamano de la imagen)
3. Ajusta tamanos individuales con los sliders
4. Haz clic en **Save All** para guardar

#### Seccion 2: Analisis del Dataset

1. Selecciona un dataset para analizar
2. Resultados:
   - **Dimensiones de imagenes**: Minimo, promedio y maximo de ancho y alto
   - **Metricas generales**: Total imagenes, anotaciones, categorias
   - **Distribucion por categoria**: Grafico de barras con anotaciones por clase

---

### 7.8 Post-Procesamiento y Balanceo de Clases

**Ruta:** Sidebar > Tools > Post-Processing

Analiza el desbalanceo de clases y aplica estrategias de balanceo.

#### Analisis de Distribucion

1. Selecciona un dataset
2. Veras un grafico de barras codificado por colores:
   - **Verde**: Clase equilibrada (cerca de la media)
   - **Amarillo**: Sub-representada (< 80% de la media)
   - **Rojo**: Severamente sub-representada (< 50% de la media)
   - **Azul**: Sobre-representada (> 200% de la media)
3. Estadisticas:
   - Total de anotaciones
   - Conteo maximo y minimo por clase
   - Ratio de desbalanceo (max/min)

#### Metodos de Balanceo

**Metodo 1: Class Weights (Recomendado)**
- Calcula pesos inversos a la frecuencia
- Copia como JSON o diccionario Python
- Usalo durante el entrenamiento sin modificar el dataset
- Las clases minoritarias reciben pesos mas altos

**Metodo 2: Undersampling**
- Reduce las clases mayoritarias al nivel de la minoritaria
- Crea un dataset equilibrado con menos muestras
- Selecciona directorio de salida y ejecuta

**Metodo 3: Oversampling**
- Duplica las clases minoritarias al nivel de la mayoritaria
- Crea un dataset equilibrado con mas muestras
- Riesgo de overfitting por duplicacion

---

### 7.9 Domain Gap

**Ruta:** Sidebar > Tools > Domain Gap

Mide y reduce el domain gap entre imagenes sinteticas y reales.

#### Pestana 1: Reference Sets (Imagenes de Referencia)

**Objetivo:** Subir y gestionar conjuntos de imagenes reales contra las cuales comparar las sinteticas.

1. Haz clic en **Upload Reference Set**
2. Completa:
   - **Nombre**: Nombre descriptivo (ej: "Fotos reales fondo marino")
   - **Descripcion**: Descripcion del set
   - **Dominio**: Selecciona el dominio asociado
3. Selecciona multiples imagenes reales representativas
4. Haz clic en **Upload**
5. El sistema calculara automaticamente estadisticas (brillo, varianza de bordes, etc.)

> **Recomendacion:** Sube al menos 20-50 imagenes reales para obtener metricas fiables.

#### Pestana 2: Validation (Validacion)

**Objetivo:** Medir cuanto se parecen las sinteticas a las reales.

1. Selecciona el **directorio de imagenes sinteticas** a evaluar
2. Selecciona el **Reference Set** de imagenes reales
3. Ajusta **Max Images** a analizar (5-500, mas imagenes = mas preciso pero mas lento)
4. Haz clic en **Analyze**

**Resultados:**

| Metrica | Rango | Interpretacion |
|---------|-------|---------------|
| **Gap Score** | 0-100 | 0 = identico a reales, 100 = totalmente diferente |
| **FID** | 0-inf | Frechet Inception Distance, menor es mejor |
| **KID** | 0-inf | Kernel Inception Distance, menor es mejor, mas robusto con pocas muestras |
| **Gap Level** | Low/Med/High/Critical | Clasificacion cualitativa |

**Semaforo de Gap Score:**
- **< 30 (Verde)**: Gap bajo. Las sinteticas son adecuadas para entrenamiento.
- **30-60 (Amarillo)**: Gap moderado. Considera aplicar sugerencias.
- **> 60 (Rojo)**: Gap alto. Ajusta parametros y regenera.

**Issues detectados:**
- Lista de problemas especificos (color, textura, bordes, iluminacion)
- Severidad: high/medium/low
- Valores metricos vs esperados

**Sugerencias de parametros:**
- El sistema genera sugerencias automaticas de que parametros del pipeline ajustar
- Cada sugerencia indica: parametro, valor actual, valor sugerido, razon e impacto esperado
- Haz clic en **Apply All** para aplicar todas las sugerencias al pipeline

#### Pestana 3: Randomization (Domain Randomization)

**Objetivo:** Aplicar variaciones aleatorias a las sinteticas para aumentar diversidad y reducir gap.

1. Selecciona el **directorio de imagenes** a procesar
2. Selecciona el **directorio de salida**
3. Configura parametros:

| Parametro | Rango | Descripcion |
|-----------|-------|-------------|
| **Variants per Image** | 1-10 | Cuantas variantes generar por imagen |
| **Intensity** | 0-1 | Intensidad general de la randomizacion |
| **Color Jitter** | 0-1 | Variacion de color en espacio LAB |
| **Noise Intensity** | 0-0.1 | Intensidad del ruido gaussiano |
| **Histogram Match** | 0-1 | Fuerza del matching de histograma con reales |
| **Preserve Annotations** | On/Off | Preservar anotaciones COCO |

4. **(Opcional)** Selecciona un **Reference Set** para histogram matching dirigido
5. Haz clic en **Apply**
6. Monitorea el progreso en el Job Monitor

**Resultado:** Imagenes con variaciones aleatorias aplicadas, sin sobreescribir las originales. Las anotaciones se preservan si la opcion esta activa.

---

## 8. Flujos de Trabajo Recomendados

### 8.1 Flujo Completo: Desde Cero

Para crear un dataset sintetico completo partiendo de imagenes reales:

```
1. Auto-Etiquetado      Etiquetar imagenes reales automaticamente
       |
2. Extraccion           Extraer objetos recortados del dataset etiquetado
       |
3. Domain Gap           Subir imagenes reales como reference set
       |
4. Workflow Paso 1      Analizar el dataset COCO generado
       |
5. Workflow Paso 2      Configurar efectos (usar preset del dominio)
       |
6. Workflow Paso 3      Seleccionar fondos reales + directorio de salida
       |
7. Workflow Paso 4      Generar dataset sintetico (primero un mini-batch)
       |
8. Paso 4.5             Validar domain gap del mini-batch
       |                   - Si gap alto: aplicar sugerencias y volver a paso 5
       |                   - Si gap bajo: continuar
       |
9. Workflow Paso 4      Regenerar dataset completo con parametros optimizados
       |
10. Domain Gap           (Opcional) Aplicar Domain Randomization
       |
11. Workflow Paso 5      Exportar a formato YOLO/VOC/COCO
       |
12. Workflow Paso 7      Dividir en Train/Val/Test
```

### 8.2 Flujo Rapido: Dataset Existente

Si ya tienes un dataset COCO y quieres generar mas datos:

```
1. Workflow Paso 1      Cargar dataset COCO existente
       |
2. Workflow Paso 2      Configurar (usar preset o configuracion anterior)
       |
3. Workflow Paso 3      Seleccionar fondos + salida
       |
4. Workflow Paso 4      Generar
       |
5. Workflow Paso 5      Exportar
       |
6. Workflow Paso 6      (Opcional) Combinar con dataset original
       |
7. Workflow Paso 7      Dividir en splits
```

### 8.3 Flujo Iterativo: Optimizar Domain Gap

Para reducir progresivamente el domain gap:

```
1. Subir reference set con imagenes reales
       |
2. Generar mini-batch (20-50 imagenes)
       |
3. Analizar domain gap
       |
4. Aplicar sugerencias de parametros
       |
5. Regenerar mini-batch con nuevos parametros
       |
6. Re-analizar domain gap
       |
   (Repetir 3-6 hasta gap < 30)
       |
7. Generar dataset completo con parametros optimizados
       |
8. (Opcional) Aplicar Domain Randomization para mas variedad
```

### 8.4 Flujo de Balanceo: Dataset Desbalanceado

Si tienes un dataset con clases muy desbalanceadas:

```
1. Post-Processing      Analizar distribucion de clases
       |
   Opcion A: Class Weights
       Copiar pesos y usarlos en el entrenamiento

   Opcion B: Generar mas datos de clases minoritarias
       |
2. Workflow Paso 1      Cargar dataset desbalanceado
       |
3. Workflow Paso 4      Generar mas imagenes para clases minoritarias
                        (ajustar targets por categoria)
       |
4. Workflow Paso 6      Combinar dataset original + nuevas sinteticas
       |
5. Post-Processing      Verificar nueva distribucion
```

---

## Consejos y Buenas Practicas

1. **Empieza con mini-batches**: Genera primero 20-50 imagenes para validar la configuracion antes de lanzar una generacion completa.

2. **Usa Domain Gap validation**: Siempre valida contra imagenes reales antes de generar datasets grandes. Unos minutos de validacion pueden ahorrarte horas de generacion.

3. **Fondos reales del dominio**: Usa imagenes de fondo del mismo entorno donde se desplegara el modelo. Esto reduce significativamente el domain gap.

4. **SAM3 para mascaras precisas**: Activa SAM3 en la extraccion de objetos para obtener mascaras de alta calidad. Los bordes limpios producen composiciones mas realistas.

5. **Balanceo de clases**: Asegurate de que todas las clases tienen suficientes muestras. Un ratio de desbalanceo mayor a 10x afectara negativamente al entrenamiento.

6. **Splits estratificados**: Siempre usa splits estratificados para mantener la proporcion de clases en train/val/test.

7. **Monitorea los servicios**: Verifica el estado de los servicios antes de lanzar jobs grandes. Si un servicio esta degradado, los resultados pueden ser suboptimos.

8. **Guarda configuraciones**: Exporta tu configuracion de dominio despues de optimizarla. Podras reutilizarla en futuros proyectos.
