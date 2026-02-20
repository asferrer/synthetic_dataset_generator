# Sistema de Auto-Tuning Predictivo con ML

## Descripción General

El sistema de **Auto-Tuning Predictivo** utiliza Machine Learning y Optimización Bayesiana para encontrar automáticamente la mejor configuración de parámetros que minimice el domain gap entre datos sintéticos y reales.

### Componentes Principales

1. **Predictor ML (XGBoost)**
   - Aprende de configuraciones históricas y sus resultados
   - Predice el gap score antes de generar datos
   - Se entrena incrementalmente con cada nueva evaluación

2. **Bayesian Optimization (Optuna)**
   - Explora el espacio de parámetros de forma inteligente
   - Usa TPE (Tree-structured Parzen Estimator) para sugerir configuraciones prometedoras
   - Converge rápidamente a la configuración óptima

3. **Hybrid Optimizer**
   - Combina predicción ML con optimización bayesiana
   - Warm-start: usa el predictor para inicializar con buenas configuraciones
   - Active learning: actualiza el modelo con resultados reales

---

## Arquitectura

```
┌─────────────────────────────────────────────────────────────┐
│                    Usuario / Frontend                        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  Gateway Service (Puerto 8000)               │
│                                                               │
│  Endpoints:                                                   │
│    POST   /ml-optimize/start          - Iniciar optimización │
│    GET    /ml-optimize/jobs/{id}      - Estado del job       │
│    GET    /ml-optimize/feature-importance - Análisis         │
│    POST   /ml-optimize/predict        - Predicción rápida    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│            Domain Gap Service (Puerto 8005)                  │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         Predictor Engine (XGBoost)                   │   │
│  │  - Config History: /shared/config_history.json       │   │
│  │  - Trained Model: /shared/gap_predictor_model.pkl    │   │
│  │  - Feature extraction & prediction                   │   │
│  └──────────────────────────────────────────────────────┘   │
│                         ▲ │                                   │
│                         │ ▼                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │    Bayesian Optimizer Engine (Optuna)               │   │
│  │  - TPE Sampler para exploración inteligente         │   │
│  │  - Study Storage: SQLite en /shared/optuna_studies/  │   │
│  │  - Pruning de trials no prometedores                │   │
│  └──────────────────────────────────────────────────────┘   │
│                         │                                     │
│                         ▼                                     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │          Metrics Engine (C-RADIOv4)                  │   │
│  │  - Evaluación real del gap score                     │   │
│  │  - RADIO-MMD, FD-RADIO, PRDC, Color dist            │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## Flujo de Trabajo

### 1. Optimización ML-Guided (Recomendado)

```python
# POST /ml-optimize/start
{
  "synthetic_dir": "/app/datasets/my_project/synthetic/images",
  "reference_set_id": "real_underwater_refs_001",
  "n_trials": 20,              # Número de configuraciones a probar
  "probe_size": 50,            # Imágenes por trial
  "warm_start": true,          # Usar predictor ML para inicializar
  "timeout_seconds": 3600,     # 1 hora máximo
  "current_config": {          # Config actual (opcional, baseline)
    "color_intensity": 0.15,
    "blur_strength": 0.5,
    ...
  }
}
```

**Respuesta:**
```json
{
  "job_id": "mlopt_a3f2e1c9b4d5",
  "status": "pending",
  "total_trials": 20
}
```

### 2. Monitorear Progreso

```python
# GET /ml-optimize/jobs/{job_id}
{
  "job_id": "mlopt_a3f2e1c9b4d5",
  "status": "optimizing",           # pending → optimizing → completed
  "current_trial": 12,
  "total_trials": 20,
  "best_gap_score": 18.5,
  "best_config": {
    "color_intensity": 0.08,
    "blur_strength": 0.3,
    "underwater_intensity": 0.12,
    ...
  },
  "optimization_history": [
    {"trial": 1, "gap_score": 45.2, "config": {...}},
    {"trial": 2, "gap_score": 38.1, "config": {...}},
    ...
  ],
  "feature_importance": {
    "color_intensity": 0.35,
    "blur_strength": 0.22,
    "lighting_intensity": 0.18,
    ...
  }
}
```

### 3. Análisis de Parámetros Importantes

```python
# GET /ml-optimize/feature-importance?source=combined
{
  "success": true,
  "source": "combined",
  "importances": {
    "color_intensity": 0.35,      # Más importante
    "blur_strength": 0.22,
    "lighting_intensity": 0.18,
    "underwater_intensity": 0.12,
    "caustics_intensity": 0.08,
    "shadow_opacity": 0.05        # Menos importante
  },
  "top_features": [
    "color_intensity",
    "blur_strength",
    "lighting_intensity"
  ]
}
```

### 4. Predicción Rápida (Sin Generar Datos)

```python
# POST /ml-optimize/predict
{
  "config": {
    "color_intensity": 0.10,
    "blur_strength": 0.4,
    "underwater_intensity": 0.15,
    ...
  }
}
```

**Respuesta:**
```json
{
  "success": true,
  "predicted_score": 22.3,
  "confidence": "high",
  "message": "Predicted from 45 historical configurations"
}
```

---

## Parámetros Optimizables

El sistema optimiza los siguientes parámetros de efectos:

| Parámetro               | Rango      | Descripción                                    |
|-------------------------|------------|------------------------------------------------|
| `color_intensity`       | 0.0 - 0.5  | Intensidad de corrección de color              |
| `blur_strength`         | 0.0 - 2.0  | Fuerza del desenfoque de matching              |
| `underwater_intensity`  | 0.0 - 0.5  | Intensidad del efecto underwater               |
| `caustics_intensity`    | 0.0 - 0.3  | Intensidad de cáusticas                        |
| `shadow_opacity`        | 0.0 - 0.3  | Opacidad de sombras                            |
| `lighting_intensity`    | 0.2 - 1.0  | Intensidad de iluminación                      |
| `motion_blur_probability` | 0.0 - 0.5 | Probabilidad de motion blur                   |
| `lighting_type`         | categórico | ambient / directional / underwater             |
| `water_clarity`         | categórico | clear / moderate / murky                       |

---

## Ventajas del Sistema ML

### vs. Optimización Manual
- ⚡ **100x más rápido**: converge en 10-20 trials vs cientos de pruebas manuales
- 🎯 **Más preciso**: encuentra óptimos globales, no solo locales
- 📊 **Data-driven**: decisiones basadas en datos históricos reales
- 🔄 **Mejora continua**: aprende de cada evaluación

### vs. Grid Search
- 🚀 **Exponencialmente más eficiente**: Grid search con 9 parámetros × 5 valores = 1.953.125 combinaciones
- 💡 **Inteligente**: explora zonas prometedoras primero
- ⏱️ **Termina más rápido**: converge antes de explorar todo el espacio

### vs. Random Search
- 🎲 **No aleatorio**: usa información de trials previos
- 📈 **Convergencia garantizada**: TPE sampler converge teóricamente
- 🧠 **Aprendizaje**: el predictor ML mejora con el tiempo

---

## Casos de Uso

### 1. Proyecto Nuevo (Sin Historial)

```bash
# Primera optimización: warm_start=false, más trials
POST /ml-optimize/start
{
  "reference_set_id": "my_refs",
  "synthetic_dir": "/app/output/baseline",
  "n_trials": 30,           # Más trials para explorar
  "warm_start": false       # Sin warm-start (no hay historial)
}
```

Después de 30 trials, el predictor estará entrenado.

### 2. Proyecto Existente (Con Historial)

```bash
# Optimizaciones subsiguientes: warm_start=true, menos trials
POST /ml-optimize/start
{
  "reference_set_id": "my_refs",
  "synthetic_dir": "/app/output/iteration_5",
  "n_trials": 10,           # Menos trials, el ML guía
  "warm_start": true        # Usar predictor para inicializar
}
```

Converge más rápido gracias al conocimiento acumulado.

### 3. Análisis "What-If"

```python
# Probar múltiples configs sin generar datos
for config in candidate_configs:
    result = POST("/ml-optimize/predict", json={"config": config})
    print(f"Config: {config} → Predicted gap: {result['predicted_score']}")

# Seleccionar la mejor y generar con ella
best_config = min(results, key=lambda x: x['predicted_score'])
```

### 4. Fine-Tuning de Config Actual

```bash
# Partir de una config conocida y optimizar alrededor de ella
POST /ml-optimize/start
{
  "current_config": {       # Config actual como baseline
    "color_intensity": 0.15,
    ...
  },
  "n_trials": 15,
  "warm_start": true
}
```

---

## Interpretación de Resultados

### Gap Score

- **0-20**: Excelente, casi indistinguible de datos reales
- **20-40**: Bueno, gap perceptible pero aceptable
- **40-60**: Moderado, necesita mejora
- **60-100**: Alto, datos sintéticos claramente diferentes

### Feature Importance

Indica qué parámetros tienen más impacto en el gap score:

- **> 0.20**: Parámetro crítico, requiere ajuste cuidadoso
- **0.10 - 0.20**: Parámetro importante
- **< 0.10**: Parámetro secundario

**Ejemplo práctico:**

Si `color_intensity` tiene importance=0.35, significa que ajustar este parámetro tiene 3.5x más impacto que ajustar `shadow_opacity` (importance=0.10).

---

## Persistencia y Escalabilidad

### Datos Persistentes

- **Historial de configuraciones**: `/shared/config_history.json`
  - Se acumula entre sesiones
  - Usado para entrenar el predictor ML

- **Modelo entrenado**: `/shared/gap_predictor_model.pkl`
  - Guardado automáticamente después de entrenar
  - Cargado al inicio del servicio

- **Optuna studies**: `/shared/optuna_studies/gap_optimization.db`
  - SQLite database con todos los trials
  - Permite continuar optimizaciones interrumpidas

### Escalabilidad

- **Parallel trials**: Optuna soporta optimización paralela
- **Distributed optimization**: Usar PostgreSQL como storage en vez de SQLite
- **Multi-domain**: Cada dominio puede tener su propio predictor

---

## Mejores Prácticas

### ✅ Recomendaciones

1. **Empezar con baseline**: Evaluar config por defecto antes de optimizar
2. **Suficientes trials**: Mínimo 20 trials para primera optimización
3. **Probe size adecuado**: 50-100 imágenes por trial (balance speed/accuracy)
4. **Usar warm-start**: Siempre que haya ≥10 configuraciones históricas
5. **Revisar feature importance**: Identificar parámetros críticos
6. **Iterar**: Optimizar en múltiples rondas para mejor convergencia

### ❌ Anti-Patrones

1. **Muy pocos trials**: < 10 trials no explora suficiente espacio
2. **Probe size muy pequeño**: < 20 imágenes da métricas ruidosas
3. **Ignorar historial**: No usar warm-start desperdicia conocimiento acumulado
4. **Cambiar reference set**: Usar siempre el mismo reference set para comparar
5. **Over-optimization**: Evitar overfitting al reference set específico

---

## Troubleshooting

### Problema: "Predictor not trained yet"

**Causa**: No hay suficientes configuraciones históricas (< 10)

**Solución**:
```bash
# Primera vez: usar optimización sin warm-start
POST /ml-optimize/start
{
  "warm_start": false,
  "n_trials": 20
}
```

### Problema: Optimización no converge

**Causa**: Espacio de búsqueda muy grande o métricas ruidosas

**Solución**:
- Aumentar `n_trials` (30-50)
- Aumentar `probe_size` (100-200)
- Reducir `parameter_ranges` (fijar algunos parámetros)

### Problema: Gap score no mejora

**Causa**: Problema no está en los parámetros de efectos

**Solución**:
- Revisar calidad de backgrounds y objetos
- Verificar placement config (oclusión, profundidad)
- Considerar Domain Randomization o Style Transfer

---

## Ejemplo Completo: Workflow End-to-End

```python
import httpx

API_BASE = "http://localhost:8000"

# 1. Crear reference set
refs_response = httpx.post(f"{API_BASE}/domain-gap/references/from-directory", json={
    "name": "Real Underwater Images",
    "directory_path": "/app/datasets/real_underwater",
    "domain_id": "underwater"
})
ref_set_id = refs_response.json()["set_id"]

# 2. Generar baseline sintético (config por defecto)
baseline_response = httpx.post(f"{API_BASE}/augment/compose-batch", json={
    "source_dataset": "/app/datasets/my_project",
    "output_dir": "/app/output/baseline",
    "num_images": 200
})
baseline_job_id = baseline_response.json()["job_id"]

# Esperar a que termine...

# 3. Medir baseline gap
metrics_response = httpx.post(f"{API_BASE}/domain-gap/metrics/compute", json={
    "synthetic_dir": "/app/output/baseline/images",
    "reference_set_id": ref_set_id
})
baseline_gap = metrics_response.json()["overall_gap_score"]
print(f"Baseline gap score: {baseline_gap}")  # e.g., 52.3

# 4. Optimizar con ML
opt_response = httpx.post(f"{API_BASE}/ml-optimize/start", json={
    "synthetic_dir": "/app/output/baseline/images",
    "reference_set_id": ref_set_id,
    "n_trials": 25,
    "probe_size": 75,
    "warm_start": True
})
opt_job_id = opt_response.json()["job_id"]

# 5. Monitorear progreso
while True:
    status = httpx.get(f"{API_BASE}/ml-optimize/jobs/{opt_job_id}").json()
    if status["status"] == "completed":
        break
    print(f"Trial {status['current_trial']}/{status['total_trials']}, "
          f"Best gap: {status['best_gap_score']}")
    time.sleep(30)

# 6. Obtener mejor config
best_config = status["best_config"]
best_gap = status["best_gap_score"]
print(f"Optimized gap score: {best_gap}")  # e.g., 21.7
print(f"Improvement: {(baseline_gap - best_gap) / baseline_gap * 100:.1f}%")

# 7. Generar dataset final con best_config
final_response = httpx.post(f"{API_BASE}/augment/compose-batch", json={
    "source_dataset": "/app/datasets/my_project",
    "output_dir": "/app/output/optimized",
    "num_images": 5000,
    "effects_config": best_config  # ¡Usar config optimizada!
})

# 8. Análisis de importancia
importance = httpx.get(f"{API_BASE}/ml-optimize/feature-importance").json()
print("Top parameters to tune:")
for param in importance["top_features"][:3]:
    print(f"  - {param}: {importance['importances'][param]:.2f}")
```

---

## Roadmap y Extensiones Futuras

### Fase 1 (Actual) ✅
- Predictor XGBoost
- Optimización Bayesiana con Optuna
- Feature importance analysis
- Persistencia de historial

### Fase 2 (Próximo)
- 🔄 Multi-objective optimization (gap + diversity + realism)
- 📊 Visualizaciones interactivas (Plotly dashboards)
- 🎯 Transfer learning entre dominios similares
- 🌐 Frontend web para configuración y monitoreo

### Fase 3 (Futuro)
- 🤖 Meta-learning: aprender estrategias de optimización
- 🔥 Neural Architecture Search para configuración óptima
- 🌊 Domain-specific priors (agua, cielo, indoor, etc.)
- ⚡ GPU-accelerated hyperparameter search

---

## Referencias

- [Optuna Documentation](https://optuna.readthedocs.io/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [TPE Algorithm Paper](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf)
- [Bayesian Optimization Survey](https://arxiv.org/abs/1807.02811)

---

## Soporte

Para preguntas o issues:
- GitHub Issues: [synthetic-dataset-generator/issues](https://github.com/your-repo/issues)
- Documentación completa: `docs/`
- API Reference: http://localhost:8000/docs (Swagger UI)
