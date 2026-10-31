# SISTEMA CUANTITATIVO DE APUESTAS DEPORTIVAS [+EV]
### Plataforma Multi-Deporte con Memoria Autonoma, Verificacion de Resultados y Analisis de Mercado

Plataforma cuantitativa disenada para la adquisicion, modelado y calificacion automatica de apuestas deportivas con valor esperado positivo (+EV).

---

## ESTRUCTURA DEL SISTEMA

- `iniciar_apuestas.bat`: Script de inicio rapido (enlazado al acceso directo del Escritorio).
- `app_ui.py`: Panel interactivo multi-pestaña para navegador web.
- `run_pipeline.py`: Motor de extraccion, 10,000 simulaciones Monte Carlo y seleccion.
- `reporte_apuestas_hoy.md`: Reporte diario estructurado.
- `src/history_tracker.py`: Modulo de memoria persistente, auditoria y calificacion con marcadores oficiales.
- `src/scraper.py`: Scraper de carteleras, cuotas y libros de Kalshi y Polymarket.
- `src/sabermetrics.py`: Metricas avanzadas (FIP, xERA, EPA/Play, Net Rating, xG).
- `src/monte_carlo.py`: Motor de 10,000 simulaciones (Normal multivariada y Dixon-Coles).
- `src/predictive_models.py`: Modelos 1 (Estadistico), 2 (Contextual) y 3 (Divergencia Sharp Money).
- `src/selection_engine.py`: Motor de descarte de duplicados y ensamble de vehiculos.

---

## FUNCIONALIDADES CLAVE

1. **Portada MVP Global:**
   - Filtra y presenta las mejores 4 selecciones consolidadas de todos los deportes donde el modelo detecta el mayor valor esperado y volumen institucional.
2. **Desglose Juego por Juego:**
   - Pestañas independientes para NBA, NHL, Futbol, MLB y NFL con probabilidades reales, cuotas de mercado y marcadores proyectados.
3. **Memoria e Historial Autonomo:**
   - Todas las selecciones generadas se guardan en `data/bet_history.json`.
   - Incluye el boton `[VERIFICAR RESULTADOS DE AYER]` que consulta los marcadores finales de ESPN, califica cada apuesta como GANADA o PERDIDA y calcula la tasa de acierto (Win Rate %) y unidades netas (PnL).
4. **Sin Gestion de Bankroll:**
   - Enfoque directo en cuotas, probabilidades reales y valor esperado (+EV).
5. **Formato Institucional:**
   - Libre de emojis y orientado a visualizacion cuantitativa profesional.
