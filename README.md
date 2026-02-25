# Grido Staffing Planner (MVP)

Planificador de dotación de personal para heladerías Grido.

Ingresá un **objetivo mensual de venta minorista (en kg)** y obtené:

- **Dotación recomendada por hora** (mapa de calor visual).
- **Métricas de productividad** por colaborador (kg/día, kg/mes).
- **Umbral de saturación**: constante operativa para decidir cuándo sumar o reducir personal.
- **Horas-persona totales** y **headcount estimado**.

## Tipos de local

| Tipo | Descripción | Mix de producto |
|------|-------------|-----------------|
| **Centro / Alto tránsito** | Local de alto paso (centro, peatonal) | 64% servido, 13% envasado |
| **Barrio** | Local de barrio, clientes habituales | 49% servido, 26% envasado |

## Cómo usar

1. Elegí el escenario (1 local o 2 locales).
2. Ingresá el objetivo de kg del mes.
3. Ajustá horario, mix de producto y parámetros operativos si es necesario.
4. Leé los resultados y descargá el plan en Excel o CSV.

## Ejecución local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Referencia

Los datos de referencia provienen de enero 2026 (temporada alta) e incluyen:
- Perfil de demanda p(día × hora) por tipo de local.
- Productividad base (kg/hora-persona) por hora.
- Composición del mix de producto por tipo de local.

Ver `DOCUMENTACION_APP.txt` para la documentación completa del modelo.
