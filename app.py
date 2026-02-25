import calendar
import json
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REFERENCE_PATH = Path(__file__).parent / "reference" / "grido_ref_2026_01.json"

WEEKDAY_NAMES_ES = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]

DISPLAY_COLS_ES = {
    "date": "Fecha",
    "dow_name": "Día",
    "hour": "Hora",
    "kg_hour": "Kg planificados (esta hora)",
    "staff_required": "Personas requeridas (esta hora)",
    "person_hours": "Horas-persona (esta hora)",
}

PEAK_HOURS: Tuple[int, ...] = (20, 21, 22, 23)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OpenSchedule:
    open_time: time
    close_time: time
    open_weekdays: Tuple[bool, bool, bool, bool, bool, bool, bool]  # Mon..Sun

    def is_open_day(self, d: date) -> bool:
        return self.open_weekdays[d.weekday()]

    def open_interval(self, d: date) -> Tuple[datetime, datetime]:
        """Return open interval start/end datetimes for date *d*.

        If close_time <= open_time, closing is considered after midnight (+1 day).
        """
        start = datetime.combine(d, self.open_time)
        end = datetime.combine(d, self.close_time)
        if end <= start:
            end += timedelta(days=1)
        return start, end

    def open_hours_for_date(self, d: date) -> List[Tuple[date, int]]:
        """Return list of (slot_date, hour) buckets that overlap with open interval."""
        if not self.is_open_day(d):
            return []

        start, end = self.open_interval(d)

        cur = start.replace(minute=0, second=0, microsecond=0)
        if cur > start:
            cur -= timedelta(hours=1)

        hours: List[Tuple[date, int]] = []
        while cur < end:
            nxt = cur + timedelta(hours=1)
            overlap = max(0.0, (min(end, nxt) - max(start, cur)).total_seconds())
            if overlap > 0:
                hours.append((cur.date(), cur.hour))
            cur = nxt
        return hours


# ---------------------------------------------------------------------------
# Pure functions (no Streamlit dependency)
# ---------------------------------------------------------------------------

def month_dates(year: int, month: int) -> List[date]:
    n = calendar.monthrange(year, month)[1]
    return [date(year, month, d) for d in range(1, n + 1)]


@st.cache_data
def load_reference(path: str) -> dict:
    """Load and cache the JSON reference file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_slot_weights(
    dates: Iterable[date],
    schedule: OpenSchedule,
    p_dow_hour: List[List[float]],
) -> pd.DataFrame:
    """Return slots with weights based on p(dow, hour)."""
    rows = []
    for d in dates:
        for slot_date, hour in schedule.open_hours_for_date(d):
            dow = slot_date.weekday()
            rows.append(
                {
                    "date": slot_date,
                    "dow": dow,
                    "dow_name": WEEKDAY_NAMES_ES[dow],
                    "hour": hour,
                    "weight": float(p_dow_hour[dow][hour]),
                }
            )
    return pd.DataFrame(rows)


def _safe_kpph_array(kpph_by_hour: List[Optional[float]], fallback: float) -> np.ndarray:
    """Return a 24-element array with safe kpph values (no None/NaN/<=0)."""
    arr = np.array(
        [v if (v is not None and np.isfinite(v) and v > 0) else fallback for v in kpph_by_hour],
        dtype=float,
    )
    return arr


def compute_staffing_plan(
    *,
    kg_goal_month: float,
    demand_multiplier: float,
    efficiency_multiplier: float,
    product_mix_factor: float,
    base_min_staff: int,
    peak_min_staff: int,
    peak_hours: Tuple[int, ...],
    kpph_by_hour: List[Optional[float]],
    kpph_fallback: float,
    slots: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Compute staffing plan for a store.

    product_mix_factor adjusts kpph for the store's product mix:
      - 1.0 = baseline (mostly scooped / high-labor products)
      - >1.0 = more packaged products → each person handles more raw kg
    """
    if slots.empty:
        return slots, {}

    if kg_goal_month < 0:
        raise ValueError("kg_goal_month debe ser >= 0")
    if demand_multiplier <= 0 or efficiency_multiplier <= 0:
        raise ValueError("Los multiplicadores deben ser > 0")

    slots = slots.copy()

    wsum = float(slots["weight"].sum())
    if wsum <= 0:
        slots["weight"] = 1.0
        wsum = float(slots["weight"].sum())

    kg_total = kg_goal_month * demand_multiplier
    slots["kg_hour"] = kg_total * (slots["weight"] / wsum)

    # --- Vectorised staffing calculation (replaces iterrows) ---
    kpph_arr = _safe_kpph_array(kpph_by_hour, kpph_fallback)
    effective_kpph = kpph_arr[slots["hour"].values] * efficiency_multiplier * product_mix_factor

    raw_staff = np.where(effective_kpph > 0, np.ceil(slots["kg_hour"].values / effective_kpph), 0).astype(int)
    raw_staff = np.maximum(raw_staff, base_min_staff)

    is_peak = np.isin(slots["hour"].values, list(peak_hours))
    raw_staff = np.where(is_peak, np.maximum(raw_staff, peak_min_staff), raw_staff)

    slots["staff_required"] = raw_staff
    slots["person_hours"] = slots["staff_required"].astype(float)

    summary = {
        "kg_total_month_adjusted": float(slots["kg_hour"].sum()),
        "person_hours_total": float(slots["person_hours"].sum()),
        "avg_staff_open_hours": float(slots["staff_required"].mean()),
        "max_staff_any_hour": float(slots["staff_required"].max()),
        "min_staff_any_hour": float(slots["staff_required"].min()),
    }
    return slots, summary


def merge_plans(plans_list: list) -> pd.DataFrame:
    """Merge multiple plan DataFrames by summing kg_hour and staff_required per slot."""
    merged = None
    for p in plans_list:
        pp = p[["date", "dow", "dow_name", "hour", "kg_hour", "staff_required"]].copy()
        if merged is None:
            merged = pp
        else:
            merged = merged.merge(
                pp, on=["date", "dow", "dow_name", "hour"], how="outer", suffixes=("", "_r")
            )
            merged["kg_hour"] = merged["kg_hour"].fillna(0) + merged["kg_hour_r"].fillna(0)
            merged["staff_required"] = merged["staff_required"].fillna(0) + merged["staff_required_r"].fillna(0)
            merged = merged[["date", "dow", "dow_name", "hour", "kg_hour", "staff_required"]]
    merged["person_hours"] = merged["staff_required"].astype(float)
    return merged


def compute_capacity_per_employee(kpph_overall: float, pmf: float, eff: float, hours_month: float) -> float:
    """Constant: max kg/month one employee can handle (independent of kg goal)."""
    return kpph_overall * pmf * eff * hours_month


def plan_for_type(
    type_ref: dict,
    kg_goal_month: float,
    pmf: float,
    dates: List[date],
    schedule: OpenSchedule,
    demand_multiplier: float,
    efficiency_multiplier: float,
    base_min_staff: int,
    peak_min_staff: int,
) -> Tuple[pd.DataFrame, Dict[str, float], float]:
    """Build the staffing plan for one store type."""
    kpph_by_hour = type_ref["kpph_by_hour"]
    kpph_fallback = float(type_ref.get("kpph_overall") or 3.9)
    slots = build_slot_weights(dates, schedule, type_ref["p_dow_hour"])
    plan, summary = compute_staffing_plan(
        kg_goal_month=float(kg_goal_month),
        demand_multiplier=float(demand_multiplier),
        efficiency_multiplier=float(efficiency_multiplier),
        product_mix_factor=float(pmf),
        base_min_staff=int(base_min_staff),
        peak_min_staff=int(peak_min_staff),
        peak_hours=PEAK_HOURS,
        kpph_by_hour=kpph_by_hour,
        kpph_fallback=kpph_fallback,
        slots=slots,
    )
    return plan, summary, kpph_fallback


def to_excel_bytes(df: pd.DataFrame, summary: Dict[str, float]) -> bytes:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="plan")
        pd.DataFrame([summary]).to_excel(w, index=False, sheet_name="resumen")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _dow_sort_key(dow_name: str) -> int:
    try:
        return WEEKDAY_NAMES_ES.index(dow_name)
    except ValueError:
        return 999


def render_staff_heatmap(avg_df: pd.DataFrame, title: str) -> None:
    """Heatmap Día × Hora with avg staff and text labels."""
    if avg_df.empty:
        st.info("No hay datos para graficar.")
        return

    chart_df = avg_df.copy()
    chart_df["Colaboradores"] = chart_df["avg_staff"].round(0).astype(int)

    n_hours = chart_df["hour"].nunique()
    cell_w = max(28, min(42, 600 // max(n_hours, 1)))
    chart_w = cell_w * n_hours + 80

    base = (
        alt.Chart(chart_df, title=alt.Title(title, anchor="start"))
        .mark_rect(stroke="white", strokeWidth=2, cornerRadius=3)
        .encode(
            x=alt.X(
                "hour:O",
                title="Hora",
                axis=alt.Axis(labelAngle=0, labelFontSize=12, titleFontSize=13),
                scale=alt.Scale(paddingOuter=0.05),
            ),
            y=alt.Y(
                "dow_name:O",
                sort=WEEKDAY_NAMES_ES,
                title="Día",
                axis=alt.Axis(labelFontSize=13, titleFontSize=13),
                scale=alt.Scale(paddingOuter=0.05),
            ),
            color=alt.Color(
                "avg_staff:Q",
                title="Colaboradores",
                scale=alt.Scale(scheme="orangered", domainMin=0),
                legend=alt.Legend(direction="horizontal", orient="top"),
            ),
            tooltip=[
                alt.Tooltip("dow_name:N", title="Día"),
                alt.Tooltip("hour:O", title="Hora"),
                alt.Tooltip("Colaboradores:Q", title="Colaboradores necesarios"),
                alt.Tooltip("avg_kg:Q", title="Kg prom. en esa hora", format=".1f"),
            ],
        )
    )

    max_staff_val = chart_df["avg_staff"].max()
    text = (
        alt.Chart(chart_df)
        .mark_text(fontSize=14, fontWeight="bold")
        .encode(
            x=alt.X("hour:O"),
            y=alt.Y("dow_name:O", sort=WEEKDAY_NAMES_ES),
            text=alt.Text("Colaboradores:Q"),
            color=alt.condition(
                alt.datum.avg_staff > (max_staff_val * 0.7),
                alt.value("white"),
                alt.value("black"),
            ),
        )
    )

    st.altair_chart(
        (base + text).properties(width=chart_w, height=380).configure_view(step=50),
        use_container_width=False,
    )


# ===========================================================================
# Streamlit UI
# ===========================================================================

st.set_page_config(page_title="Grido Staffing Planner (MVP)", layout="wide")

st.title("Grido Staffing Planner (MVP)")
st.caption("Planificador de dotación por hora en base a kg minoristas/mes y curvas tipo (Centro vs Barrio).")

ref = load_reference(str(REFERENCE_PATH))

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.subheader("Inputs")

    scenario = st.radio(
        "Escenario",
        options=["1 local", "2 locales (Centro + Barrio)"],
        help=(
            "Si tu objetivo mensual de kg corresponde a la suma de 2 heladerías (como Edén + España), "
            "usá '2 locales' para repartir los kg entre ambos tipos y estimar dotación total coherente."
        ),
    )

    # Always define the two type keys we support
    tipo_centro_key = "centro"
    tipo_barrio_key = "barrio"
    tipo_centro = ref["types"][tipo_centro_key]
    tipo_barrio = ref["types"][tipo_barrio_key]

    if scenario == "1 local":
        tipo = st.selectbox(
            "Tipo de local",
            options=list(ref["types"].keys()),
            format_func=lambda k: ref["types"][k]["label"],
        )
        st.caption(
            "En '1 local', el objetivo de kg se asume para ese local únicamente. "
            "La app usa una curva de demanda (día×hora) y una productividad (kg/hora-persona) de referencia."
        )
    else:
        st.caption(
            "En '2 locales', el objetivo de kg se reparte entre Centro y Barrio y se calcula dotación por separado."
        )
        default_share_centro = float(tipo_centro.get("tickets_share_jan5_31", 0.67))
        share_centro = st.slider(
            "Reparto de kg al local Centro (%)",
            min_value=0,
            max_value=100,
            value=int(round(default_share_centro * 100)),
            step=1,
            help="Si no sabés el reparto real, dejalo en el porcentaje de referencia.",
        )
        share_barrio = 100 - share_centro
        st.caption(f"Reparto actual: **Centro {share_centro}%** / **Barrio {share_barrio}%**.")

    today = date.today()
    col_a, col_b = st.columns(2)
    with col_a:
        year = st.number_input("Año", min_value=2020, max_value=2035, value=int(today.year), step=1)
    with col_b:
        month = st.number_input("Mes", min_value=1, max_value=12, value=int(today.month), step=1)

    kg_goal = st.number_input("Objetivo kg minorista (mes)", min_value=0.0, value=8000.0, step=100.0)

    st.divider()
    st.subheader("Horario del local (simplificado)")

    default_open = time(11, 0)
    default_close = time(2, 0)

    open_time = st.time_input("Apertura", value=default_open)
    close_time = st.time_input("Cierre", value=default_close)

    st.caption("Si el cierre es menor/igual a la apertura, se asume cierre después de medianoche.")
    open_days = st.multiselect(
        "Días abiertos",
        options=list(range(7)),
        default=[0, 1, 2, 3, 4, 5, 6],
        format_func=lambda d: WEEKDAY_NAMES_ES[d],
    )
    open_weekdays = tuple(d in open_days for d in range(7))

    # --- Mix de producto ---
    st.divider()
    st.subheader("Mix de producto (intensidad de mano de obra)")
    st.caption(
        "No todos los kg requieren el mismo esfuerzo. Servir 1 kg de bochas demanda mucha más "
        "mano de obra que vender 1 kg de bombones en caja. Este factor ajusta la productividad "
        "según el mix del local."
    )

    def _show_mix_caption(detail: dict, label: str = "") -> None:
        if detail:
            prefix = f"Ref. {label}: " if label else ""
            st.caption(
                f"{prefix}"
                f"🍦 **{detail.get('high_labor_kg_pct', 0)*100:.0f}%** servido (bochas, kilos) · "
                f"🍡 **{detail.get('medium_labor_kg_pct', 0)*100:.0f}%** intermedio (sueltos) · "
                f"📦 **{detail.get('low_labor_kg_pct', 0)*100:.0f}%** envasado (cajas, familiares)"
            )

    if scenario == "1 local":
        default_pmf = float(ref["types"].get(tipo, {}).get("product_mix_factor", 1.0))
        pmf_label = ref["types"][tipo]["label"]
        mix_detail = ref["types"].get(tipo, {}).get("product_mix_detail", {})
        product_mix_factor_single = st.slider(
            f"Factor mix — {pmf_label}",
            min_value=0.80,
            max_value=1.50,
            value=default_pmf,
            step=0.05,
            help=(
                "1.0 = mix de referencia Centro (mayormente servido). "
                ">1.0 = más productos envasados → cada persona despacha más kg. "
                "Referencia Barrio: 1.20."
            ),
        )
        _show_mix_caption(mix_detail, pmf_label)
    else:
        col_pmf1, col_pmf2 = st.columns(2)
        with col_pmf1:
            product_mix_factor_centro = st.slider(
                "Factor mix — Centro",
                min_value=0.80,
                max_value=1.50,
                value=float(tipo_centro.get("product_mix_factor", 1.0)),
                step=0.05,
                help="1.0 = mayormente servido (bochas, cucuruchos).",
            )
            _show_mix_caption(tipo_centro.get("product_mix_detail", {}))
        with col_pmf2:
            product_mix_factor_barrio = st.slider(
                "Factor mix — Barrio",
                min_value=0.80,
                max_value=1.50,
                value=float(tipo_barrio.get("product_mix_factor", 1.20)),
                step=0.05,
                help=">1.0 = más productos envasados (menos mano de obra por kg).",
            )
            _show_mix_caption(tipo_barrio.get("product_mix_detail", {}))

    # --- Parámetros operativos ---
    st.divider()
    st.subheader("Parámetros operativos")
    demand_multiplier = 1.0 + (st.slider("Margen de demanda (%)", 0, 40, 0, 1) / 100.0)
    efficiency_multiplier = st.slider("Eficiencia (1.0 = referencia)", 0.70, 1.20, 1.00, 0.01)

    base_min_staff = st.number_input("Mínimo personas por hora (si está abierto)", min_value=0, value=1, step=1)
    peak_min_staff = st.number_input(
        f"Mínimo en horas pico ({PEAK_HOURS[0]}–{PEAK_HOURS[-1]})", min_value=0, value=2, step=1
    )

    # --- Conversión a empleados ---
    st.divider()
    st.subheader("Conversión a empleados")
    hours_per_employee_month = st.number_input(
        "Horas/mes por empleado (para convertir horas-persona → empleados)",
        min_value=40,
        max_value=240,
        value=180,
        step=5,
        help=(
            "La dotación se calcula primero como horas-persona. "
            "Luego se estima 'empleados necesarios' como: ceil(horas-persona del mes / horas/mes por empleado)."
        ),
    )


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------

schedule = OpenSchedule(
    open_time=open_time,
    close_time=close_time,
    open_weekdays=open_weekdays,
)

dates = month_dates(int(year), int(month))

plan_kwargs = dict(
    dates=dates,
    schedule=schedule,
    demand_multiplier=float(demand_multiplier),
    efficiency_multiplier=float(efficiency_multiplier),
    base_min_staff=int(base_min_staff),
    peak_min_staff=int(peak_min_staff),
)

if scenario == "1 local":
    type_ref = ref["types"][tipo]
    plan, summary, kpph_fallback = plan_for_type(type_ref, float(kg_goal), product_mix_factor_single, **plan_kwargs)
    plans = [(type_ref["label"], plan, summary, kpph_fallback)]
else:
    kg_centro = float(kg_goal) * (share_centro / 100.0)
    kg_barrio = float(kg_goal) * (share_barrio / 100.0)
    plan_c, sum_c, kpph_c = plan_for_type(tipo_centro, kg_centro, product_mix_factor_centro, **plan_kwargs)
    plan_b, sum_b, kpph_b = plan_for_type(tipo_barrio, kg_barrio, product_mix_factor_barrio, **plan_kwargs)
    plans = [
        (tipo_centro["label"], plan_c, sum_c, kpph_c),
        (tipo_barrio["label"], plan_b, sum_b, kpph_b),
    ]

# --- Guards ---
if any(p.empty for _, p, _, _ in plans):
    st.warning("No se generaron slots abiertos. Revisá días abiertos y horario.")
    st.stop()

# --- Aggregated metrics ---
summary_total = {
    "kg_total_month_adjusted": float(sum(s["kg_total_month_adjusted"] for _, _, s, _ in plans)),
    "person_hours_total": float(sum(s["person_hours_total"] for _, _, s, _ in plans)),
    "max_staff_any_hour": float(max(s["max_staff_any_hour"] for _, _, s, _ in plans)),
}
headcount_total = (
    int(np.ceil(summary_total["person_hours_total"] / float(hours_per_employee_month)))
    if hours_per_employee_month
    else 0
)

open_days_in_month = sum(1 for d in dates if schedule.is_open_day(d))

kg_per_employee_month = (
    summary_total["kg_total_month_adjusted"] / headcount_total if headcount_total > 0 else 0.0
)
kg_per_employee_day = kg_per_employee_month / open_days_in_month if open_days_in_month > 0 else 0.0

# --- Constant capacity per employee (independent of kg goal) ---
if scenario == "1 local":
    capacity_per_emp_month = compute_capacity_per_employee(
        float(ref["types"][tipo].get("kpph_overall") or 3.9),
        product_mix_factor_single,
        float(efficiency_multiplier),
        float(hours_per_employee_month),
    )
else:
    cap_c = compute_capacity_per_employee(
        float(tipo_centro.get("kpph_overall") or 3.9),
        product_mix_factor_centro,
        float(efficiency_multiplier),
        float(hours_per_employee_month),
    )
    cap_b = compute_capacity_per_employee(
        float(tipo_barrio.get("kpph_overall") or 3.9),
        product_mix_factor_barrio,
        float(efficiency_multiplier),
        float(hours_per_employee_month),
    )
    capacity_per_emp_month = cap_c * (share_centro / 100.0) + cap_b * (share_barrio / 100.0)

capacity_per_emp_day = capacity_per_emp_month / open_days_in_month if open_days_in_month > 0 else 0.0
team_capacity_kg = capacity_per_emp_month * headcount_total
headroom_kg = team_capacity_kg - summary_total["kg_total_month_adjusted"]

# ===========================================================================
# MAIN AREA — Metrics
# ===========================================================================

col1, col2, col3 = st.columns(3)
col1.metric("Kg planificados (ajustado)", f"{summary_total['kg_total_month_adjusted']:.0f}")
col2.metric("Empleados necesarios (estimación)", f"{headcount_total}")
col3.metric("Máximo personas en una hora", f"{summary_total['max_staff_any_hour']:.0f}")

col4, col5, col6 = st.columns(3)
col4.metric("Kg / colaborador / mes", f"{kg_per_employee_month:.0f}")
col5.metric("Kg / colaborador / día", f"{kg_per_employee_day:.1f}")
col6.metric(
    "Horas-persona (mes)",
    f"{summary_total['person_hours_total']:.0f}",
    help=f"Días abiertos en el mes: {open_days_in_month}",
)

st.divider()

# --- Saturation indicator (constant-based) ---
st.subheader("Umbral de saturación (constante operativa)")
st.caption(
    "Este umbral es una **constante** derivada de la productividad de referencia. "
    "No cambia con los kg del mes: si en febrero un colaborador puede hacer hasta X kg, "
    "en marzo también. Usalo para decidir cuándo sumar o reducir personal según la estacionalidad."
)

sat_col1, sat_col2, sat_col3 = st.columns(3)
sat_col1.metric(
    "Máx. kg / colaborador / mes",
    f"{capacity_per_emp_month:.0f}",
    delta=f"{capacity_per_emp_day:.1f} kg/día",
    delta_color="off",
    help=(
        "Constante operativa: kg máximos que un colaborador puede cubrir por mes "
        "(kpph referencia × mix producto × eficiencia × horas/mes). "
        "Si la venta por persona supera este valor → sumar gente."
    ),
)
sat_col2.metric(
    "Capacidad del equipo actual (kg/mes)",
    f"{team_capacity_kg:.0f}",
    help=f"= {headcount_total} empleados × {capacity_per_emp_month:.0f} kg/emp/mes",
)
sat_col3.metric(
    "Margen disponible (kg/mes)",
    f"+{headroom_kg:.0f}" if headroom_kg >= 0 else f"{headroom_kg:.0f}",
    help="Cuántos kg más puede absorber el equipo actual antes de necesitar +1 colaborador.",
)

st.divider()

# ===========================================================================
# MAIN AREA — Heatmaps
# ===========================================================================

st.subheader("Mapa de calor: dotación por día y hora (promedio del mes)")
st.caption(
    "Este heatmap muestra la **cantidad promedio de personas requeridas** por hora para cubrir el objetivo de kg, "
    "respetando la concentración horaria del modelo de referencia."
)

tabs = st.tabs([name for name, *_ in plans] + (["TOTAL"] if len(plans) > 1 else []))

for idx, (name, p, s, kpph_used) in enumerate(plans):
    with tabs[idx]:
        if scenario == "1 local":
            _pmf_used = product_mix_factor_single
        else:
            _pmf_used = product_mix_factor_centro if idx == 0 else product_mix_factor_barrio
        eff_kpph = kpph_used * _pmf_used * efficiency_multiplier
        st.caption(
            f"Productividad base: **{kpph_used:.2f} kg/h-persona** × mix **{_pmf_used:.2f}** "
            f"× eficiencia **{efficiency_multiplier:.2f}** = "
            f"**{eff_kpph:.2f} kg/h-persona efectivo**."
        )
        avg = (
            p.groupby(["dow", "dow_name", "hour"], as_index=False)
            .agg(avg_staff=("staff_required", "mean"), avg_kg=("kg_hour", "mean"))
            .sort_values(["dow", "hour"])
        )
        render_staff_heatmap(avg, title=f"{name} — Personas requeridas (prom.)")

        st.divider()
        st.subheader("Tabla: dotación promedio por hora (día de semana)")
        avg_table = avg.rename(
            columns={
                "dow_name": "Día",
                "hour": "Hora",
                "avg_staff": "Personas prom. requeridas",
                "avg_kg": "Kg prom. en esa hora",
            }
        )[["Día", "Hora", "Personas prom. requeridas", "Kg prom. en esa hora"]]
        st.dataframe(avg_table, hide_index=True)

if len(plans) > 1:
    with tabs[-1]:
        merged = merge_plans([p for _, p, _, _ in plans])
        avg_t = (
            merged.groupby(["dow", "dow_name", "hour"], as_index=False)
            .agg(avg_staff=("staff_required", "mean"), avg_kg=("kg_hour", "mean"))
            .sort_values(["dow", "hour"])
        )
        render_staff_heatmap(avg_t, title="TOTAL — Personas requeridas (prom.)")

        st.divider()
        st.subheader("Tabla: dotación promedio por hora (día de semana) — TOTAL")
        avg_t_table = avg_t.rename(
            columns={
                "dow_name": "Día",
                "hour": "Hora",
                "avg_staff": "Personas prom. requeridas",
                "avg_kg": "Kg prom. en esa hora",
            }
        )[["Día", "Hora", "Personas prom. requeridas", "Kg prom. en esa hora"]]
        st.dataframe(avg_t_table, hide_index=True)

# ===========================================================================
# Detailed plan + Downloads
# ===========================================================================

st.subheader("Plan detallado (fecha/hora)")
if len(plans) == 1:
    plan_show = plans[0][1].sort_values(["date", "hour"]).copy()
else:
    plan_show = merge_plans([p for _, p, _, _ in plans]).sort_values(["date", "hour"])

plan_show_es = plan_show.rename(columns=DISPLAY_COLS_ES)
plan_show_es = plan_show_es[[c for c in DISPLAY_COLS_ES.values() if c in plan_show_es.columns]]
st.dataframe(plan_show_es, hide_index=True)

st.divider()
st.subheader("Descargas")

excel_summary = {
    "Kg planificados (ajustado)": summary_total["kg_total_month_adjusted"],
    "Horas-persona (mes)": summary_total["person_hours_total"],
    "Empleados necesarios (estimación)": headcount_total,
    "Máximo personas en una hora": summary_total["max_staff_any_hour"],
    "Kg / colaborador / mes": kg_per_employee_month,
    "Kg / colaborador / día": kg_per_employee_day,
    "Umbral máx. kg / colaborador / mes": capacity_per_emp_month,
    "Umbral máx. kg / colaborador / día": capacity_per_emp_day,
    "Capacidad del equipo (kg/mes)": team_capacity_kg,
    "Margen disponible (kg/mes)": headroom_kg,
}

csv_bytes = plan_show.to_csv(index=False).encode("utf-8")
st.download_button("Descargar CSV", data=csv_bytes, file_name="plan_dotacion.csv", mime="text/csv")

xlsx_bytes = to_excel_bytes(plan_show_es, excel_summary)
st.download_button(
    "Descargar Excel",
    data=xlsx_bytes,
    file_name="plan_dotacion.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

# ===========================================================================
# Transparency / Reference
# ===========================================================================

with st.expander("Ver referencia y supuestos (transparencia del modelo)"):
    st.markdown(
        "- **Curva de demanda**: proporción por **día de semana × hora** (derivada del detalle con timestamp).\n"
        "- **Productividad base**: kg por **hora-persona** por hora (derivada de los horarios programados).\n"
        "- **Factor de mix de producto**: ajusta la productividad según la proporción de productos "
        "servidos (bochas, cucuruchos) vs envasados (cajas, familiares, tortas). "
        "Un local con más envasados necesita menos mano de obra por kg.\n"
        "- **Productividad efectiva**: `kpph_base × factor_mix × eficiencia`.\n"
        "- **Umbral de saturación**: `kpph_overall × factor_mix × eficiencia × horas/mes por empleado`. "
        "Es una constante que no depende del objetivo de kg.\n"
        "- **Empleados necesarios (estimación)**: "
        "\\(\\lceil \\text{horas-persona del mes} / \\text{horas/mes por empleado} \\rceil\\)."
    )

    st.subheader("Tipo de local seleccionado")
    if scenario == "1 local":
        st.json(
            {
                "tipo_local": type_ref["label"],
                "mapeo_referencia": {
                    "horarios_marker": type_ref.get("store_marker"),
                    "demanda_ticket_store": type_ref.get("ticket_store"),
                },
            }
        )
    else:
        st.json(
            {
                "tipo_local": "2 locales (Centro + Barrio)",
                "reparto_kg_pct": {"centro": share_centro, "barrio": share_barrio},
                "mapeo_referencia": {
                    "centro": {
                        "horarios_marker": tipo_centro.get("store_marker"),
                        "demanda_ticket_store": tipo_centro.get("ticket_store"),
                    },
                    "barrio": {
                        "horarios_marker": tipo_barrio.get("store_marker"),
                        "demanda_ticket_store": tipo_barrio.get("ticket_store"),
                    },
                },
            }
        )

    st.subheader("Cobertura de la referencia")
    st.json(ref.get("coverage", {}))

    st.subheader("Supuestos")
    st.json(ref.get("assumptions", {}))
