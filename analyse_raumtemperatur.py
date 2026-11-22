"""
Raumtemperatur- und Wetterkorrelationsanalyse
Friedrich-Hölderlin-Schule, Stadt Asperg (Mai – September 2026)

Liest die Raumtemperatur-Messdaten und die Wetterstationsdaten ein,
aggregiert sie pro Raum, Stockwerk, Stunde, Tag und Monat und erzeugt
einen interaktiven, eigenständigen HTML-Bericht im hellen, seriösen Design.
"""

import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(r"D:\Antigravity\RaumtemperaturAnalyse")
ROOM_CSV = BASE_DIR / "2026-11-10 Stadt Asperg Hoelderlin 05-26 bis 09-26.CSV"
WEATHER_CSV = BASE_DIR / "Wetterdaten_Tobi-Wolf_Wetterstation-Garten.csv"
OUTPUT_HTML = BASE_DIR / "Raumtemperatur_Analyse_Asperg.html"

MONTHS = [
    ("all", "Gesamt (Mai–Sep 2026)"),
    ("2026-05", "Mai 2026"),
    ("2026-06", "Juni 2026"),
    ("2026-07", "Juli 2026"),
    ("2026-08", "August 2026"),
    ("2026-09", "September 2026"),
]


def parse_room_name(raw_name: str) -> tuple[str, str]:
    """Extrahiert Stockwerk und bereinigten Raumnamen aus dem Sensornamen."""
    short = re.sub(r"^Friedrich-Hölderlin-Schule\s*-\s*", "", raw_name.strip())
    base = re.sub(r"\s+-\s+[a-z]$", "", short)
    parts = base.split(" - ", 1)
    floor = parts[0] if len(parts) > 1 else "Sonstige"
    return floor, base


def load_room_data(path: Path) -> pd.DataFrame:
    """Lädt die Raumtemperaturdaten (Spalte A, D, E) und filtert Mai–Sep 2026."""
    df = pd.read_csv(path, sep=";", encoding="cp1252", usecols=[0, 3, 4], dtype=str)
    df.columns = ["time", "name", "temp_raw"]
    df["temp"] = pd.to_numeric(df["temp_raw"].str.replace(",", ".", regex=False), errors="coerce")
    df = df.dropna(subset=["temp", "name"]).copy()
    # Bekanntes Encoding-Artefakt in einzelnen Raumnamen bereinigen
    df["name"] = df["name"].str.replace("Ã¼", "ü", regex=False)
    df["dt"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert("Europe/Berlin")
    df = df[(df["dt"] >= "2026-05-01") & (df["dt"] < "2026-10-01")].copy()

    parsed = df["name"].apply(parse_room_name)
    df["floor"] = [p[0] for p in parsed]
    df["room"] = [p[1] for p in parsed]
    df["hour"] = df["dt"].dt.floor("h")
    df["date"] = df["dt"].dt.strftime("%Y-%m-%d")
    df["month"] = df["dt"].dt.strftime("%Y-%m")
    return df


def load_weather_data(path: Path) -> pd.DataFrame:
    """Lädt die Wetterstationsdaten (Garten) inkl. Bereinigung doppelter Anführungszeichen."""
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            s = line.strip()
            if s.startswith('"') and s.endswith('"'):
                s = s[1:-1].replace('""', '"')
            if s.startswith("Zeitstempel,") or not s:
                continue
            lines.append(s)

    df = pd.read_csv(io.StringIO(header + "\n".join(lines)), low_memory=False)
    df["ts_num"] = pd.to_numeric(df["Zeitstempel"], errors="coerce")
    df["out_temp"] = pd.to_numeric(df["Wert"].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    df = df.dropna(subset=["ts_num", "out_temp"]).copy()
    df["dt"] = pd.to_datetime(df["ts_num"], unit="ms", utc=True).dt.tz_convert("Europe/Berlin")
    df = df[(df["dt"] >= "2026-05-01") & (df["dt"] < "2026-10-01")].copy()
    df["hour"] = df["dt"].dt.floor("h")
    df["date"] = df["dt"].dt.strftime("%Y-%m-%d")
    df["month"] = df["dt"].dt.strftime("%Y-%m")
    return df


def round_list(series: pd.Series, decimals: int = 2) -> list:
    return [None if pd.isna(v) else round(float(v), decimals) for v in series]


def build_payload(df_r: pd.DataFrame, df_w: pd.DataFrame) -> dict:
    """Bereitet alle aggregierten Kennzahlen und Zeitreihen für das HTML-Dashboard auf."""
    # Vollständige Tages- und Stundenachsen (Mai–Sep 2026)
    all_days = pd.date_range("2026-05-01", "2026-09-30", freq="D", tz="Europe/Berlin")
    day_strs = [d.strftime("%Y-%m-%d") for d in all_days]

    all_hours = pd.date_range("2026-05-01 00:00", "2026-09-30 23:00", freq="h", tz="Europe/Berlin")
    hour_strs = [h.strftime("%Y-%m-%d %H:00") for h in all_hours]

    # Wetter-Aggregation
    w_daily = df_w.groupby("date")["out_temp"].agg(["mean", "min", "max"]).reindex(day_strs)
    w_hourly = df_w.groupby("hour")["out_temp"].mean().reindex(all_hours)

    # Schul- und Stockwerksdurchschnitte
    r_daily_school = df_r.groupby("date")["temp"].mean().reindex(day_strs)
    r_hourly_school = df_r.groupby("hour")["temp"].mean().reindex(all_hours)

    floors = ["1. OG", "EG", "UG"]
    floor_stats = {}
    floor_daily = {}
    floor_hourly = {}
    for fl in floors:
        sub = df_r[df_r["floor"] == fl]
        floor_daily[fl] = round_list(sub.groupby("date")["temp"].mean().reindex(day_strs), 2)
        floor_hourly[fl] = round_list(sub.groupby("hour")["temp"].mean().reindex(all_hours), 2)
        floor_stats[fl] = {
            "mean": round(float(sub["temp"].mean()), 2),
            "min": round(float(sub["temp"].min()), 2),
            "max": round(float(sub["temp"].max()), 2),
            "over26_pct": round(float((sub["temp"] > 26.0).mean() * 100), 1),
            "rooms": int(sub["room"].nunique()),
        }

    # Lag-Korrelation (Gesamtschule Stundenwerte vs. Außentemperatur)
    lags = [0, 1, 2, 3, 4, 6, 8, 12, 18, 24]
    lag_corrs = []
    for lag in lags:
        c = r_hourly_school.corr(w_hourly.shift(lag))
        lag_corrs.append({"lag": lag, "r": round(float(c), 4) if pd.notna(c) else None})

    # Raumweise Daten
    rooms_sorted = sorted(df_r["room"].unique())
    sensor_counts = df_r.groupby("room")["name"].nunique().to_dict()
    room_floors = df_r.groupby("room")["floor"].first().to_dict()

    # Tagesmittel und Stundenmittel pro Raum
    r_room_daily = df_r.groupby(["room", "date"])["temp"].mean().unstack(level=0).reindex(day_strs)
    r_room_hourly = df_r.groupby(["room", "hour"])["temp"].mean().unstack(level=0).reindex(all_hours)

    # Monatsmatrix & Gesamtstatistik pro Raum
    month_keys = [m[0] for m in MONTHS if m[0] != "all"]
    room_meta = []
    for rm in rooms_sorted:
        sub = df_r[df_r["room"] == rm]
        d_series = r_room_daily[rm]
        valid_mask = d_series.notna() & w_daily["mean"].notna()
        if valid_mask.sum() >= 5:
            x = w_daily.loc[valid_mask, "mean"].values
            y = d_series.loc[valid_mask].values
            r_val = float(np.corrcoef(x, y)[0, 1])
            slope, intercept = np.polyfit(x, y, 1)
        else:
            r_val, slope, intercept = np.nan, np.nan, np.nan

        monthly = {}
        for mk in month_keys:
            msub = sub[sub["month"] == mk]
            if len(msub) > 0:
                monthly[mk] = {
                    "mean": round(float(msub["temp"].mean()), 2),
                    "min": round(float(msub["temp"].min()), 2),
                    "max": round(float(msub["temp"].max()), 2),
                    "over26_pct": round(float((msub["temp"] > 26.0).mean() * 100), 1),
                }
            else:
                monthly[mk] = None

        room_meta.append({
            "room": rm,
            "floor": room_floors[rm],
            "sensors": int(sensor_counts[rm]),
            "mean": round(float(sub["temp"].mean()), 2),
            "min": round(float(sub["temp"].min()), 2),
            "max": round(float(sub["temp"].max()), 2),
            "over26_pct": round(float((sub["temp"] > 26.0).mean() * 100), 1),
            "r_daily": round(r_val, 3) if not np.isnan(r_val) else None,
            "slope": round(float(slope), 3) if not np.isnan(slope) else None,
            "intercept": round(float(intercept), 2) if not np.isnan(intercept) else None,
            "monthly": monthly,
        })

    # Gesamtkorrelation (Tagesmittel)
    valid_school = r_daily_school.notna() & w_daily["mean"].notna()
    school_r = float(np.corrcoef(w_daily.loc[valid_school, "mean"], r_daily_school.loc[valid_school])[0, 1])
    school_slope, school_intercept = np.polyfit(w_daily.loc[valid_school, "mean"], r_daily_school.loc[valid_school], 1)

    best_lag = max(lag_corrs, key=lambda x: x["r"] or -1)
    warmest_room = max(room_meta, key=lambda x: x["mean"])

    payload = {
        "kpis": {
            "room_mean": round(float(df_r["temp"].mean()), 2),
            "out_mean": round(float(df_w["out_temp"].mean()), 2),
            "out_min": round(float(df_w["out_temp"].min()), 1),
            "out_max": round(float(df_w["out_temp"].max()), 1),
            "school_r": round(school_r, 3),
            "school_r2": round(school_r ** 2, 3),
            "school_slope": round(float(school_slope), 3),
            "school_intercept": round(float(school_intercept), 2),
            "best_lag_h": best_lag["lag"],
            "best_lag_r": best_lag["r"],
            "over26_pct": round(float((df_r["temp"] > 26.0).mean() * 100), 1),
            "og1_over26_pct": floor_stats["1. OG"]["over26_pct"],
            "warmest_room": warmest_room["room"],
            "warmest_mean": warmest_room["mean"],
            "warmest_max": warmest_room["max"],
            "room_count": len(rooms_sorted),
            "sensor_count": int(df_r["name"].nunique()),
            "measurement_count": int(len(df_r)),
        },
        "months": [{"key": k, "label": lbl} for k, lbl in MONTHS],
        "days": day_strs,
        "hours": hour_strs,
        "weather": {
            "daily_mean": round_list(w_daily["mean"], 2),
            "daily_min": round_list(w_daily["min"], 2),
            "daily_max": round_list(w_daily["max"], 2),
            "hourly": round_list(w_hourly, 2),
        },
        "school": {
            "daily": round_list(r_daily_school, 2),
            "hourly": round_list(r_hourly_school, 2),
        },
        "floors": {
            "stats": floor_stats,
            "daily": floor_daily,
            "hourly": floor_hourly,
        },
        "lag_corrs": lag_corrs,
        "rooms_meta": room_meta,
        "rooms_daily": {rm: round_list(r_room_daily[rm], 2) for rm in rooms_sorted},
        "rooms_hourly": {rm: round_list(r_room_hourly[rm], 1) for rm in rooms_sorted},
    }
    return payload


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="de">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Raumtemperatur- & Wetteranalyse · Friedrich-Hölderlin-Schule Asperg</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
  <style>
    :root {
      --bg: #f4f6f9;
      --card: #ffffff;
      --border: #e2e8f0;
      --text: #0f172a;
      --text-sec: #475569;
      --text-muted: #64748b;
      --primary: #1d4ed8;
      --primary-soft: #eff6ff;
      --teal: #0f766e;
      --amber: #d97706;
      --danger: #b91c1c;
      --radius: 10px;
      --shadow: 0 1px 3px rgba(15, 23, 42, 0.06), 0 1px 2px rgba(15, 23, 42, 0.04);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: 'Segoe UI', system-ui, -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
      font-variant-numeric: tabular-nums;
    }
    .container {
      max-width: 1380px;
      margin: 0 auto;
      padding: 28px 24px 48px;
    }
    header {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 22px 26px;
      box-shadow: var(--shadow);
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      margin-bottom: 22px;
    }
    .brand-eyebrow {
      font-size: 12px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--primary);
      margin-bottom: 4px;
    }
    h1 {
      margin: 0;
      font-size: 22px;
      font-weight: 700;
      color: var(--text);
    }
    .subtitle {
      margin: 4px 0 0;
      font-size: 14px;
      color: var(--text-sec);
    }
    .badges {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .badge {
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 6px 12px;
      font-size: 12.5px;
      color: var(--text-sec);
      font-weight: 500;
    }
    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
      margin-bottom: 22px;
    }
    .kpi-card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 16px 18px;
      box-shadow: var(--shadow);
    }
    .kpi-label {
      font-size: 12px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: var(--text-muted);
    }
    .kpi-value {
      font-size: 26px;
      font-weight: 700;
      color: var(--text);
      margin: 6px 0 4px;
    }
    .kpi-sub {
      font-size: 12.5px;
      color: var(--text-sec);
    }
    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 20px 24px;
      box-shadow: var(--shadow);
      margin-bottom: 22px;
    }
    .card-header {
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: flex-start;
      gap: 12px;
      margin-bottom: 16px;
    }
    .card-title {
      margin: 0;
      font-size: 17px;
      font-weight: 700;
      color: var(--text);
    }
    .card-desc {
      margin: 3px 0 0;
      font-size: 13px;
      color: var(--text-sec);
    }
    .controls {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
      background: #f8fafc;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px 14px;
      margin-bottom: 16px;
    }
    .control-group {
      display: flex;
      flex-direction: column;
      gap: 4px;
      min-width: 160px;
    }
    .control-group label {
      font-size: 11.5px;
      font-weight: 600;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    select, button {
      font-family: inherit;
      font-size: 13.5px;
      padding: 7px 10px;
      border: 1px solid #cbd5e1;
      border-radius: 6px;
      background: #ffffff;
      color: var(--text);
      outline: none;
    }
    select:focus {
      border-color: var(--primary);
      box-shadow: 0 0 0 2px rgba(29, 78, 216, 0.12);
    }
    .btn-group {
      display: inline-flex;
      border: 1px solid #cbd5e1;
      border-radius: 6px;
      overflow: hidden;
    }
    .btn-group button {
      border: none;
      border-right: 1px solid #cbd5e1;
      border-radius: 0;
      padding: 7px 12px;
      cursor: pointer;
      background: #ffffff;
      color: var(--text-sec);
      font-weight: 500;
    }
    .btn-group button:last-child { border-right: none; }
    .btn-group button.active {
      background: var(--primary);
      color: #ffffff;
    }
    .mini-stats {
      display: flex;
      flex-wrap: wrap;
      gap: 18px;
      padding: 10px 14px;
      background: var(--primary-soft);
      border: 1px solid #bfdbfe;
      border-radius: 8px;
      margin-bottom: 14px;
      font-size: 13px;
      color: #1e3a8a;
    }
    .mini-stats strong { font-weight: 700; }
    .chart-wrap {
      position: relative;
      height: 380px;
      width: 100%;
    }
    .grid-2 {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(460px, 1fr));
      gap: 22px;
      margin-bottom: 22px;
    }
    .grid-2 .card { margin-bottom: 0; }
    .chart-wrap-sm {
      position: relative;
      height: 320px;
      width: 100%;
    }
    .table-wrap {
      overflow-x: auto;
      max-height: 540px;
      overflow-y: auto;
      border: 1px solid var(--border);
      border-radius: 8px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      text-align: left;
    }
    th, td {
      padding: 9px 12px;
      border-bottom: 1px solid var(--border);
      white-space: nowrap;
    }
    th {
      background: #f8fafc;
      color: var(--text-sec);
      font-weight: 600;
      position: sticky;
      top: 0;
      z-index: 2;
      cursor: pointer;
      user-select: none;
    }
    th:hover { background: #eef2f6; }
    tbody tr { cursor: pointer; transition: background 0.12s; }
    tbody tr:hover { background: #f8fafc; }
    tbody tr.selected-row { outline: 2px solid var(--primary); outline-offset: -2px; }
    .num { text-align: right; }
    .heat-cell {
      text-align: center;
      font-weight: 600;
      border-radius: 4px;
    }
    .floor-pill {
      display: inline-block;
      padding: 2px 8px;
      border-radius: 999px;
      font-size: 11.5px;
      font-weight: 600;
    }
    .floor-1OG { background: #fee2e2; color: #991b1b; }
    .floor-EG  { background: #fef3c7; color: #92400e; }
    .floor-UG  { background: #e0f2fe; color: #075985; }
    footer {
      font-size: 12.5px;
      color: var(--text-muted);
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 16px 20px;
    }
    #offline-warn {
      display: none;
      background: #fef2f2;
      border: 1px solid #fecaca;
      color: #991b1b;
      padding: 12px 16px;
      border-radius: 8px;
      margin-bottom: 16px;
      font-size: 13.5px;
    }
    @media print {
      body { background: #fff; }
      .controls { display: none; }
      .card, header { box-shadow: none; break-inside: avoid; }
    }
  </style>
</head>
<body>
  <div class="container">
    <div id="offline-warn">
      Hinweis: Chart.js konnte nicht geladen werden (keine Internetverbindung). Tabellen und Kennzahlen stehen vollständig zur Verfügung; für die interaktiven Diagramme ist eine Internetverbindung erforderlich.
    </div>

    <header>
      <div>
        <div class="brand-eyebrow">Stadt Asperg · Gebäudemanagement &amp; Raumklima-Auswertung</div>
        <h1>Raumtemperaturverlauf &amp; Wetterkorrelation — Friedrich-Hölderlin-Schule</h1>
        <p class="subtitle">Auswertung der Raumtemperaturen (Mai – September 2026) im Vergleich zur lokalen Wetterstation (Garten)</p>
      </div>
      <div class="badges" id="header-badges"></div>
    </header>

    <section class="kpi-grid" id="kpi-grid"></section>

    <section class="card">
      <div class="card-header">
        <div>
          <h2 class="card-title">1. Temperaturverlauf pro Raum und Monat</h2>
          <p class="card-desc">Vergleich der gemessenen Raumtemperatur mit der Außentemperatur (Wetterstation Garten) sowie dem Richtwert 26 °C (ASR A3.5).</p>
        </div>
      </div>

      <div class="controls">
        <div class="control-group">
          <label for="sel-month">Zeitraum / Monat</label>
          <select id="sel-month"></select>
        </div>
        <div class="control-group">
          <label for="sel-floor">Stockwerk-Filter</label>
          <select id="sel-floor">
            <option value="ALL">Alle Stockwerke</option>
            <option value="1. OG">1. OG</option>
            <option value="EG">EG</option>
            <option value="UG">UG</option>
          </select>
        </div>
        <div class="control-group" style="min-width: 280px; flex: 1;">
          <label for="sel-room">Raum / Vergleichsebene</label>
          <select id="sel-room"></select>
        </div>
        <div class="control-group">
          <label>Zeitliche Auflösung</label>
          <div class="btn-group" id="res-toggle">
            <button type="button" data-res="daily" class="active">Tagesmittel</button>
            <button type="button" data-res="hourly">Stundenwerte</button>
          </div>
        </div>
      </div>

      <div class="mini-stats" id="selection-stats"></div>
      <div class="chart-wrap">
        <canvas id="timeseries-chart"></canvas>
      </div>
    </section>

    <section class="card">
      <div class="card-header">
        <div>
          <h2 class="card-title">2. Monatsübersicht aller 35 Räume (Mai – September 2026)</h2>
          <p class="card-desc">Klicken Sie auf eine Raumzeile, um den Verlauf oben direkt anzuzeigen. Spaltenköpfe sind sortierbar.</p>
        </div>
        <div class="btn-group" id="heatmap-mode">
          <button type="button" data-mode="mean" class="active">Ø Temp (°C)</button>
          <button type="button" data-mode="max">Max Temp (°C)</button>
          <button type="button" data-mode="over26_pct">Anteil &gt; 26 °C (%)</button>
        </div>
      </div>
      <div class="table-wrap">
        <table id="matrix-table">
          <thead>
            <tr>
              <th data-sort="room">Raum</th>
              <th data-sort="floor">Stockwerk</th>
              <th data-sort="m_2026-05" class="num">Mai 26</th>
              <th data-sort="m_2026-06" class="num">Jun 26</th>
              <th data-sort="m_2026-07" class="num">Jul 26</th>
              <th data-sort="m_2026-08" class="num">Aug 26</th>
              <th data-sort="m_2026-09" class="num">Sep 26</th>
              <th data-sort="mean" class="num">Gesamt Ø</th>
              <th data-sort="max" class="num">Max</th>
              <th data-sort="over26_pct" class="num">&gt; 26 °C</th>
              <th data-sort="r_daily" class="num">Korrelation r</th>
              <th data-sort="slope" class="num">Steigung ΔT</th>
            </tr>
          </thead>
          <tbody id="matrix-body"></tbody>
        </table>
      </div>
    </section>

    <div class="grid-2">
      <section class="card">
        <div class="card-header">
          <div>
            <h2 class="card-title">3. Korrelation: Außen- vs. Raumtemperatur</h2>
            <p class="card-desc" id="scatter-desc">Tagesmittelwerte (Mai–Sep 2026) inkl. linearer Regressionsgerade für die aktuelle Raumauswahl.</p>
          </div>
        </div>
        <div class="chart-wrap-sm">
          <canvas id="scatter-chart"></canvas>
        </div>
      </section>

      <section class="card">
        <div class="card-header">
          <div>
            <h2 class="card-title">4. Thermische Trägheit &amp; Stockwerksvergleich</h2>
            <p class="card-desc">Zeitversatz-Korrelation (Stundenwerte) zeigt die Verzögerung der Gebäudeaufheizung.</p>
          </div>
          <div class="btn-group" id="right-chart-toggle">
            <button type="button" data-view="lag" class="active">Zeitversatz (Lag)</button>
            <button type="button" data-view="floors">Stockwerke</button>
          </div>
        </div>
        <div class="chart-wrap-sm">
          <canvas id="right-chart"></canvas>
        </div>
      </section>
    </div>

    <footer>
      <strong>Methodik &amp; Datenhinweise:</strong>
      Ausgewertet wurden <strong>__MEAS_COUNT__</strong> gültige Temperaturmessungen (Spalte E) aus <strong>__SENSOR_COUNT__</strong> Thermostaten/Sensoren, aggregiert auf <strong>__ROOM_COUNT__</strong> Räume (Zeitraum 01.05.2026 – 30.09.2026, Zeitzone Europe/Berlin).
      Die Außentemperatur stammt aus <code>Wetterdaten_Tobi-Wolf_Wetterstation-Garten.csv</code> (Kategorie <code>actualTemperature</code>).
      Hinweis: Einzelne Spitzenwerte der Wetterstation bis __OUT_MAX__ °C deuten auf direkte Sonneneinstrahlung am Außensensor hin; die Tagesmittelwerte liefern hierfür ein robustes Korrelationsmaß.
      Die Steigung <em>ΔT</em> gibt an, um wie viel °C die Raumtemperatur im Mittel steigt, wenn das Tagesmittel der Außentemperatur um +1 °C zunimmt.
    </footer>
  </div>

  <script>
    const DATA = __PAYLOAD_JSON__;

    if (typeof Chart === 'undefined') {
      document.getElementById('offline-warn').style.display = 'block';
    }

    // Zustand
    let state = {
      month: 'all',
      floor: 'ALL',
      target: '__SCHOOL__', // '__SCHOOL__', '__FLOOR__1. OG', oder Raumname
      res: 'daily',
      heatMode: 'mean',
      sortKey: 'mean',
      sortAsc: false,
      rightView: 'lag'
    };

    let tsChart = null;
    let scatterChart = null;
    let rightChart = null;

    function fmt(val, dec = 1, suffix = '') {
      if (val === null || val === undefined || Number.isNaN(val)) return '–';
      return val.toFixed(dec).replace('.', ',') + suffix;
    }

    function initHeaderAndKPIs() {
      const k = DATA.kpis;
      document.getElementById('header-badges').innerHTML = `
        <span class="badge">Zeitraum: 01.05. – 30.09.2026</span>
        <span class="badge">${k.room_count} Räume (${k.sensor_count} Sensoren)</span>
        <span class="badge">${k.measurement_count.toLocaleString('de-DE')} Messwerte</span>
      `;

      document.getElementById('kpi-grid').innerHTML = `
        <div class="kpi-card">
          <div class="kpi-label">Ø Raumtemperatur</div>
          <div class="kpi-value">${fmt(k.room_mean, 2, ' °C')}</div>
          <div class="kpi-sub">Über alle ${k.room_count} Räume (Mai–Sep)</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Ø Außentemperatur</div>
          <div class="kpi-value">${fmt(k.out_mean, 2, ' °C')}</div>
          <div class="kpi-sub">Min ${fmt(k.out_min, 1, ' °C')} · Max ${fmt(k.out_max, 1, ' °C')}</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Wetter-Korrelation (r)</div>
          <div class="kpi-value">${fmt(k.school_r, 3)}</div>
          <div class="kpi-sub">R² = ${fmt(k.school_r2, 2)} · Optimum bei ${k.best_lag_h}h Zeitversatz</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Messwerte &gt; 26 °C</div>
          <div class="kpi-value">${fmt(k.over26_pct, 1, ' %')}</div>
          <div class="kpi-sub">Im 1. OG deutlich höher: <strong>${fmt(k.og1_over26_pct, 1, ' %')}</strong></div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Wärmster Raum (Ø)</div>
          <div class="kpi-value">${fmt(k.warmest_mean, 2, ' °C')}</div>
          <div class="kpi-sub">${k.warmest_room} (Max ${fmt(k.warmest_max, 1, ' °C')})</div>
        </div>
      `;
    }

    function initSelectors() {
      const selMonth = document.getElementById('sel-month');
      DATA.months.forEach(m => {
        const opt = document.createElement('option');
        opt.value = m.key;
        opt.textContent = m.label;
        selMonth.appendChild(opt);
      });
      selMonth.addEventListener('change', e => {
        state.month = e.target.value;
        updateAllCharts();
      });

      document.getElementById('sel-floor').addEventListener('change', e => {
        state.floor = e.target.value;
        populateRoomSelect();
        updateAllCharts();
      });

      document.getElementById('sel-room').addEventListener('change', e => {
        state.target = e.target.value;
        renderMatrix();
        updateAllCharts();
      });

      document.querySelectorAll('#res-toggle button').forEach(btn => {
        btn.addEventListener('click', () => {
          document.querySelectorAll('#res-toggle button').forEach(b => b.classList.remove('active'));
          btn.classList.add('active');
          state.res = btn.dataset.res;
          updateTimeSeries();
        });
      });

      document.querySelectorAll('#heatmap-mode button').forEach(btn => {
        btn.addEventListener('click', () => {
          document.querySelectorAll('#heatmap-mode button').forEach(b => b.classList.remove('active'));
          btn.classList.add('active');
          state.heatMode = btn.dataset.mode;
          renderMatrix();
        });
      });

      document.querySelectorAll('#right-chart-toggle button').forEach(btn => {
        btn.addEventListener('click', () => {
          document.querySelectorAll('#right-chart-toggle button').forEach(b => b.classList.remove('active'));
          btn.classList.add('active');
          state.rightView = btn.dataset.view;
          updateRightChart();
        });
      });

      document.querySelectorAll('#matrix-table th[data-sort]').forEach(th => {
        th.addEventListener('click', () => {
          const key = th.dataset.sort;
          if (state.sortKey === key) state.sortAsc = !state.sortAsc;
          else { state.sortKey = key; state.sortAsc = (key === 'room' || key === 'floor'); }
          renderMatrix();
        });
      });

      populateRoomSelect();
    }

    function populateRoomSelect() {
      const selRoom = document.getElementById('sel-room');
      const prev = state.target;
      selRoom.innerHTML = '';

      const grpAvg = document.createElement('optgroup');
      grpAvg.label = 'Durchschnittswerte';
      const optSchool = new Option('Gesamtschule (Durchschnitt aller 35 Räume)', '__SCHOOL__');
      grpAvg.appendChild(optSchool);
      ['1. OG', 'EG', 'UG'].forEach(fl => {
        if (state.floor === 'ALL' || state.floor === fl) {
          grpAvg.appendChild(new Option(`Stockwerk ${fl} (Durchschnitt)`, `__FLOOR__${fl}`));
        }
      });
      selRoom.appendChild(grpAvg);

      const grpRooms = document.createElement('optgroup');
      grpRooms.label = 'Einzelne Räume';
      const filtered = DATA.rooms_meta.filter(r => state.floor === 'ALL' || r.floor === state.floor);
      filtered.forEach(r => {
        grpRooms.appendChild(new Option(`${r.room} (${r.sensors} Sensoren)`, r.room));
      });
      selRoom.appendChild(grpRooms);

      const validValues = Array.from(selRoom.options).map(o => o.value);
      if (validValues.includes(prev)) selRoom.value = prev;
      else {
        selRoom.value = state.floor === 'ALL' ? '__SCHOOL__' : `__FLOOR__${state.floor}`;
        state.target = selRoom.value;
      }
    }

    function getTargetSeries(res) {
      if (state.target === '__SCHOOL__') {
        return { label: 'Gesamtschule Ø', values: DATA.school[res] };
      }
      if (state.target.startsWith('__FLOOR__')) {
        const fl = state.target.replace('__FLOOR__', '');
        return { label: `Stockwerk ${fl} Ø`, values: DATA.floors[res][fl] };
      }
      const map = res === 'daily' ? DATA.rooms_daily : DATA.rooms_hourly;
      return { label: state.target, values: map[state.target] || [] };
    }

    function pearson(xs, ys) {
      const pairs = [];
      for (let i = 0; i < xs.length; i++) {
        if (xs[i] !== null && ys[i] !== null && !Number.isNaN(xs[i]) && !Number.isNaN(ys[i])) {
          pairs.push([xs[i], ys[i]]);
        }
      }
      if (pairs.length < 3) return { r: null, slope: null, intercept: null, pairs };
      const n = pairs.length;
      let sx = 0, sy = 0;
      for (const [x, y] of pairs) { sx += x; sy += y; }
      const mx = sx / n, my = sy / n;
      let num = 0, dx2 = 0, dy2 = 0;
      for (const [x, y] of pairs) {
        const dx = x - mx, dy = y - my;
        num += dx * dy; dx2 += dx * dx; dy2 += dy * dy;
      }
      const denom = Math.sqrt(dx2 * dy2);
      const r = denom > 0 ? num / denom : null;
      const slope = dx2 > 0 ? num / dx2 : null;
      const intercept = slope !== null ? my - slope * mx : null;
      return { r, slope, intercept, pairs };
    }

    function updateTimeSeries() {
      const res = state.res;
      const labelsAll = res === 'daily' ? DATA.days : DATA.hours;
      const outAll = res === 'daily' ? DATA.weather.daily_mean : DATA.weather.hourly;
      const target = getTargetSeries(res);

      const indices = [];
      for (let i = 0; i < labelsAll.length; i++) {
        if (state.month === 'all' || labelsAll[i].startsWith(state.month)) {
          indices.push(i);
        }
      }

      const labels = indices.map(i => {
        const s = labelsAll[i];
        return res === 'daily' ? s.slice(8, 10) + '.' + s.slice(5, 7) + '.' : s.slice(8, 10) + '.' + s.slice(5, 7) + '. ' + s.slice(11, 16);
      });
      const rVals = indices.map(i => target.values[i]);
      const wVals = indices.map(i => outAll[i]);
      const ref26 = indices.map(() => 26.0);

      // Kennzahlen für Auswahl
      const validR = rVals.filter(v => v !== null);
      const validW = wVals.filter(v => v !== null);
      const rMean = validR.length ? validR.reduce((a, b) => a + b, 0) / validR.length : null;
      const rMin = validR.length ? Math.min(...validR) : null;
      const rMax = validR.length ? Math.max(...validR) : null;
      const over26 = validR.length ? (validR.filter(v => v > 26).length / validR.length) * 100 : null;
      const wMean = validW.length ? validW.reduce((a, b) => a + b, 0) / validW.length : null;
      const corr = pearson(wVals, rVals);

      document.getElementById('selection-stats').innerHTML = `
        <span>Auswahl: <strong>${target.label}</strong></span>
        <span>Ø Raum: <strong>${fmt(rMean, 2, ' °C')}</strong></span>
        <span>Min / Max: <strong>${fmt(rMin, 1, ' °C')} / ${fmt(rMax, 1, ' °C')}</strong></span>
        <span>&gt; 26 °C: <strong>${fmt(over26, 1, ' %')}</strong></span>
        <span>Ø Außen: <strong>${fmt(wMean, 2, ' °C')}</strong></span>
        <span>Korrelation r: <strong>${fmt(corr.r, 3)}</strong></span>
      `;

      if (typeof Chart === 'undefined') return;
      const ctx = document.getElementById('timeseries-chart').getContext('2d');
      if (tsChart) tsChart.destroy();
      tsChart = new Chart(ctx, {
        type: 'line',
        data: {
          labels,
          datasets: [
            {
              label: target.label + ' (°C)',
              data: rVals,
              borderColor: '#1d4ed8',
              backgroundColor: 'rgba(29, 78, 216, 0.08)',
              borderWidth: res === 'daily' ? 2.2 : 1.4,
              pointRadius: res === 'daily' && labels.length <= 35 ? 2.5 : 0,
              tension: 0.2,
              fill: true,
              spanGaps: true
            },
            {
              label: 'Außentemperatur Garten (°C)',
              data: wVals,
              borderColor: '#d97706',
              borderWidth: res === 'daily' ? 1.8 : 1.2,
              borderDash: [5, 4],
              pointRadius: 0,
              tension: 0.2,
              fill: false,
              spanGaps: true
            },
            {
              label: 'Richtwert 26 °C (ASR A3.5)',
              data: ref26,
              borderColor: '#b91c1c',
              borderWidth: 1.2,
              borderDash: [2, 4],
              pointRadius: 0,
              fill: false
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          interaction: { mode: 'index', intersect: false },
          plugins: {
            legend: { position: 'top', labels: { usePointStyle: true, boxWidth: 10 } },
            tooltip: {
              callbacks: {
                label: ctx => `${ctx.dataset.label}: ${fmt(ctx.parsed.y, 2, ' °C')}`
              }
            }
          },
          scales: {
            x: { ticks: { maxTicksLimit: 16, color: '#64748b' }, grid: { display: false } },
            y: {
              title: { display: true, text: 'Temperatur (°C)', color: '#475569' },
              grid: { color: '#f1f5f9' },
              ticks: { color: '#64748b' }
            }
          }
        }
      });
    }

    function updateScatter() {
      if (typeof Chart === 'undefined') return;
      const target = getTargetSeries('daily');
      const labelsAll = DATA.days;
      const outAll = DATA.weather.daily_mean;

      const xs = [], ys = [], pts = [];
      for (let i = 0; i < labelsAll.length; i++) {
        if (state.month === 'all' || labelsAll[i].startsWith(state.month)) {
          if (outAll[i] !== null && target.values[i] !== null) {
            xs.push(outAll[i]);
            ys.push(target.values[i]);
            pts.push({ x: outAll[i], y: target.values[i], date: labelsAll[i] });
          }
        }
      }

      const stats = pearson(xs, ys);
      const desc = document.getElementById('scatter-desc');
      if (stats.r !== null) {
        desc.innerHTML = `<strong>${target.label}</strong>: r = <strong>${fmt(stats.r, 3)}</strong> (R² = ${fmt(stats.r * stats.r, 2)}) · Steigung: <strong>+${fmt(stats.slope, 2)} °C</strong> Raum je +1 °C Außen.`;
      } else {
        desc.textContent = 'Nicht genügend Datenpunkte für Regression.';
      }

      let regLine = [];
      if (stats.slope !== null && xs.length > 0) {
        const minX = Math.min(...xs), maxX = Math.max(...xs);
        regLine = [
          { x: minX, y: stats.intercept + stats.slope * minX },
          { x: maxX, y: stats.intercept + stats.slope * maxX }
        ];
      }

      const ctx = document.getElementById('scatter-chart').getContext('2d');
      if (scatterChart) scatterChart.destroy();
      scatterChart = new Chart(ctx, {
        type: 'scatter',
        data: {
          datasets: [
            {
              label: 'Tagesmittel (Außen vs. Raum)',
              data: pts,
              backgroundColor: 'rgba(29, 78, 216, 0.55)',
              borderColor: '#1d4ed8',
              pointRadius: 4
            },
            {
              label: 'Lineare Regression',
              type: 'line',
              data: regLine,
              borderColor: '#b91c1c',
              borderWidth: 2,
              pointRadius: 0,
              fill: false
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { position: 'top' },
            tooltip: {
              callbacks: {
                label: ctx => {
                  const raw = ctx.raw;
                  if (raw.date) return `${raw.date}: Außen ${fmt(raw.x, 1, ' °C')} → Raum ${fmt(raw.y, 1, ' °C')}`;
                  return `Trend: ${fmt(raw.y, 1, ' °C')}`;
                }
              }
            }
          },
          scales: {
            x: { title: { display: true, text: 'Außentemperatur Tagesmittel (°C)', color: '#475569' }, grid: { color: '#f1f5f9' } },
            y: { title: { display: true, text: 'Raumtemperatur Tagesmittel (°C)', color: '#475569' }, grid: { color: '#f1f5f9' } }
          }
        }
      });
    }

    function updateRightChart() {
      if (typeof Chart === 'undefined') return;
      const ctx = document.getElementById('right-chart').getContext('2d');
      if (rightChart) rightChart.destroy();

      if (state.rightView === 'lag') {
        rightChart = new Chart(ctx, {
          type: 'bar',
          data: {
            labels: DATA.lag_corrs.map(d => d.lag + ' h'),
            datasets: [{
              label: 'Pearson-Korrelation r (Stundenwerte)',
              data: DATA.lag_corrs.map(d => d.r),
              backgroundColor: DATA.lag_corrs.map(d => d.lag === DATA.kpis.best_lag_h ? '#1d4ed8' : '#93c5fd'),
              borderRadius: 4
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
              legend: { display: false },
              tooltip: { callbacks: { label: ctx => `r = ${fmt(ctx.parsed.y, 4)}` } }
            },
            scales: {
              x: { title: { display: true, text: 'Zeitversatz der Außentemperatur (Stunden)', color: '#475569' }, grid: { display: false } },
              y: { min: 0.4, max: 0.8, title: { display: true, text: 'Korrelation r', color: '#475569' }, grid: { color: '#f1f5f9' } }
            }
          }
        });
      } else {
        const fls = ['UG', 'EG', '1. OG'];
        const stats = fls.map(f => DATA.floors.stats[f]);
        rightChart = new Chart(ctx, {
          type: 'bar',
          data: {
            labels: fls,
            datasets: [
              {
                label: 'Ø Temperatur (°C)',
                data: stats.map(s => s.mean),
                backgroundColor: '#1d4ed8',
                borderRadius: 4,
                yAxisID: 'y'
              },
              {
                label: 'Anteil > 26 °C (%)',
                data: stats.map(s => s.over26_pct),
                backgroundColor: '#d97706',
                borderRadius: 4,
                yAxisID: 'y1'
              }
            ]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
              y: { position: 'left', min: 15, title: { display: true, text: 'Ø Temperatur (°C)' }, grid: { color: '#f1f5f9' } },
              y1: { position: 'right', min: 0, max: 45, title: { display: true, text: 'Anteil > 26 °C (%)' }, grid: { display: false } }
            }
          }
        });
      }
    }

    function heatColor(val, mode) {
      if (val === null || val === undefined) return 'transparent';
      let t = 0;
      if (mode === 'mean') t = Math.max(0, Math.min(1, (val - 19) / (28 - 19)));
      else if (mode === 'max') t = Math.max(0, Math.min(1, (val - 22) / (36 - 22)));
      else t = Math.max(0, Math.min(1, val / 60));

      // Dezente seriöse Skala: helles Eisblau -> neutrales Creme -> sanftes Terrakotta
      if (t < 0.5) {
        const k = t * 2;
        const r = Math.round(239 + (254 - 239) * k);
        const g = Math.round(246 + (243 - 246) * k);
        const b = Math.round(255 + (199 - 255) * k);
        return `rgb(${r}, ${g}, ${b})`;
      } else {
        const k = (t - 0.5) * 2;
        const r = Math.round(254 + (254 - 254) * k);
        const g = Math.round(243 + (202 - 243) * k);
        const b = Math.round(199 + (202 - 199) * k);
        return `rgb(${r}, ${g}, ${b})`;
      }
    }

    function renderMatrix() {
      const mode = state.heatMode;
      const monthKeys = ['2026-05', '2026-06', '2026-07', '2026-08', '2026-09'];

      const rows = [...DATA.rooms_meta].sort((a, b) => {
        const k = state.sortKey;
        let va, vb;
        if (k.startsWith('m_')) {
          const mk = k.slice(2);
          va = a.monthly[mk] ? a.monthly[mk][mode] : -999;
          vb = b.monthly[mk] ? b.monthly[mk][mode] : -999;
        } else {
          va = a[k] ?? -999;
          vb = b[k] ?? -999;
        }
        if (typeof va === 'string') return state.sortAsc ? va.localeCompare(vb, 'de') : vb.localeCompare(va, 'de');
        return state.sortAsc ? va - vb : vb - va;
      });

      const tbody = document.getElementById('matrix-body');
      tbody.innerHTML = '';
      rows.forEach(r => {
        const tr = document.createElement('tr');
        if (state.target === r.room) tr.classList.add('selected-row');
        tr.addEventListener('click', () => {
          state.floor = 'ALL';
          document.getElementById('sel-floor').value = 'ALL';
          populateRoomSelect();
          state.target = r.room;
          document.getElementById('sel-room').value = r.room;
          renderMatrix();
          updateAllCharts();
          window.scrollTo({ top: 180, behavior: 'smooth' });
        });

        const flClass = r.floor === '1. OG' ? 'floor-1OG' : (r.floor === 'EG' ? 'floor-EG' : 'floor-UG');
        let cells = `
          <td><strong>${r.room}</strong> <span style="color:#64748b;font-size:11.5px;">(${r.sensors})</span></td>
          <td><span class="floor-pill ${flClass}">${r.floor}</span></td>
        `;
        monthKeys.forEach(mk => {
          const m = r.monthly[mk];
          const val = m ? m[mode] : null;
          const bg = heatColor(val, mode);
          const dec = mode === 'over26_pct' ? 1 : 1;
          const suf = mode === 'over26_pct' ? '%' : '°';
          cells += `<td class="heat-cell" style="background:${bg};">${fmt(val, dec, suf)}</td>`;
        });
        cells += `
          <td class="num"><strong>${fmt(r.mean, 2, ' °C')}</strong></td>
          <td class="num">${fmt(r.max, 1, ' °C')}</td>
          <td class="num">${fmt(r.over26_pct, 1, ' %')}</td>
          <td class="num"><strong>${fmt(r.r_daily, 3)}</strong></td>
          <td class="num">+${fmt(r.slope, 2)}</td>
        `;
        tr.innerHTML = cells;
        tbody.appendChild(tr);
      });
    }

    function updateAllCharts() {
      updateTimeSeries();
      updateScatter();
    }

    initHeaderAndKPIs();
    initSelectors();
    renderMatrix();
    updateAllCharts();
    updateRightChart();
  </script>
</body>
</html>
"""


def main() -> None:
    print("Lade Raumtemperaturdaten...")
    df_r = load_room_data(ROOM_CSV)
    print(f"  -> {len(df_r)} gültige Messwerte in {df_r['room'].nunique()} Räumen.")

    print("Lade Wetterdaten...")
    df_w = load_weather_data(WEATHER_CSV)
    print(f"  -> {len(df_w)} Wetter-Messwerte im Zeitraum Mai–Sep 2026.")

    print("Berechne Korrelationen und Aggregationen...")
    payload = build_payload(df_r, df_w)

    html = HTML_TEMPLATE
    html = html.replace("__PAYLOAD_JSON__", json.dumps(payload, ensure_ascii=False))
    html = html.replace("__MEAS_COUNT__", f"{payload['kpis']['measurement_count']:,}".replace(",", "."))
    html = html.replace("__SENSOR_COUNT__", str(payload["kpis"]["sensor_count"]))
    html = html.replace("__ROOM_COUNT__", str(payload["kpis"]["room_count"]))
    html = html.replace("__OUT_MAX__", str(payload["kpis"]["out_max"]).replace(".", ","))

    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print(f"Bericht erfolgreich erstellt: {OUTPUT_HTML} ({OUTPUT_HTML.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
