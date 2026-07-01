# SeismoTrack — Earthquake Monitoring Dashboard

An interactive earthquake data visualization and analysis dashboard built with **Dash** and **Plotly**. SeismoTrack aggregates seismic data from **USGS** and **EMSC**, detects Indonesian provinces from coordinates, and provides real-time filters, statistical summaries, regional analysis, and emergency preparedness tools.

## Features

- **Interactive Map** — Scatter mapbox visualization of earthquake epicenters colored by magnitude, with click-to-zoom and pan controls
- **Multi-source Data** — Combines earthquake records from USGS and EMSC into a unified dataset
- **Province Detection** — Automatically maps earthquake coordinates to Indonesian provinces using Ball Tree nearest-neighbor search (via worldcities dataset)
- **Flexible Filtering** — Filter by province, magnitude range, year (multi-select or range), and click data on the map
- **Statistics Overview** — Total earthquakes, average magnitude, deepest and shallowest events
- **Data Export** — Download filtered earthquake data as CSV with one click
- **Frequency & Depth Analysis** — Magnitude distribution histogram and magnitude vs. depth scatter plot
- **Regional Summary** — Average magnitude by province with sortable bar charts
- **Safety & Emergency Page** — Earthquake safety tips, evacuation shelter map with dynamic pin addition, and a curated article library

## Pages

| Route | Page |
|-------|------|
| `/overview` | Interactive map, statistics cards, filter panel, and data table |
| `/analysis` | Magnitude distribution and magnitude-depth correlation charts |
| `/regional` | Average magnitude per province bar chart |
| `/settings` | Earthquake safety tips, evacuation shelter map, and reference articles |
| `/help` | Usage guide and support |

## Data Sources

- **USGS** — United States Geological Survey earthquake catalog (CSV exports)
- **EMSC** — European-Mediterranean Seismological Centre earthquake data (CSV exports)
- **BMKG** — Real-time earthquake feed from Indonesian Meteorological Agency (via `gempa-realtime` companion project)

Data is pre-processed and combined using `combine_data.py`, which standardizes column names, deduplicates records, and merges all sources into a single CSV at `data/combined/combined.csv`.

## Tech Stack

- **Python** 3.8+
- **Dash** — Web framework for analytical apps
- **Plotly** — Interactive charting library
- **Dash Bootstrap Components** — UI components and styling
- **Pandas** — Data manipulation
- **NumPy** — Numerical operations
- **Scikit-learn** — Ball Tree for province detection
- **OpenStreetMap** — Map tiles via Plotly mapbox

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/Pengembangan-Aplikasi-Gempa.git
cd Pengembangan-Aplikasi-Gempa
pip install dash dash-bootstrap-components plotly pandas numpy scikit-learn
```

### Prepare Data

Place your USGS CSV files in `data/USGS/` and EMSC CSV files in `data/EMSC/`, then run:

```bash
python combine_data.py
```

This generates `data/combined/combined.csv` used by the dashboard.

### Run the Dashboard

```bash
python dashv2.py
```

Open `http://127.0.0.1:8050` in your browser.

## Project Structure

```
├── dashv2.py              # Main dashboard application (v2)
├── gempa_dash.py           # Dashboard prototype (v1)
├── combine_data.py         # Data merging script (USGS + EMSC)
├── data/
│   ├── combined/
│   │   └── combined.csv    # Merged earthquake dataset
│   ├── USGS/               # USGS CSV exports
│   ├── EMSC/               # EMSC CSV exports
│   └── worldcities.csv     # City-to-province mapping
└── README.md
```

## Companion Project — `gempa-realtime`

A separate automated pipeline that fetches the latest earthquake data from BMKG every 10 minutes via GitHub Actions. It stores real-time Indonesian earthquake data used alongside the global USGS/EMSC dataset.

## License

MIT
