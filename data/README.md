# Data

The raw data is **not** included in this repository (it is confidential challenge material and ~130 MB).

Place it like this before running:

```
data/
├── raw/
│   └── DataScienceChallenge/
│       ├── ps1/                    # 73 CSVs, 2000-2020, known latitude (training)
│       ├── ps2/                    # 35 CSVs, 2020-2021, latitude = -9999 (prediction)
│       └── columns_description.csv
└── processed/                      # written by the pipeline (station-month feature tables)
```

Or point the code elsewhere: `./run.sh /path/to/DataScienceChallenge`, `make run DATA_DIR=...`,
or `DATA_DIR=... python -m latpred.pipeline`.

## Schema notes
* One file per station and period; readings every 4 hours at 00, 04, 08, 12, 16, 20 UTC.
* Missing values are encoded as `-9999`. In PS2, *all* location fields (region, state, station,
  station code, latitude, longitude, height) are `-9999`, not only latitude.
* Column meanings and short codes are in `columns_description.csv`; the code maps the Portuguese
  headers to those short codes (`src/latpred/config.py`, `RAW_COLUMNS`).
