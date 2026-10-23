from pathlib import Path
import pandas as pd

class DataIngestionError(ValueError):
    """Raised when an input dataset cannot be ingested safely."""

def load_dataset(path: str | Path, timestamp_column: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise DataIngestionError(f"Input file does not exist: {path}")
    if path.suffix.lower() != ".csv":
        raise DataIngestionError("Only CSV input is supported by this starter.")
    try:
        df = pd.read_csv(path)
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise DataIngestionError(f"Unable to read CSV: {exc}") from exc
    if df.empty:
        raise DataIngestionError("Input dataset is empty.")
    if timestamp_column not in df.columns:
        raise DataIngestionError(f"Missing timestamp column: {timestamp_column}")
    if "source_record_id" in df.columns:
        raise DataIngestionError("Input already contains reserved column 'source_record_id'.")
    df.insert(0, "source_record_id", [f"ROW-{i:08d}" for i in range(len(df))])
    df["timestamp_original"] = df[timestamp_column].copy()
    # Parse to UTC only when timestamps carry timezone information; naive timestamps
    # are retained as naive so we do not silently assume a source timezone.
    parsed = pd.to_datetime(df[timestamp_column], errors="coerce")
    df["timestamp_parsed"] = parsed
    return df
