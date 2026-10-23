import json
from pathlib import Path
from datetime import datetime, timezone

def write_audit(records: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            payload = {
                "audit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "record": record,
            }
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
