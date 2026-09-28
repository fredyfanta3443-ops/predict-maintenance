"""Step 6b: the work-order system the nightly job posts to.

Run:  uv run uvicorn api:app --app-dir src --reload
Docs: http://127.0.0.1:8000/docs
"""
import os
import sqlite3
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Response, status
from pydantic import BaseModel, Field

DB_PATH = Path(os.environ.get("WORKORDER_DB", Path(__file__).resolve().parent.parent / "workorders.db"))

class WorkOrderIn(BaseModel):
    unit: int = Field(..., description="Engine id")
    predicted_rul: float = Field(..., ge=0, description="Predicted flights left")
    last_cycle: int = Field(..., ge=1, description="Last flight the prediction is based on")
    threshold: int = Field(..., description="Policy k: service when predicted_rul < k")
    model_version: str
    run_date: str = Field(..., description="Batch run date, YYYY-MM-DD")


class WorkOrder(WorkOrderIn):
    id: int
    status: Literal["open", "done"]
    created_at: str


@contextmanager
def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS work_orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                unit INTEGER NOT NULL,
                predicted_rul REAL NOT NULL,
                last_cycle INTEGER NOT NULL,
                threshold INTEGER NOT NULL,
                model_version TEXT NOT NULL,
                run_date TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'open',
                created_at TEXT NOT NULL
            )""")
        # At most one open order per engine, so re-running the nightly job can't create duplicates.
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS one_open_per_unit ON work_orders(unit) WHERE status = 'open'")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    yield


app = FastAPI(title="Engine work orders", lifespan=lifespan)


def _get(conn, order_id: int) -> WorkOrder:
    row = conn.execute("SELECT * FROM work_orders WHERE id = ?", (order_id,)).fetchone()
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "work order not found")
    return WorkOrder(**dict(row))


@app.post("/work-orders", response_model=WorkOrder, status_code=status.HTTP_201_CREATED)
def create_work_order(order: WorkOrderIn, response: Response):
    """Create a service order. If the engine already has an open one, return it (200) instead."""
    with db() as conn:
        existing = conn.execute("SELECT id FROM work_orders WHERE unit = ? AND status = 'open'", (order.unit,)).fetchone()
        if existing:
            response.status_code = status.HTTP_200_OK
            return _get(conn, existing["id"])
        cur = conn.execute(
            "INSERT INTO work_orders (unit, predicted_rul, last_cycle, threshold, model_version, run_date, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (order.unit, order.predicted_rul, order.last_cycle, order.threshold, order.model_version,
             order.run_date, datetime.now(timezone.utc).isoformat(timespec="seconds")),
        )
        return _get(conn, cur.lastrowid)


@app.get("/work-orders", response_model=list[WorkOrder])
def list_work_orders(status_filter: Literal["open", "done"] | None = None):
    with db() as conn:
        if status_filter:
            rows = conn.execute("SELECT * FROM work_orders WHERE status = ? ORDER BY predicted_rul", (status_filter,))
        else:
            rows = conn.execute("SELECT * FROM work_orders ORDER BY id")
        return [WorkOrder(**dict(r)) for r in rows]


@app.post("/work-orders/{order_id}/complete", response_model=WorkOrder)
def complete_work_order(order_id: int):
    """Technician marks the service done. The engine can get a new order after this."""
    with db() as conn:
        _get(conn, order_id)
        conn.execute("UPDATE work_orders SET status = 'done' WHERE id = ?", (order_id,))
        return _get(conn, order_id)


@app.get("/health")
def health():
    return {"status": "ok"}
