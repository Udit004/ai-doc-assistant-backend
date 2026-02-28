from datetime import datetime

from pydantic import BaseModel, ConfigDict


class DocumentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int | None = None
    title: str
    filename: str
    status: str
    created_at: datetime
    updated_at: datetime
    chunk_count: int = 0
