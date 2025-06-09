from pydantic import BaseModel
from typing import Optional

class CarPriceSchema(BaseModel):
    year: int
    make: str
    model: str
    trim: Optional[str] = None
    body: str
    transmission: Optional[str] = None
    vin: str
    state: str
    condition: int
    odometer: int
    color: str
    interior: str
    seller: str
    saledate: str