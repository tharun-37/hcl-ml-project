from pydantic import BaseModel, Field
from typing import Optional

class BatteryPredictionRequest(BaseModel):
    model_name: Optional[str] = "Random Forest Regressor"
    cycle: float = Field(..., ge=0, le=10000, description="Cycle count")
    discharge_capacity: float = Field(..., ge=300.0, le=800.0, description="Discharge Capacity (mAh)")
    mean_voltage: float = Field(..., ge=2.5, le=4.5, description="Mean Voltage (V)")
    min_voltage: float = Field(..., ge=2.0, le=4.0, description="Min Voltage (V)")
    max_voltage: float = Field(..., ge=3.5, le=4.5, description="Max Voltage (V)")
    std_voltage: float = Field(..., ge=0.0, le=1.0, description="Std Voltage (V)")
    mean_temp: float = Field(..., ge=20.0, le=60.0, description="Mean Temperature (C)")
    max_temp: float = Field(..., ge=20.0, le=65.0, description="Max Temperature (C)")
    discharge_duration: float = Field(..., ge=0.01, le=1.0, description="Discharge Duration (hr)")
    energy: float = Field(..., ge=0.5, le=4.0, description="Energy Delivered (Wh)")
