from typing import Union, Optional

from pydantic import BaseModel, Field

class ChargePointLocatorInput(BaseModel):
    address: str = Field(..., description="The location to find EV charging stations")
    max_distance: int = Field(..., description="The distance in kilometers to search for charging stations")
    socket_type: Union[str, None] = Field(None, description="The type of charging socket (e.g., 'CCS', 'CHAdeMO', 'Type 2'). If not provided, all types will be returned.")



# Schema for EV trip planner tool
class EvTripPlannerInput(BaseModel):
    user_address: str = Field(
        description="The starting address for the trip"
    )
    user_destination_address: str = Field(
        description="The destination address for the trip"
    )
    socket_type: Optional[str] = Field(
        None, 
        description="The type of charging socket (e.g., 'CCS', 'CHAdeMO', 'Type 2'). If not provided, all types will be considered."
    )

# JSON Schema representations
CHARGE_POINT_LOCATOR_SCHEMA = {
    "type": "object",
    "properties": {
        "address": {
            "type": "string",
            "description": "The location to find EV charging stations"
        },
        "max_distance": {
            "type": "integer",
            "description": "The distance in kilometers to search for charging stations"
        },
        "socket_type": {
            "type": ["string", "null"],
            "description": "The type of charging socket (e.g., 'CCS', 'CHAdeMO', 'Type 2'). If not provided, all types will be returned."
        }
    },
    "required": ["address", "max_distance"],
    "additionalProperties": False
}

EV_TRIP_PLANNER_SCHEMA = {
    "type": "object",
    "properties": {
        "user_address": {
            "type": "string",
            "description": "The starting address for the trip"
        },
        "user_destination_address": {
            "type": "string",
            "description": "The destination address for the trip"
        },
        "socket_type": {
            "type": ["string", "null"],
            "description": "The type of charging socket (e.g., 'CCS', 'CHAdeMO', 'Type 2'). If not provided, all types will be considered."
        }
    },
    "required": ["user_address", "user_destination_address"],
    "additionalProperties": False
}

# schemas for prompts
class FindChargingStationsPrompt(BaseModel):
    location: str = Field(..., description="The location to find EV charging stations")
    radius: Optional[str] = Field(None, description="The distance in kilometers to search for charging stations")
    socket_type: Optional[str] = Field(None, description="The type of charging socket (e.g., 'CCS', 'CHAdeMO', 'Type 2'). If not provided, all types will be considered.")


class ChargingTimeEstimatePrompt(BaseModel):
    vehicle_model: str = Field(..., description="The model of the EV")
    current_charge: str = Field(..., description="The current charge of the EV")
    target_charge: str = Field(..., description="The target charge of the EV")
    charger_power: str = Field(..., description="The power of the charger")


class RoutePlannerPrompt(BaseModel):
    start_location: str = Field(..., description="The starting address for the trip")
    end_location: str = Field(..., description="The destination address for the trip")
    vehicle_range: str = Field(..., description="The range of the EV")
    current_charge: Optional[str] = Field(None, description="The current charge of the EV")




