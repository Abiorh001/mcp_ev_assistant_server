import os
from typing import Union, List, Optional, Dict, Any
import json

import requests
from decouple import config
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from .constant import DISTANCE_IN_KM
from .utils.charge_station_availability import \
    is_charge_station_available
from .utils.google_location import (
    get_distance_and_route_info, get_latitude_longitude)
from core.logger import logger

# Get API key from config
opencharge_map_api_key = config("OPENCHARGE_MAP_API_KEY")

if not opencharge_map_api_key:
    logger.error("OpenChargeMap API Key is not provided")
    raise ValueError("OpenChargeMap API Key is not provided")

# Schema for charge station locator tool
class ChargeStationInput(BaseModel):
    address: str = Field(
        description="The address or location to search for charging stations"
    )
    socket_type: Optional[str] = Field(
        None, 
        description="The type of charging socket (e.g., 'CCS', 'CHAdeMO', 'Type 2'). If not provided, all types will be returned."
    )

class ChargePointsLocatorTool:
    """
    This class is used to locate EV charging stations near a specified location.
    """
    def __init__(self):
        """
        This function is used to initialize the charge points locator tool.
        """
        self._charge_stations = []
        self._filtered_charging_stations = []

    def get_closest_charge_stations(
        self,
        user_latitude: float,
        user_longitude: float,
        max_distance: Union[int, float],
        max_results: int = 10,
    ) -> list:
        """
        This function is used to get the closest charging stations to the user's location using OpenChargeMap API.
        Args:
            user_latitude: float
            user_longitude: float
            max_results: int
            max_distance: Union[int, float]
        Returns:
            list: A list of charging stations
        """
        base_url = f"https://api.openchargemap.io/v3/poi?key={opencharge_map_api_key}"
        headers = {"Accept": "application/json"}
        params = {
            "output": "json",
            "maxresults": max_results,
            "latitude": user_latitude,
            "longitude": user_longitude,
            "distance": max_distance,
            "distanceunit": DISTANCE_IN_KM,
        }
        try:
            response = requests.get(base_url, params=params, headers=headers)
            response.raise_for_status()
            return response.json() if response.status_code == 200 else []
        except requests.RequestException as e:
            logger.error(
                "Error getting charging stations from OpenChargeMap API: %s", e
            )
            return []

    def charge_point_locator_unfiltered(
        self, address: str, max_distance: int
    ):
        """
        This function is used to get all charging stations near the user's location.
        Args:
            address: str
            max_distance: int
        Returns:
            list: A list of charging stations
        """
        try:
            user_latitude, user_longitude = get_latitude_longitude(address=address)
            if user_latitude is None or user_longitude is None:
                logger.error(f"Could not get coordinates for address: {address}")
                return []

            logger.info(f"Got geocode for {address}: {user_latitude}, {user_longitude}")

            charge_points = self.get_closest_charge_stations(
                user_latitude,
                user_longitude,
                max_distance,
            )
            
            for charge_point in charge_points:
                charge_point_uuid = charge_point.get("UUID")
                if charge_point_uuid and not any(
                    charge_point_uuid == cp.get("UUID") for cp in self._charge_stations
                ):
                    address_info = charge_point.get("AddressInfo", {})
                    charge_point_address = address_info.get("AddressLine1", "")
                    charge_point_city = address_info.get("Town", "")
                    charge_point_state = address_info.get("StateOrProvince", "")
                    charge_point_country = address_info.get("Country", {}).get("Title", "")
                    charge_point_full_address = ", ".join(
                        filter(
                            None,
                            [
                                charge_point_address,
                                charge_point_city,
                                charge_point_state,
                                charge_point_country,
                            ],
                        )
                    )
                    
                    try:
                        distance_info = get_distance_and_route_info(address, charge_point_full_address)
                        if distance_info:
                            distance, duration, steps = distance_info
                            # Safely convert distance string to float
                            try:
                                distance_value = float(distance.split()[0]) * 1.60934  # Convert from mile to km
                                distance_value = round(distance_value, 2)
                                distance_str = f"{distance_value} {DISTANCE_IN_KM}"
                                
                                # Add steps with proper error handling
                                step_instructions = []
                                if steps:
                                    step_instructions = [step.get("html_instructions", "") for step in steps]
                                
                                charge_point.update({
                                    "DistanceToUserLocation": distance_str,
                                    "DurationToUserLocation": duration,
                                    "StepsDirectionFromUserLocationToChargeStation": step_instructions,
                                })
                            except (ValueError, AttributeError, IndexError) as e:
                                # Fallback if parsing the distance fails
                                logger.error(f"Error parsing distance: {e}")
                                charge_point["DistanceToUserLocation"] = "Unknown distance"
                        else:
                            # Set default if route information is not available
                            charge_point["DistanceToUserLocation"] = "Unknown distance"
                    except Exception as e:
                        logger.error(f"Error getting distance and route info: {e}")
                        charge_point["DistanceToUserLocation"] = "Unknown distance"
                        
                    logger.info("Charge point added to the the charge stations list.")
                    self._charge_stations.append(charge_point)

            return self._charge_stations
        except Exception as e:
            logger.error(f"Error in charge_point_locator_unfiltered: {e}")
            return []

    def charge_points_locator(self, address: str, max_distance: int, socket_type: str = None):
        """
        This function is used to get the closest charging stations to the user's location using OpenChargeMap API.
        Args:
            address: str
            socket_type: str
        Returns:
            list: A list of charging stations
        """
        self._filtered_charging_stations = []  # Reset filtered stations
        charge_stations = self.charge_point_locator_unfiltered(address=address, max_distance=max_distance)
        
        for charge_point in charge_stations:
            # Safety check for connections
            connections = charge_point.get("Connections", [])
            if not connections:
                continue
                
            for connection in connections:
                connection_type = connection.get("ConnectionType", {}).get("Title", "")
                if socket_type is None or connection_type.startswith(socket_type):
                    if is_charge_station_available(charge_point):
                        self._filtered_charging_stations.append(charge_point)
                    break

        # Safe sorting with fallback
        try:
            sorted_charging_points = sorted(
                self._filtered_charging_stations,
                key=lambda x: float(str(x.get("DistanceToUserLocation", "")).split()[0]),
            )
            logger.info("Sorted charging points by distance to user location.")
        except Exception as e:
            logger.error(f"Error sorting charging points: {e}")
            # Return unsorted if sorting fails
            sorted_charging_points = self._filtered_charging_stations
            
        return sorted_charging_points

# Initialize the tool instance
_charge_station_locator_instance = ChargePointsLocatorTool()


def charge_points_locator(address: str, max_distance: int = 5, socket_type: str = None) -> str:
    """Find EV charging stations near a specified location.
    
    Args:
        address: The address or location to search for charging stations
        socket_type: The type of charging socket (e.g., 'CCS', 'CHAdeMO', 'Type 2'). If not provided, all types will be returned.
    """
    try:
        stations = _charge_station_locator_instance.charge_points_locator(address, max_distance=max_distance, socket_type=socket_type)
        
        # Convert complex data structure to a simple formatted string
        result_lines = []
        result_lines.append(f"Found {len(stations)} charging stations near {address}:")
        
        for i, station in enumerate(stations[:10], 1):  # Limit to 10 stations max
            address_info = station.get("AddressInfo", {})
            connections = station.get("Connections", [])
            
            name = address_info.get("Title", "Unnamed Station")
            address_str = f"{address_info.get('AddressLine1', '')}, {address_info.get('Town', '')}, {address_info.get('StateOrProvince', '')}"
            
            # Handle distance safely
            distance = station.get("DistanceToUserLocation", "Unknown distance")
            
            # Get connection types safely
            connection_types = []
            for conn in connections:
                conn_type = conn.get("ConnectionType", {}).get("Title", "Unknown")
                if conn_type not in connection_types:
                    connection_types.append(conn_type)
            
            conn_str = ", ".join(connection_types) if connection_types else "No connection info"
            
            result_lines.append(f"{i}. {name} ({distance})")
            result_lines.append(f"   Address: {address_str}")
            result_lines.append(f"   Connection Types: {conn_str}")
            result_lines.append("")
        
        if not stations:
            result_lines.append("No charging stations found in this area.")
            
        return "\n".join(result_lines)
    
    except Exception as e:
        logger.error(f"Error in charge_points_locator: {e}")
        return f"I couldn't find charging stations near {address} due to a technical issue. Please try a different location or check again later."


# result = charge_points_locator("2360 Mission College Blvd, Santa Clara, California, USA")
# print(result)

# charge_points = _charge_station_locator_instance.get_closest_charge_stations(37.338208, -121.929416, 10, 5)
# print(charge_points)