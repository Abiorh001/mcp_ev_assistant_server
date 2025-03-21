from typing import Optional, List, Dict, Any
import json
from pydantic import BaseModel, Field
from decouple import config

from agentTools.charge_station_locator import ChargePointsLocatorTool
from agentTools.constant import DISTANCE_IN_KM
from agentTools.utils.charge_station_availability import \
    is_charge_station_available
from agentTools.utils.google_location import (
    get_distance_and_route_info, get_latitude_longitude)
from core.logger import logger
from core.schemas import EvTripPlannerInput

# Get API key from config
opencharge_map_api_key = config("OPENCHARGE_MAP_API_KEY")



class EvTripPlannerTool:
    """
    This class is used to plan a trip for an electric vehicle.
    """
    def __init__(self):
        """
        This function is used to initialize the ev trip planner tool.
        """
        self.charge_points_locator = ChargePointsLocatorTool()

    def segment_trip(self, user_address: str, user_destination_address: str):
        """
        This function is used to segment the trip into smaller segments.
        Args:
            user_address: str
            user_destination_address: str
        """
        # Get distance and route information
        try:
            distance, duration, steps = get_distance_and_route_info(
                user_address, user_destination_address
            )
            
            # Simple segments implementation - just return the destination
            # In a real implementation, this would break the trip into multiple segments
            return [user_destination_address]
        except Exception as e:
            logger.error(f"Error in segment_trip: {e}")
            return []

    def get_charge_point(self, user_address: str, user_destination_address: str):
        """
        This function is used to get the charge points between the user's location and the destination.
        Args:
            user_address: str
            user_destination_address: str
        """
        # Reset charge stations
        self.charge_points_locator._charge_stations = []
        
        # Get trip segments
        segments = self.segment_trip(user_address, user_destination_address)
        
        for segment in segments:
            try:
                # Get charge stations for each segment
                charge_stations = self.charge_points_locator.charge_point_locator_unfiltered(
                    segment, max_distance=5
                )
                # No additional processing needed as charge_point_locator_unfiltered now handles this correctly
            except Exception as e:
                logger.error(f"Error in get_charge_point: {e}")
                
        return self.charge_points_locator._charge_stations

    def ev_trip_planner(self, user_address, user_destination_address, socket_type=None):
        """
        This function is used to get the charge points for the trip.
        Args:
            user_address: str
            user_destination_address: str
            socket_type: str
        """
        self.charge_points_locator._filtered_charging_stations = []  # Reset filtered stations
        charge_stations = self.get_charge_point(user_address, user_destination_address)
        for charge_point in charge_stations:
            for connection in charge_point["Connections"]:
                connection_type = connection.get("ConnectionType", {}).get("Title", "")
                if socket_type is None or connection_type.startswith(socket_type):
                    if is_charge_station_available(charge_point):
                        self.charge_points_locator._filtered_charging_stations.append(
                            charge_point
                        )
                        break
        sorted_charging_points = sorted(
            self.charge_points_locator._filtered_charging_stations,
            key=lambda x: float(x["DistanceToUserLocation"].split()[0]),
        )
        return sorted_charging_points

# Initialize the tool instance
_ev_trip_planner_instance = EvTripPlannerTool()

# Simple function for LangGraph
def ev_trip_planner(user_address: str, user_destination_address: str, socket_type: str = None) -> str:
    """Plan a trip for an electric vehicle, including charging stops along the route.
    
    Args:
        user_address: The starting address for the trip
        user_destination_address: The destination address for the trip
        socket_type: The type of charging socket (e.g., 'CCS', 'CHAdeMO', 'Type 2'). If not provided, all types will be considered.
    """
    try:
        # Get route information
        distance_info = get_distance_and_route_info(user_address, user_destination_address)
        if distance_info:
            distance, duration, _ = distance_info
        else:
            distance = "Unknown distance"
            duration = "Unknown duration"
        
        # Get charging stations
        stations = _ev_trip_planner_instance.ev_trip_planner(user_address, user_destination_address, socket_type)
        
        # Convert complex data structure to a simple formatted string
        result_lines = []
        result_lines.append(f"EV Trip from {user_address} to {user_destination_address}:")
        result_lines.append(f"Total distance: {distance}")
        result_lines.append(f"Expected driving time: {duration}")
        result_lines.append("")
        
        # Add charging stations information
        if stations:
            result_lines.append(f"Found {len(stations)} charging stations along the route:")
            
            for i, station in enumerate(stations[:5], 1):  # Limit to 5 stations max
                address_info = station.get("AddressInfo", {})
                connections = station.get("Connections", [])
                
                name = address_info.get("Title", "Unnamed Station")
                address_str = f"{address_info.get('AddressLine1', '')}, {address_info.get('Town', '')}, {address_info.get('StateOrProvince', '')}"
                distance = station.get("DistanceToUserLocation", "Unknown distance")
                
                # Get connection types
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
        else:
            result_lines.append("No charging stations found along the route.")
        
        return "\n".join(result_lines)
    
    except Exception as e:
        logger.error(f"Error in ev_trip_planner: {e}")
        return f"Error planning trip: {str(e)}"

