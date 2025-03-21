import requests
from decouple import config


google_map_api_key = config("GOOGLE_MAP_API_KEY")

# Check if the API keys are provided
if google_map_api_key is None:
    raise ValueError("Google Map API Key is not provided")


def get_latitude_longitude(
    address: str, api_key: str = google_map_api_key
) -> tuple[float, float]:
    """
    Get the latitude and longitude of the user's address using Google Maps Geocoding API.
    """
    base_url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"address": address, "key": api_key}
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        data = response.json()
        if data.get("results"):
            latitude, longtitude = data["results"][0]["geometry"]["location"].values()
            return latitude, longtitude
    print(response.text)
    print("Error: Unable to retrieve user location")
    return None, None


def get_distance_and_route_info(
    origin: str, destination: str, api_key: str = google_map_api_key
) -> tuple:
    # Define the API endpoint and parameters
    url = "https://maps.googleapis.com/maps/api/directions/json"
    params = {"origin": origin, "destination": destination, "key": api_key}

    # Send the HTTP request
    response = requests.get(url, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()
        # Extract relevant information from the response
        routes = data["routes"]
        if routes:
            # Extract route information
            route = routes[0]  # Assuming there's only one route
            legs = route["legs"]
            distance = legs[0]["distance"]["text"]
            duration = legs[0]["duration"]["text"]
            steps = legs[0]["steps"]
            return distance, duration, steps
        else:
            print("No routes found.")
    else:
        print("Error:", response.status_code)
