from datetime import datetime


def is_charge_station_available(charging_station: list) -> bool:
    # Check if the station is operational and user-selectable
    status_type = charging_station["StatusType"]
    if status_type is not None:
        if status_type["IsOperational"] and status_type["IsUserSelectable"]:
            # Check if the station's status was recently updated
            # last_status_update = datetime.fromisoformat(
            #     # Remove Z from the end
            #     charging_station["DateLastStatusUpdate"][:-1]
            # )
            # current_date = datetime.now()
            # time_difference = current_date - last_status_update
            # # Check if the status was updated within the last 180 days or any
            # # prefered day(s)
            # if time_difference.days <= 180:
                return True
    return False
