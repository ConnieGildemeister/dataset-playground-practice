from geopy.distance import geodesic
from geopy.geocoders import Nominatim

def find_closest_cities(latitude, longitude, max_distance_km, num_cities, target_city):
    """
    Find the closest cities to a given location within a maximum distance, excluding the target city.
    :param latitude: Latitude of the target location.
    :param longitude: Longitude of the target location.
    :param max_distance_km: Maximum distance in kilometers.
    :param num_cities: Number of closest cities to return.
    :param target_city: Name of the city to exclude from the list of closest cities.
    :return: List of closest cities within the specified distance from the target location.
    """
    geolocator = Nominatim(user_agent="city_finder")
    target_location = (latitude, longitude)

    closest_cities = []

    # List of cities to search from
    cities = ["New York, USA", "Brooklyn, USA", "Chicago, USA", "Houston, USA", "Philadelphia, USA"]

    for city in cities:
        if city == target_city:
            continue  # Skip the target city

        city_location = geolocator.geocode(city)

        if city_location is not None:
            distance = geodesic(target_location, (city_location.latitude, city_location.longitude)).kilometers

            if distance <= max_distance_km:
                closest_cities.append((city, distance))

    # Sort the list by distance and take the first 'num_cities' elements
    closest_cities.sort(key=lambda x: x[1])
    closest_cities = closest_cities[:num_cities]

    return closest_cities

# Example usage:
target_latitude = 40.712728  # Latitude of New York
target_longitude = -74.006015  # Longitude of New York
max_distance_km = 30  # Maximum distance threshold
num_cities = 5  # Number of closest cities to return
target_city = "New York, USA"  # Exclude New York from the list of closest cities

closest_cities = find_closest_cities(target_latitude, target_longitude, max_distance_km, num_cities, target_city)

if closest_cities:
    print(f"The {num_cities} closest cities to New York within {max_distance_km} km (excluding New York) are:")
    for city, distance in closest_cities:
        print(f"{city} - {distance:.2f} km away")
else:
    print("No cities found within the specified distance.")