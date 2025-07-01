# import distance
# def get_path(station,alldepots):
#     distances = []
#     for depot in alldepots:
#         distance_sd = distance.calculate_distance(depot.location, station.location)
#
#
#
#     return distances
import distance
import New_characters
def get_path(station, alldepots):
    closest_depot = None
    min_distance = float('inf')  # 初始化为无限大

    for depot in alldepots:
        distance_sd = distance.calculate_distance(depot.location, station.location)
        if distance_sd < min_distance:
            min_distance = distance_sd
            closest_depot = depot
    final_depot=New_characters.find_depot_by_id(closest_depot.id)
    return final_depot
# alldepots=New_characters.alldepots
# allstations=New_characters.allstations
# for index, station in enumerate(allstations):
#     depot=get_path.get_path(station,alldepots)