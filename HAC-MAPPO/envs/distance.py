import math

import New_characters
import get_path

alldistance={}
def calculate_distance(location1, location2):
    # x1, y1 = location1
    # x2, y2 = location2
    # # distance = geodesic(location1, location2).meters
    # distance = int(math.sqrt((x2 - x1)**2 + (y2 - y1)**2))
    lon1, lat1 = location1
    lon2, lat2 = location2

    # 将经纬度从度转换为弧度
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # 计算差值
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Haversine公式
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # 地球半径（公里）
    R = 6371.0
    distance = R * c

    return distance

# ...（之前的代码）

# 计算站点之间的距离
# 计算站点之间的距离
def station_station_distance(allstations):
    fuel_station_distances = {}
    for i in range(len(allstations)):
        for j in range(i + 1, len(allstations)):
            station1 = allstations[i]
            station2 = allstations[j]
            location1 = station1.location
            location2 = station2.location
            distance = calculate_distance(location1, location2)

            # 存储距离
            fuel_station_distances[(station1.station_id, station2.station_id)] = distance
            fuel_station_distances[(station2.station_id, station1.station_id)] = distance  # 添加这一行

    return fuel_station_distances




def station_depot_distance(alldepots, allstations):
    fuel_depot_distances = {}
    for station in allstations:
        location1 = station.location
        for depot in alldepots:
            location2 = depot.location
            distance = calculate_distance(location1, location2)

            # 存储距离
            fuel_depot_distances[(station.station_id, depot.id)] = distance
            fuel_depot_distances[(depot.id, station.station_id)] = distance  # 添加这一行

    return fuel_depot_distances

ss = station_station_distance(New_characters.allstations)
sd = station_depot_distance(New_characters.alldepots, New_characters.allstations)
alldistance.update(ss)
alldistance.update(sd)
# print(alldistance)
def get_path_distance(alldistance,path):
    total_distace=0
    for i in range(len(path)-1):
       total_distace+=alldistance[(path[i],path[i+1])]
    return total_distace
# all_vehicle=New_characters.all_vehicle
# veh_num=len(all_vehicle)
# print(veh_num)