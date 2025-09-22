import math
import data
import New_characters
import get_path

alldistance={}

alldistance=data.distance
alltime=data.time
def get_path_distance(alldistance,path):
    total_distace=0
    for i in range(len(path)-1):
       total_distace+=alldistance[(path[i],path[i+1])]
    return total_distace
# all_vehicle=New_characters.all_vehicle
# veh_num=len(all_vehicle)

# print(veh_num)
