import data
import Nodes
# import data_30
# fuel_depots = data.fuel_depots
# fuel_stations = data.fuel_stations
def make_character():
    fuel_depots = data.fuel_depots
    fuel_stations = data.fuel_stations
    d1 = data.d1
    d2 = data.d2
    # d3 = data.d3
# 创建油库对象
    alldepots=[] # 包含全体油库对象的列表
    all_vehicle=[]
    for depot in fuel_depots :
            id=depot['id']
            location=depot['location']
            tube=depot['tube']
            new_depot=Nodes.depot(
                id = id,  # 油库名
                location = location,  # 油库位置
                tube = tube,  # 鹤管数量

            )
            alldepots.append(new_depot)

    # 创建油站对象
    allstations=[] # 包含全体油站对象的列表

    for station in fuel_stations:
        station_id=station['id']
        location=station['location']
        capacity=station['capacity']

        oil_92 = station['92']
        oil_95 = station['95']
        oil_derv = station['derv']
        new_station=Nodes.station(

            station_id = station_id,  # 加油站id
            location = location,  # 加油站位置
            capacity = capacity,  # 三种油品最大容量
            oil_92 = oil_92,
            oil_95 = oil_95,
            oil_derv = oil_derv,

        )
        allstations.append(new_station)

    # 创建车辆对象
    #油库1 汽油 单舱
    for vehicle in d1[0]['Single_gasoline']:
        capacity=vehicle['capacity']
        truck_id=vehicle['id']
        speed=vehicle['speed']
        cabin=vehicle['cabin']
        cost=vehicle['cost']
        new_vehicle=Nodes.TankerTruck(
            capacity = capacity,  # 车厢容量
            truck_id = truck_id , # 车类id
            speed = speed,
            cabin = cabin , # 油罐的数量（1或2）
            cost = cost
        )
        alldepots[0].Single_gasoline.put(new_vehicle)
        all_vehicle.append(new_vehicle)
    #油库1 柴油 单舱
    for vehicle in d1[1]['Single_diesel']:
        capacity=vehicle['capacity']
        truck_id=vehicle['id']
        speed=vehicle['speed']
        cabin=vehicle['cabin']
        cost=vehicle['cost']
        new_vehicle=Nodes.TankerTruck(
            capacity = capacity,  # 车厢容量
            truck_id = truck_id , # 车类id
            speed = speed,
            cabin = cabin , # 油罐的数量（1或2）
            cost = cost
        )
        alldepots[0].Single_diesel.put(new_vehicle)
        all_vehicle.append(new_vehicle)
    #油库1 汽油 双舱
    for vehicle in d1[2]['Double_gasoline']:
        capacity=vehicle['capacity']
        truck_id=vehicle['id']
        speed=vehicle['speed']
        cabin=vehicle['cabin']
        cost=vehicle['cost']
        new_vehicle=Nodes.TankerTruck(
            capacity = capacity,  # 车厢容量
            truck_id = truck_id , # 车类id
            speed = speed,
            cabin = cabin , # 油罐的数量（1或2）
            cost = cost
        )
        alldepots[0].Double_gasoline.put(new_vehicle)
        all_vehicle.append(new_vehicle)
    #油库1 柴油 双舱
    for vehicle in d1[3]['Double_diesel']:
        capacity=vehicle['capacity']
        truck_id=vehicle['id']
        speed=vehicle['speed']
        cabin=vehicle['cabin']
        cost=vehicle['cost']
        new_vehicle=Nodes.TankerTruck(
            capacity = capacity,  # 车厢容量
            truck_id = truck_id , # 车类id
            speed = speed,
            cabin = cabin , # 油罐的数量（1或2）
            cost = cost
        )
        alldepots[0].Double_diesel.put(new_vehicle)
        all_vehicle.append(new_vehicle)
    #油库2 汽油 单舱
    for vehicle in d2[0]['Single_gasoline']:
        capacity=vehicle['capacity']
        truck_id=vehicle['id']
        speed=vehicle['speed']
        cabin=vehicle['cabin']
        cost=vehicle['cost']
        new_vehicle=Nodes.TankerTruck(
            capacity = capacity,  # 车厢容量
            truck_id = truck_id , # 车类id
            speed = speed,
            cabin = cabin , # 油罐的数量（1或2）
            cost = cost
        )
        alldepots[1].Single_gasoline.put(new_vehicle)
        all_vehicle.append(new_vehicle)
    #油库2 柴油 单舱
    for vehicle in d2[1]['Single_diesel']:
        capacity=vehicle['capacity']
        truck_id=vehicle['id']
        speed=vehicle['speed']
        cabin=vehicle['cabin']
        cost=vehicle['cost']
        new_vehicle=Nodes.TankerTruck(
            capacity = capacity,  # 车厢容量
            truck_id = truck_id , # 车类id
            speed = speed,
            cabin = cabin , # 油罐的数量（1或2）
            cost = cost
        )
        alldepots[1].Single_diesel.put(new_vehicle)
        all_vehicle.append(new_vehicle)
    #油库2 汽油 双舱
    for vehicle in d2[2]['Double_gasoline']:
        capacity=vehicle['capacity']
        truck_id=vehicle['id']
        speed=vehicle['speed']
        cabin=vehicle['cabin']
        cost=vehicle['cost']
        new_vehicle=Nodes.TankerTruck(
            capacity = capacity,  # 车厢容量
            truck_id = truck_id , # 车类id
            speed = speed,
            cabin = cabin , # 油罐的数量（1或2）
            cost = cost
        )
        alldepots[1].Double_gasoline.put(new_vehicle)
        all_vehicle.append(new_vehicle)
    #油库2 柴油 双舱
    for vehicle in d2[3]['Double_diesel']:
        capacity=vehicle['capacity']
        truck_id=vehicle['id']
        speed=vehicle['speed']
        cabin=vehicle['cabin']
        cost=vehicle['cost']
        new_vehicle=Nodes.TankerTruck(
            capacity = capacity,  # 车厢容量
            truck_id = truck_id , # 车类id
            speed = speed,
            cabin = cabin , # 油罐的数量（1或2）
            cost = cost
        )
        alldepots[1].Double_diesel.put(new_vehicle)
        all_vehicle.append(new_vehicle)
    #油库3 汽油 单舱
    # for vehicle in d3[0]['Single_gasoline']:
    #     capacity=vehicle['capacity']
    #     truck_id=vehicle['id']
    #     speed=vehicle['speed']
    #     cabin=vehicle['cabin']
    #     cost=vehicle['cost']
    #     new_vehicle=Nodes.TankerTruck(
    #         capacity = capacity,  # 车厢容量
    #         truck_id = truck_id , # 车类id
    #         speed = speed,
    #         cabin = cabin , # 油罐的数量（1或2）
    #         cost = cost
    #     )
    #     alldepots[2].Single_gasoline.put(new_vehicle)
    #     all_vehicle.append(new_vehicle)
    # #油库3 柴油 单舱
    # for vehicle in d3[1]['Single_diesel']:
    #     capacity=vehicle['capacity']
    #     truck_id=vehicle['id']
    #     speed=vehicle['speed']
    #     cabin=vehicle['cabin']
    #     cost=vehicle['cost']
    #     new_vehicle=Nodes.TankerTruck(
    #         capacity = capacity,  # 车厢容量
    #         truck_id = truck_id , # 车类id
    #         speed = speed,
    #         cabin = cabin , # 油罐的数量（1或2）
    #         cost = cost
    #     )
    #     alldepots[2].Single_diesel.put(new_vehicle)
    #     all_vehicle.append(new_vehicle)
    # #油库3 汽油 双舱
    # for vehicle in d3[2]['Double_gasoline']:
    #     capacity=vehicle['capacity']
    #     truck_id=vehicle['id']
    #     speed=vehicle['speed']
    #     cabin=vehicle['cabin']
    #     cost=vehicle['cost']
    #     new_vehicle=Nodes.TankerTruck(
    #         capacity = capacity,  # 车厢容量
    #         truck_id = truck_id , # 车类id
    #         speed = speed,
    #         cabin = cabin , # 油罐的数量（1或2）
    #         cost = cost
    #     )
    #     alldepots[2].Double_gasoline.put(new_vehicle)
    #     all_vehicle.append(new_vehicle)
    # # 油库3 柴油 双舱
    # for vehicle in d3[3]['Double_diesel']:
    #     capacity=vehicle['capacity']
    #     truck_id=vehicle['id']
    #     speed=vehicle['speed']
    #     cabin=vehicle['cabin']
    #     cost=vehicle['cost']
    #     new_vehicle=Nodes.TankerTruck(
    #         capacity = capacity,  # 车厢容量
    #         truck_id = truck_id , # 车类id
    #         speed = speed,
    #         cabin = cabin , # 油罐的数量（1或2）
    #         cost = cost
    #     )
    #     alldepots[2].Double_diesel.put(new_vehicle)
    #     all_vehicle.append(new_vehicle)
    return alldepots,allstations,all_vehicle
alldepots,allstations,all_vehicle=make_character()




def find_depot_by_id(depot_id):
        for depot in alldepots:
            if depot.id == depot_id:
                return depot
def find_station_by_id(station_id):
        for station in allstations:
            if station.station_id == station_id:
                return station
def find_vehicle_by_id(vehicle_id):
        for vehicle in all_vehicle:
            if vehicle.truck_id == vehicle_id:
                return vehicle