
import distance
# 计算距离的占位函数，实际项目中需要根据油库和油站的地理位置计算
import random

random.seed(42)  # 固定随机数种子，确保每次运行生成相同的初始解和邻域解

alldistance=distance.alldistance
empty_cost = 10
single_cost = 20
double_cost = 40
def calculate_distance(route):
    depot, routes = route
    total_distance = 0
    for sub_route in routes:
        # 假设每个订单距离的随机数表示（实际应该替换为真实的距离计算）
        if len(sub_route)==2:
            distance = sum(alldistance[(depot,sub_route[0])]+alldistance[(sub_route[0],sub_route[1])]+alldistance[(sub_route[1],depot)] for _ in sub_route)
            # distance=distance+
            cost=double_cost*alldistance[(depot,sub_route[0])]+single_cost*alldistance[(sub_route[0],sub_route[1])]+empty_cost*alldistance[(sub_route[1],depot)]
            distance += cost
            total_distance += distance

        elif len(sub_route)==1:
            distance = sum(alldistance[(depot, sub_route[0])] + alldistance[(sub_route[0], depot)] for _ in sub_route)
            cost=single_cost*alldistance[(depot,sub_route[0])]+empty_cost*alldistance[(sub_route[0],depot)]
            distance+=cost
            total_distance += distance
    return total_distance

# 随机初始化：将订单随机分配到一个油库并生成路径
def initial_solution(depot, orders):
    return depot, create_routes(orders)

# 生成邻域解，允许每趟行程最多有两个订单
def get_neighbors(route):
    depot, routes = route
    neighbors = []

    # 交换操作：在同一油库内交换订单顺序
    for i in range(len(routes)):
        sub_route = routes[i]
        if len(sub_route) == 2:  # 仅对包含两个油站的行程进行交换操作
            new_routes = routes[:]
            new_sub_route = sub_route[:]
            new_sub_route[0], new_sub_route[1] = new_sub_route[1], new_sub_route[0]  # 交换两个油站的顺序
            new_routes[i] = new_sub_route  # 更新该行程
            neighbors.append((depot, new_routes))  # 将新的邻域解加入列表

    # 插入操作：从一个行程移除一个订单并插入到另一个行程
    for i in range(len(routes)):
        for j in range(i + 1, len(routes)):
            if len(routes[i]) > 0 and len(routes[j]) < 2:  # 插入到未满的行程
                new_routes = routes[:]
                new_sub_route_i = new_routes[i][:]  # 复制行程
                new_sub_route_j = new_routes[j][:]

                # 移动一个订单
                order_to_move = new_sub_route_i.pop(0)
                new_sub_route_j.append(order_to_move)

                # 更新新的行程
                new_routes[i] = new_sub_route_i
                new_routes[j] = new_sub_route_j

                # 过滤掉可能出现的空列表
                new_routes = [r for r in new_routes if r]  # 移除空列表
                neighbors.append((depot, new_routes))

    # 拆分操作：将两个油站的行程拆分为两个单个油站的行程
    for i in range(len(routes)):
        sub_route = routes[i]
        if len(sub_route) == 2:
            new_routes = routes[:]
            new_routes[i] = [sub_route[0]]  # 拆分为两个单油站路径
            new_routes.append([sub_route[1]])
            new_routes = [r for r in new_routes if r]  # 过滤掉空列表
            neighbors.append((depot, new_routes))

    # 合并操作：将两个单油站的行程合并为一个双油站的行程
    for i in range(len(routes)):
        for j in range(i + 1, len(routes)):
            if len(routes[i]) == 1 and len(routes[j]) == 1:  # 两条路径都是单油站的情况
                new_routes = routes[:]
                new_sub_route_i = new_routes[i][:]  # 复制行程
                new_sub_route_j = new_routes[j][:]

                # 合并两个单油站为一个双油站
                new_routes[i] = [new_sub_route_i[0], new_sub_route_j[0]]
                new_routes.pop(j)  # 移除被合并的路径

                # 过滤掉可能出现的空列表
                new_routes = [r for r in new_routes if r]  # 移除空列表
                neighbors.append((depot, new_routes))

    return neighbors


def create_routes(orders):
    routes = []
    order_indices = list(range(len(orders)))  # 创建订单索引列表
    random.shuffle(order_indices)  # 随机打乱顺序

    i = 0
    while i < len(order_indices):
        if i + 1 < len(order_indices) and random.random() > 0.3:  # 70%的概率分配两个油站
            routes.append([orders[order_indices[i]]['station_name'], orders[order_indices[i + 1]]['station_name']])
            i += 2
        else:  # 30%的概率分配单个油站
            routes.append([orders[order_indices[i]]['station_name']])
            i += 1
    return routes


def tabu_search(depot, orders, iterations, tabu_size):
    # 初始解：将订单随机分配到固定的一个油库
    current_solution = initial_solution(depot, orders)
    best_solution = current_solution
    tabu_list = []

    for _ in range(iterations):
        neighbors = get_neighbors(current_solution)  # 生成邻域解
        neighbors = [n for n in neighbors if n not in tabu_list]  # 过滤禁忌解

        if not neighbors:
            continue

        # 选择邻域解中距离最短的
        current_solution = min(neighbors, key=calculate_distance)

        # 如果当前解优于历史最佳解，则更新最佳解
        if calculate_distance(current_solution) < calculate_distance(best_solution):
            best_solution = current_solution

        # 将当前解添加到禁忌列表
        tabu_list.append(current_solution)
        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)  # 保持禁忌列表大小
    a=int(calculate_distance(best_solution) / len(orders))
    return best_solution,int(calculate_distance(best_solution)/len(orders))
# orders = []  # 订单
#
# depot = 'd1'  # 固定油库
# order1 = {}
# order1['station_name'] ='s1'
# order1['oil_class'] = '95'
# orders.append(order1)
#
# order2 = {}
# order2['station_name'] ='s2'
# order2['oil_class'] = '95'
# orders.append(order2)
#
# order3 = {}
# order3['station_name'] ='s3'
# order3['oil_class'] = '92'
# orders.append(order3)
#
# order4 = {}
# order4['station_name'] ='s4'
# order4['oil_class'] = '95'
# orders.append(order4)
#
# order5 = {}
# order5['station_name'] ='s5'
# order5['oil_class'] = '92'
# orders.append(order5)
#
# order6 = {}
# order6['station_name'] ='s6'
# order6['oil_class'] = '92'
# orders.append(order6)
# iterations = 100  # 禁忌搜索迭代次数
# tabu_size = 10  # 禁忌列表大小
#
# best_route,dis = tabu_search(depot, orders, iterations, tabu_size)
# print(f"Best Route from {best_route[0]}: {best_route[1]}")
# print(f"Total Distance: {calculate_distance(best_route)}")
# print(dis)