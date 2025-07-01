def calculate_cost(empty_cost,single_cost,double_cost,order,alldistance):
    if 'single' in order['order_name'] :
        path=order['path']
        distance_single=alldistance[(path[0],path[1])]
        cost=distance_single*single_cost+distance_single*empty_cost
        return cost
    else:
        path = order['path']
        distance_double = alldistance[(path[0], path[1])]
        distance_empty= alldistance[(path[1], path[2])]
        cost = distance_double * double_cost +distance_empty*empty_cost
        return cost

def choose_cost(empty_cost,single_cost,double_cost,alldistance,path):
        distance_double = alldistance[(path[0], path[1])]
        distance_single = alldistance[(path[1], path[2])]
        distance_empty = alldistance[(path[2], path[3])]
        cost= distance_double * double_cost + distance_single * single_cost+distance_empty*empty_cost
        return cost