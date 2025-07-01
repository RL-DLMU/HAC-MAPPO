import choose_order
import New_characters
import calculate_cost
import distance

# total_orders:油库的全部订单
# gasoline_orders:汽油单舱订单
# diesel_orders:柴油单舱订单
def station_action_92(action,station,station_states,depot,empty_cost,single_cost,double_cost,alldistance,step_num):

    depot_id = depot.id
    
    order= choose_order.choose_order_92(action, station, station_states,depot_id,step_num)
    if order== None:
#         # print('订单建立失败')
        return

    if '92_single' in order['order_name']:

        order['path'] = [depot_id, station.station_id, depot_id]
        order['distance'] = distance.get_path_distance(alldistance, order['path'])
        order['cost'] = calculate_cost.calculate_cost(empty_cost, single_cost, double_cost, order, alldistance)
        order['order_type']='not_combined'
        depot.gasoline_orders.append(order)
        # print('生成订单：', order['order_name'],"加入单仓汽油订单列表内")
        depot.flag = True
        depot.total_orders.append(order)
    elif '92_double' in order['order_name']:
        order['path'] = [depot_id, station.station_id, depot_id]
        order['distance'] = distance.get_path_distance(alldistance, order['path'])
        order['cost'] = calculate_cost.calculate_cost(empty_cost, single_cost, double_cost, order, alldistance)
        order['order_type'] = 'not_combined'
        depot.total_orders.append(order)
    if depot.flag:
#         print('生成订单：', order['order_name'], "检查是否需合成")
        while find_optimal_shipping_gasoline(depot, alldistance,
                                    step_num, order_class="gasoline"):
            pass
        depot.flag = False

    # if order['order_type']=='not_combined':
    #

def station_action_95(action, station, station_states, depot, empty_cost, single_cost, double_cost, alldistance,step_num):
    # 改动
    # if action==0:
#     #     print('92汽油')
    # elif action==1:
#     #     print('95汽油')
    # elif action==2:
#     #     print('92+95')
    # elif action==3:
#     #     print('柴油')
    # elif action==4:
#     #     print('不送')
    depot_id = depot.id

    order = choose_order.choose_order_95(action, station, station_states, depot_id, step_num)
    if order == None:
#         # print('订单建立失败')
        return

    if '95_single' in order['order_name']:
        order['path'] = [depot_id, station.station_id, depot_id]
        order['distance'] = distance.get_path_distance(alldistance, order['path'])
        order['cost'] = calculate_cost.calculate_cost(empty_cost, single_cost, double_cost, order, alldistance)
        order['order_type'] = 'not_combined'
        depot.gasoline_orders.append(order)
#         print('生成订单：', order['order_name'], "加入单仓汽油订单列表内")
        depot.flag = True
        depot.total_orders.append(order)
    elif '95_double' in order['order_name']:
        order['path'] = [depot_id, station.station_id, depot_id]
        order['distance'] = distance.get_path_distance(alldistance, order['path'])
        order['cost'] = calculate_cost.calculate_cost(empty_cost, single_cost, double_cost, order, alldistance)
        order['order_type'] = 'not_combined'
        depot.total_orders.append(order)
    if depot.flag:
#         print('生成订单：', order['order_name'], "检查是否需合成")
        while find_optimal_shipping_gasoline(depot,  alldistance,
                                    step_num, order_class="gasoline"):
            pass
        depot.flag = False

    # if order['order_type']=='not_combined':

def station_action_derv(action, station, station_states, depot, empty_cost, single_cost, double_cost, alldistance,step_num):

    depot_id = depot.id

    order = choose_order.choose_order_derv(action, station, station_states, depot_id, step_num)
    if order == None:
#         # print('订单建立失败')
        return

    if 'diesel_single' in order['order_name']:
        order['path'] = [depot_id, station.station_id, depot_id]
        order['distance'] = distance.get_path_distance(alldistance, order['path'])
        order['cost'] = calculate_cost.calculate_cost(empty_cost, single_cost, double_cost, order, alldistance)
        order['order_type'] = 'not_combined'
        depot.diesel_orders.append(order)
#         print('生成订单：', order['order_name'], "加入单仓柴油订单列表内")
        depot.flag = True
        depot.total_orders.append(order)
#         # print('当前油库:',depot.id,'的所有单舱柴油订单：',depot.diesel_orders)

    elif 'diesel_double' in order['order_name']:
        order['path'] = [depot_id, station.station_id, depot_id]
        order['distance'] = distance.get_path_distance(alldistance, order['path'])
        order['cost'] = calculate_cost.calculate_cost(empty_cost, single_cost, double_cost, order, alldistance)
        order['order_type'] = 'not_combined'
        depot.total_orders.append(order)
    if depot.flag:
#         print('生成订单：', order['order_name'], "检查是否需合成")
        while find_optimal_shipping_diesel(depot,  alldistance,
             step_num, order_class="diesel"):
            pass
        depot.flag = False
    # if order['order_type']=='not_combined':

def find_optimal_shipping_gasoline(depot, alldistance,step_num,order_class):
#     # print("进入find_optimal_shipping")
#     # print(f'len(orders):{len(orders)}')


    if len(depot.gasoline_orders) == 0 or len(depot.gasoline_orders) == 1:
#         print("单仓订单不足两个，无需合成")
        return False
    # optimal_pair = None  # 初始化最优订单对为None

    # 遍历所有可能的订单对
    for i in range(len(depot.gasoline_orders)):
        # 由于后面函数会对orders的内容进行删除，可能会出现i等于len(orders)的情况
        if i >=len(depot.gasoline_orders):
            break
        # final_cost= float('inf')
        # final_i = 0
        # final_j = 0
        # final_path = []
        optimal_pair = None


        for j in range(i+1, len(depot.gasoline_orders)):
            # 两个订单都为其他合成订单的子订单，则不做匹配

            if depot.gasoline_orders[i]['is_combined'] == 1 and depot.gasoline_orders[j]['is_combined'] == 1:

                continue

            else:

                if depot.gasoline_orders[i]['station_name'] == depot.gasoline_orders[j]['station_name']:
                    path = depot.gasoline_orders[i]['path']
                    distance_double = alldistance[(path[0], path[1])]
                    distance_empty = alldistance[(path[1], path[2])]
                    cost = distance_double * 30 + distance_empty * 10
                    # orders[i]['order_name']='double'
                    optimal_pair = (i, j, depot.gasoline_orders[i]['path'], cost )
#                     # print("找到pair else")
#                     # print(f'i:{i},j:{j} else')
                    break

    # 如果找到了可以合并的订单对，则将其合并
        if optimal_pair is not None and order_class == "gasoline":
            i, j ,path,cost= optimal_pair
            # 为被合并的订单增加合并后的订单标识，以便后续查找
            combine_orders_gasoline(depot.gasoline_orders[i], depot.gasoline_orders[j], depot, path, cost, step_num,i,j)
#             # print(f'optimal_pair:{optimal_pair}')
#             # print('生成汽油合成订单')
#             # print(f'combined_orders:{combined_orders}')

            # return True

#             # print('生成柴油合成订单')
            # return True

        if len(depot.gasoline_orders) == 0 or len(depot.gasoline_orders) == 1:
            return False
    # 如果没有找到可以合并的订单对，则返回False

    return False
def find_optimal_shipping_diesel(depot, alldistance,step_num,order_class):
#     # print("进入find_optimal_shipping")
#     # print(f'len(orders):{len(orders)}')


    if len(depot.diesel_orders) == 0 or len(depot.diesel_orders) == 1:
#         print("单仓订单不足两个，无需合成")
        return False
    # optimal_pair = None  # 初始化最优订单对为None

    # 遍历所有可能的订单对
    for i in range(len(depot.diesel_orders)):
        # 由于后面函数会对orders的内容进行删除，可能会出现i等于len(orders)的情况
        if i >=len(depot.diesel_orders):
            break
        # final_cost= float('inf')
        # final_i = 0
        # final_j = 0
        # final_path = []
        optimal_pair = None


        for j in range(i+1, len(depot.diesel_orders)):
            # 两个订单都为其他合成订单的子订单，则不做匹配

            if depot.diesel_orders[i]['is_combined'] == 1 and depot.diesel_orders[j]['is_combined'] == 1:

                continue

            else:

                if depot.diesel_orders[i]['station_name'] == depot.diesel_orders[j]['station_name']:
                    path = depot.diesel_orders[i]['path']
                    distance_double = alldistance[(path[0], path[1])]
                    distance_empty = alldistance[(path[1], path[2])]
                    cost = distance_double * 30 + distance_empty * 10
                    # orders[i]['order_name']='double'
                    optimal_pair = (i, j, depot.diesel_orders[i]['path'], cost )
#                     # print("找到pair else")
#                     # print(f'i:{i},j:{j} else')
                    break

    # 如果找到了可以合并的订单对，则将其合并

#             # print(f'optimal_pair:{optimal_pair}')
#             # print('生成汽油合成订单')
#             # print(f'combined_orders:{combined_orders}')

            # return True
        if optimal_pair is not None and order_class == "diesel":
            i, j, path,cost = optimal_pair

            # 为被合并的订单增加合并后的订单标识，以便后续查找

            combine_orders_diesel(depot.diesel_orders[i],depot.diesel_orders[j], depot, path, cost, step_num,i,j)
#             # print('生成柴油合成订单')
            # return True

        if len(depot.diesel_orders) == 0 or len(depot.diesel_orders) == 1:
            return False
    # 如果没有找到可以合并的订单对，则返回False

    return False
def combine_orders_gasoline(order1, order2, depot,path,cost,step_num,i,j):
#     print(order1['order_name'],'+',order2['order_name'],'将合成')
    if order1["oil_class"] == order2["oil_class"]:
        oil_class = order1['oil_class']
    else:
        oil_class = f'{order1["oil_class"]}+{order2["oil_class"]}'
    # 若两个订单都不是其他合成订单的子订单
    if order1['is_combined'] == 0 and order2['is_combined'] == 0:
        # 若两个订单的要运送的油站相同
        # print(order1['order_name'], '和', order2['order_name'], '都不是子订单')
        if order1['station_name'] == order2['station_name']:
            combined_order = {
                'order_name': f'{oil_class}_double_{order1["station_name"]}_{step_num}',
                'station_name': order1['station_name'],
                'depot': order1['depot'],
                'time_to_empty': min(order1['time_to_empty'], order2['time_to_empty']),
                'vehicle_type': 'Double_gasoline',
                'oil_class': oil_class,
                'path': path,
                'distance': distance.get_path_distance(distance.alldistance, path),
                'cost': cost,
                #订单等待
                'time1': max(order1['time1'],order2['time1']),
                'time2': 0,
                #订单等待

                'order_type': 'not_combined'
            }
            # 删除总订单列表中的两个子订单,同时在单舱列表里删除两个子订单，两个子订单不需要被再次合成
            # total_orders = [order for order in total_orders if order['order_name'] != order1['order_name']]
            # total_orders = [order for order in total_orders if order['order_name'] != order2['order_name']]
            depot.total_orders.remove(order1)

            depot.total_orders.remove(order2)
            # print('总订单列表移除：',order1['order_name'],'+',order2['order_name'])
            depot.gasoline_orders.remove(order1)
            depot.gasoline_orders.remove(order2)
            # print('单汽油列表移除：', order1['order_name'], '+', order2['order_name'])
            if combined_order['order_name'] not in depot.total_orders:

                # 将合并后的订单添加到总订单列表
                depot.total_orders.append(combined_order)


    # 若其中一个订单是其他合成订单的子订单（不应该存在两个订单都是其他合成订单的子订单的情况）
    else:
        # print(order1['order_name'], '和', order2['order_name'], '不都是子订单')
        if order1['station_name'] == order2['station_name']:
            # 找到先前的合成订单
            # 若两个订单的is_combined都为1，则为错误，一定有一个订单为新订单
            for old_combined_order in depot.combined_orders:
                # 若是子订单1被重新合成，找出先前合成订单的子订单2，并恢复到总订单里
                if old_combined_order['original_order1'] == order1['order_name'] and order1['is_combined'] == 1:
                    # print(order1['order_name'], '是已合成的，所以拆分原合成订单')
                    for order in depot.gasoline_orders:
                        if order['order_name'] == old_combined_order['original_order2']:

                            order['is_combined'] = 0 # 将恢复的订单变为未合成状态
                            depot.total_orders.append(order)
                            # print('将子订单改为未合成并重新加入总订单列表内：', order['order_name'])
                    # 删除先前的合并订单
                    order1['is_combined'] = 0
                    for l in depot.total_orders:
                        if l['order_name']==old_combined_order['order_name']:
                            depot.total_orders.remove(l)
                            break
                    for k in depot.combined_orders:
                        if k['order_name']==old_combined_order['order_name']:
                            depot.combined_orders.remove(k)
                            break

                    # print('总订单列表移除合成订单：', old_combined_order)
                    # print('当前所有订单：',depot.total_orders)
                    # print('当前所有合成订单：', depot.combined_orders)
                    # 删除总订单列表中的一个子订单
                    depot.total_orders.remove(order2)
                elif old_combined_order['original_order2'] == order1['order_name'] and order1['is_combined'] == 1:
                    # print(order1['order_name'], '是已合成的，所以拆分原合成订单')
                    for order in depot.gasoline_orders:
                        if order['order_name'] == old_combined_order['original_order1']:
                            order['is_combined'] = 0 # 将恢复的订单变为未合成状态
                            depot.total_orders.append(order)
                            # print('将子订单改为未合成并重新加入总订单列表内：', order['order_name'])
                    # 删除先前的合并订单
                    order1['is_combined'] =0
                    for l in depot.total_orders:
                        if l['order_name'] == old_combined_order['order_name']:
                            depot.total_orders.remove(l)
                            break
                    for k in depot.combined_orders:
                        if k['order_name'] == old_combined_order['order_name']:
                            depot.combined_orders.remove(k)
                            break
                    # print('总订单列表移除合成订单：', old_combined_order)
                    # print('当前所有订单：', depot.total_orders)
                    # print('当前所有合成订单：', depot.combined_orders)
                    # 删除总订单列表中的一个子订单
                    depot.total_orders.remove(order2)
                # 若是子订单2被重新合成，找出先前合成订单的子订单1，并恢复到总订单里
                elif old_combined_order['original_order1'] == order2['order_name'] and order2['is_combined'] == 1:
                    # print(order2['order_name'], '是已合成的，所以拆分原合成订单')
                    for order in depot.gasoline_orders:
                        if order['order_name'] == old_combined_order['original_order2']:
                            order['is_combined'] = 0  # 将恢复的订单变为未合成状态
                            depot.total_orders.append(order)
                            # print('将子订单改为未合成并重新加入总订单列表内：', order['order_name'])
                    # 删除先前的合并订单
                    order2['is_combined'] =0
                    for l in depot.total_orders:
                        if l['order_name']==old_combined_order['order_name']:
                            depot.total_orders.remove(l)
                            break
                    for k in depot.combined_orders:
                        if k['order_name']==old_combined_order['order_name']:
                            depot.combined_orders.remove(k)
                            break
                    # print('总订单列表移除合成订单：', old_combined_order)
                    #
                    # print('当前所有订单：', depot.total_orders)
                    # print('当前所有合成订单：', depot.combined_orders)
                    # 删除总订单列表中的一个子订单
                    depot.total_orders.remove(order1)
                elif old_combined_order['original_order2'] == order2['order_name'] and order2['is_combined'] == 1:
                    # print(order2['order_name'], '是已合成的，所以拆分原合成订单')
                    for order in depot.gasoline_orders:
                        if order['order_name'] == old_combined_order['original_order1']:
                            order['is_combined'] = 0  # 将恢复的订单变为未合成状态
                            depot.total_orders.append(order)
                            # print('将子订单改为未合成并重新加入总订单列表内：', order['order_name'])
                    # 删除先前的合并订单
                    order2['is_combined'] =0
                    for l in depot.total_orders:
                        if l['order_name']==old_combined_order['order_name']:
                            depot.total_orders.remove(l)
                            break
                    for k in depot.combined_orders:
                        if k['order_name']==old_combined_order['order_name']:
                            depot.combined_orders.remove(k)
                            break
                    # print('总订单列表移除合成订单：',old_combined_order)
                    #
                    # print('当前所有订单：', depot.total_orders)
                    # print('当前所有合成订单：', depot.combined_orders)
                    # 删除总订单列表中的一个子订单
                    depot.total_orders.remove(order1)
            combined_order = {
                'order_name': f'{oil_class}_double_{order1["station_name"]}_{step_num}',
                'station_name': order1['station_name'],
                'depot': order1['depot'],
                'time_to_empty': min(order1['time_to_empty'], order2['time_to_empty']),
                'vehicle_type': 'Double_gasoline',
                'oil_class': oil_class,
                'path': path,
                'distance': distance.get_path_distance(distance.alldistance, path),
                'cost': cost,
                #订单等待
                'time1': max(order1['time1'], order2['time1']),
                'time2': 0,
                #订单等待

                'order_type': 'not_combined'
            }
            # 删除总订单列表中的两个子订单,同时在单舱列表里删除两个子订单，两个子订单不需要被再次合成
            depot.gasoline_orders.remove(order2)
            depot.gasoline_orders.remove(order1)
            # print(order1['order_name'], order2['order_name'], '被移除单仓汽油订单')
            if combined_order['order_name'] not in depot.total_orders:
                # 将合并后的订单添加到总订单列表中
                # print(order1['order_name'], '+', order2['order_name'], '将合成')
                depot.total_orders.append(combined_order)


def combine_orders_diesel(order1, order2, depot,path,cost,step_num,i,j):
    # 若两个订单都不是其他合成订单的子订单
#     # print('进入合成柴油函数')
    if order1['is_combined'] == 0 and order2['is_combined'] == 0:
#         # print('之前未合成订单：', order1)
#         # print('之前未合成订单：', order2)
        # 若两个订单的要运送的油站相同
        if order1['station_name'] == order2['station_name']:
            combined_order = {
                'order_name': f'diesel_double_{order1["station_name"]}_{step_num}', 
                'station_name': order1['station_name'],
                'depot': order1['depot'],
                'time_to_empty': min(order1['time_to_empty'], order2['time_to_empty']),
                'vehicle_type': 'Double_diesel',
                'oil_class': 'derv',
                'path': path,
                'distance': distance.get_path_distance(distance.alldistance, path),
                'cost': cost,
                #订单等待
                'time1': max(order1['time1'], order2['time1']),
                'time2': 0,
                #订单等待

                'order_type': 'not_combined'
            }
            # 删除总订单列表中的两个子订单，同时在单舱列表里删除两个子订单，两个子订单不需要被再次合成
#             # print('总订单移除但生成同站订单：', order1)
#             # print('总订单移除但生成同站订单：', order2)
            depot.total_orders.remove(order1)
            depot.total_orders.remove(order2)
#             # print('柴油单仓移除：', order1)
#             # print('柴油单仓移除：', order2)
            depot.diesel_orders.remove(order1)
            depot.diesel_orders.remove(order2)
            if combined_order['order_name'] not in depot.total_orders:
                # 将合并后的订单添加到总订单列表以及合成列表中
                depot.total_orders.append(combined_order)

    # 若其中一个订单是其他合成订单的子订单（不应该存在两个订单都是其他合成订单的子订单的情况）
    else:
#         # print('之前已经合成订单：', order1)
#         # print('之前已经合成订单：', order2)
        if order1['station_name'] == order2['station_name']:
            # 找到先前的合成订单
            # 若两个订单的is_combined都为1，则为错误，一定有一个订单为新订单
            for old_combined_order in depot.combined_orders :
                # 若是子订单1被重新合成，找出先前合成订单的子订单2，并恢复到总订单里
                if old_combined_order['original_order1'] == order1['order_name'] and order1['is_combined'] == 1:
                    for order in depot.diesel_orders:
                        if order['order_name'] == old_combined_order['original_order2']:
                            order['is_combined']=0
                            depot.total_orders.append(order)
                    # 删除先前的合并订单
                    order1['is_combined'] =0
                    depot.total_orders.remove(old_combined_order)
                    depot.combined_orders.remove(old_combined_order)
                    # 删除总订单列表中的一个子订单
                    depot.total_orders.remove(order2)
                elif old_combined_order['original_order2'] == order1['order_name'] and order1['is_combined'] == 1:
                    for order in depot.diesel_orders:
                        if order['order_name'] == old_combined_order['original_order1']:
                            order['is_combined'] = 0
                            depot.total_orders.append(order)
                    # 删除先前的合并订单
                    order1['is_combined'] = 0
                    depot.total_orders.remove(old_combined_order)
                    depot.combined_orders.remove(old_combined_order)
                    # 删除总订单列表中的一个子订单
                    depot.total_orders.remove(order2)
                # 若是子订单2被重新合成，找出先前合成订单的子订单1，并恢复到总订单里
                elif old_combined_order['original_order1'] == order2['order_name'] and order2['is_combined'] == 1:
                    for order in depot.diesel_orders:
                        if order['order_name'] == old_combined_order['original_order2']:
                            order['is_combined'] = 0
                            depot.total_orders.append(order)
                    # 删除先前的合并订单
                    order2['is_combined'] = 0
                    depot.total_orders.remove(old_combined_order)
                    depot.combined_orders.remove(old_combined_order)
                    # 删除总订单列表中的一个子订单
                    depot.total_orders.remove(order1)

#                     # print("已删除")
                elif old_combined_order['original_order2'] == order2['order_name'] and order2['is_combined'] == 1:
                    for order in depot.diesel_orders:
                        if order['order_name'] == old_combined_order['original_order1']:
                            order['is_combined'] = 0
                            depot.total_orders.append(order)
                    # 删除先前的合并订单
                    order2['is_combined'] = 0
                    depot.total_orders.remove(old_combined_order)
                    depot.combined_orders.remove(old_combined_order)
                    # 删除总订单列表中的一个子订单
                    depot.total_orders.remove(order1)

#                     # print("已删除")
            # 合成新的合成订单
            combined_order = {
                'order_name': f'diesel_double_{order1["station_name"]}_{step_num}',
                'station_name': order1['station_name'],
                'depot': order1['depot'],
                'time_to_empty': min(order1['time_to_empty'], order2['time_to_empty']),
                'vehicle_type': 'Double_diesel',
                'oil_class': 'derv',
                'path': path,
                'distance': distance.get_path_distance(distance.alldistance, path),
                'cost': cost,
                #订单等待
                'time1': max(order1['time1'], order2['time1']),
                'time2': 0,
                #订单等待

                'order_type': 'not_combined'
            }
            depot.diesel_orders.remove(order2)
            depot.diesel_orders.remove(order1)
#             # print("已删除")
            if combined_order['order_name'] not in depot.total_orders:
                # 将合并后的订单添加到总订单列表
                depot.total_orders.append(combined_order)
