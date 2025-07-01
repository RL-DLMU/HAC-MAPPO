import check_condition
import distance
def dispatch_vehicle(order,depot_states,vehicle_states,depot,station_states,tube):

    vehicle= check_condition.check_condition(order,depot)

    if vehicle == None:
            # print('配送失败')
            return False,None

    depot_state=depot_states[depot.id]
    vehicle_state=vehicle_states[vehicle.truck_id]
    # print('派车')
    vehicle_type = order['vehicle_type']
    # print(vehicle_state)
    # print(vehicle.truck_id)
    if  order['order_type'] == 'combined':
        vehicle.oil_class_1=order['oil_class_1']
        vehicle.oil_class_2=order['oil_class_2']
        vehicle.current_refuel_cabin1 = order['oil_class_1']
        vehicle.current_refuel_cabin2 = order['oil_class_2']
        vehicle.order_type = order['order_type']
        vehicle_state["target labeling"] += 1


        if vehicle_type == 'Double_gasoline':
            if order['oil_class_1']!=order['oil_class_2']:
                depot_state['vehicle_types_count'][2] -= 1
                vehicle.target = order['path']
                order['vehicle'] = vehicle.truck_id

                vehicle.order = order
                # 改动
                station_state1 = station_states[vehicle.target[1]]

                station_state2 = station_states[vehicle.target[2]]
                # 派车改动
                if order['oil_class_1'] == '92':
                    station_state1['Vehicle dispatch count_92'] += 1
                    station_state2['Vehicle dispatch count_95'] += 1
                else:
                    station_state1['Vehicle dispatch count_95'] += 1
                    station_state2['Vehicle dispatch count_92'] += 1
                # 派车改动

                vehicle_state['distance_traveled'] = distance.alldistance[(order['path'][0], order['path'][1])]
                if depot_state["refueling_vehicles_count"][0] < tube:
                    result = order['oil_class_1'] if order['oil_class_1'] == '92' else order['oil_class_2']
                    vehicle.current_refuel_cabin1 =result
                    vehicle.current_refuel_cabin2='95'
                    # print(vehicle.truck_id)
                    # print(vehicle.order)
                    # print('首发派双舱不同车92装油')
                    # print('正在装92油车数', depot_state["refueling_vehicles_count"][2])
                    vehicle_state["status"] = 4
                    depot_state["refueling_vehicles_count"][0] += 1
                elif depot_state["refueling_vehicles_count"][1] < tube:
                    result = order['oil_class_1'] if order['oil_class_1'] == '95' else order['oil_class_2']
                    vehicle.current_refuel_cabin1 = result
                    vehicle.current_refuel_cabin2 = '92'
                    vehicle_state["status"] = 4
                    # print(vehicle.truck_id)
                    # print(vehicle.order)
                    # print('首发派双舱不同车95装油')
                    # print('正在装95油车数',depot_state["refueling_vehicles_count"][1])
                    depot_state["refueling_vehicles_count"][1] += 1
                    # print('派车直接在', depot.id, '95装油')
                    # print('正在装95油车数',depot_state["refueling_vehicles_count"][1])
                else:
                    shortest_queue = depot.queue_92 if depot.queue_92.qsize() < depot.queue_95.qsize() else depot.queue_95
                    if shortest_queue == depot.queue_92:
                        vehicle_state["status"] = 2
                        result = order['oil_class_1'] if order['oil_class_1'] == '92' else order['oil_class_2']
                        vehicle.current_refuel_cabin1 = result
                        vehicle.current_refuel_cabin2 = '95'
                        depot_state["waiting_vehicles_count"][0] += 1
                        depot.queue_92.put(vehicle)
                    else:
                        # print('首发派双舱不同车95等待')
                        result = order['oil_class_1'] if order['oil_class_1'] == '95' else order['oil_class_2']
                        vehicle.current_refuel_cabin1 = result
                        vehicle.current_refuel_cabin2 = '92'
                        vehicle_state["status"] = 2
                        depot_state["waiting_vehicles_count"][1] += 1
                        depot.queue_95.put(vehicle)


                # print('派合成订单汽油车完毕,车辆为:'+vehicle.truck_id)
                return True,vehicle_type
            else:
                if order['oil_class_1'] =='92':
                    # 派车改动

                    depot_state['vehicle_types_count'][2] -= 1
                    vehicle.target = order['path']
                    station_state1 = station_states[vehicle.target[1]]
                    station_state2 = station_states[vehicle.target[2]]
                    station_state1['Vehicle dispatch count_92'] += 1
                    station_state2['Vehicle dispatch count_92'] += 1
                    vehicle_state['distance_traveled'] = distance.alldistance[(order['path'][0], order['path'][1])]
                    if depot_state["refueling_vehicles_count"][0] < tube:
                        vehicle_state["status"] = 4
                        depot_state["refueling_vehicles_count"][0] += 1
                    else:
                        vehicle_state["status"] = 2
                        depot_state["waiting_vehicles_count"][0] += 1
                        depot.queue_92.put(vehicle)
                    order['vehicle'] = vehicle.truck_id

                    vehicle.order = order
                    # print('派车完毕,车辆为:'+vehicle.truck_id)
                    return True, vehicle_type
                else:

                    depot_state['vehicle_types_count'][2] -= 1
                    order['vehicle'] = vehicle.truck_id

                    vehicle.order = order
                    vehicle.target = order['path']
                    # 派车改动
                    station_state1 = station_states[vehicle.target[1]]
                    station_state2 = station_states[vehicle.target[2]]
                    station_state1['Vehicle dispatch count_95'] += 1
                    station_state2['Vehicle dispatch count_95'] += 1
                    vehicle_state['distance_traveled'] = distance.alldistance[(order['path'][0], order['path'][1])]
                    if depot_state["refueling_vehicles_count"][1] < tube:
                        vehicle_state["status"] = 4
                        # print(vehicle.truck_id)
                        # print(vehicle.order)
                        # print('首发派车95装油')
                        # print('正在装95油车数',depot_state["refueling_vehicles_count"][1])
                        depot_state["refueling_vehicles_count"][1] += 1
                        # print('派车直接在', depot.id, '95装油')
                        # print('正在装95油车数',depot_state["refueling_vehicles_count"][1])
                    else:
                        vehicle_state["status"] = 2
                        # print('首发派车95等待')
                        depot_state["waiting_vehicles_count"][1] += 1
                        depot.queue_95.put(vehicle)


                    # print('派车完毕,车辆为:'+vehicle.truck_id)
                    return True,vehicle_type
        elif vehicle_type == 'Double_diesel':

            depot_state['vehicle_types_count'][3] -= 1
            if depot_state["refueling_vehicles_count"][2] < tube:
                vehicle_state["status"] = 4
                depot_state["refueling_vehicles_count"][2] += 1
            else:
                vehicle_state["status"] = 2
                depot_state["waiting_vehicles_count"][2] += 1
                depot.queue_derv.put(vehicle)
            vehicle.target = order['path']
            # 派车改动
            station_state1 = station_states[vehicle.target[1]]
            station_state2 = station_states[vehicle.target[2]]
            station_state1['Vehicle dispatch count_derv'] += 1
            station_state2['Vehicle dispatch count_derv'] += 1


            vehicle_state['distance_traveled'] = distance.alldistance[(order['path'][0], order['path'][1])]

            order['vehicle'] = vehicle.truck_id

            vehicle.order = order
            # print('派合成订单柴油车完毕 车辆为:'+vehicle.truck_id)
            return True,vehicle_type
    else:
        vehicle.order_type = order['order_type']
        if order['oil_class'] == '92':  # 只需要92的油
            if vehicle_type == 'Single_gasoline':
                depot_state['vehicle_types_count'][0] -= 1
                vehicle.current_refuel_cabin1 = '92'

            elif vehicle_type == 'Double_gasoline':
                depot_state['vehicle_types_count'][2] -= 1
                vehicle.current_refuel_cabin1 = '92'
                vehicle.current_refuel_cabin2 = '92'

            vehicle.target = order['path']
            # 改动
            station_state = station_states[vehicle.target[1]]
            station_state['Vehicle dispatch count_92'] += 1

            vehicle_state["target labeling"] += 1
            vehicle_state['distance_traveled'] = distance.alldistance[(order['path'][0],order['path'][1])]
            if depot_state["refueling_vehicles_count"][0] < tube:
                vehicle_state["status"] = 4
                depot_state["refueling_vehicles_count"][0] += 1
            else:
                vehicle_state["status"] = 2
                depot_state["waiting_vehicles_count"][0] += 1
                depot.queue_92.put(vehicle)

            order['vehicle'] = vehicle.truck_id

            vehicle.order = order
            # print('派车完毕,车辆为:' + vehicle.truck_id)
            return True,vehicle_type
        elif order['oil_class'] == '95':  # 只需要95的油

            if vehicle_type == 'Single_gasoline':
                depot_state['vehicle_types_count'][0] -= 1
                vehicle.current_refuel_cabin1 = '95'

            elif vehicle_type == 'Double_gasoline':
                depot_state['vehicle_types_count'][2] -= 1
                vehicle.current_refuel_cabin1 = '95'
                vehicle.current_refuel_cabin2 = '95'

            vehicle.target = order['path']
            order['vehicle'] = vehicle.truck_id

            vehicle.order = order
            # 改动
            station_state = station_states[vehicle.target[1]]
            station_state['Vehicle dispatch count_95'] += 1

            vehicle_state["target labeling"] += 1
            vehicle_state['distance_traveled'] = distance.alldistance[(order['path'][0],order['path'][1])]
            if depot_state["refueling_vehicles_count"][1] < tube:
                vehicle_state["status"] = 4
                # print(vehicle.truck_id)
                # print(vehicle.order)
                # print('首发派车95装油')
                # print('正在装95油车数',depot_state["refueling_vehicles_count"][1])
                depot_state["refueling_vehicles_count"][1] += 1
                # print('派车直接在', depot.id, '95装油')
                # print('正在装95油车数',depot_state["refueling_vehicles_count"][1])
            else:
                vehicle_state["status"] = 2
                depot_state["waiting_vehicles_count"][1] += 1
                depot.queue_95.put(vehicle)


            # print('派车完毕,车辆为:' + vehicle.truck_id)
            return True,vehicle_type
        elif order['oil_class'] == '92+95'or order['oil_class'] == '95+92' :  # 需要92和95的油
            depot_state['vehicle_types_count'][2] -= 1
            vehicle.current_refuel_cabin1 = '92'
            vehicle.current_refuel_cabin2 = '95'
            vehicle.target = order['path']
            order['vehicle'] = vehicle.truck_id

            vehicle.order = order
            # 改动
            station_state = station_states[vehicle.target[1]]
            station_state['Vehicle dispatch count_92'] += 1
            station_state['Vehicle dispatch count_95'] += 1

            vehicle_state["target labeling"] += 1
            vehicle_state['distance_traveled'] = distance.alldistance[(order['path'][0],order['path'][1])]
            if depot_state["refueling_vehicles_count"][0] < tube:
                vehicle.current_refuel_cabin1 = '92'
                vehicle.current_refuel_cabin2 = '95'
                vehicle_state["status"] = 4
                depot_state["refueling_vehicles_count"][0] += 1
            elif depot_state["refueling_vehicles_count"][1] < tube:
                vehicle.current_refuel_cabin1 = '95'
                vehicle.current_refuel_cabin2 = '92'
                vehicle_state["status"] = 4
                # print(vehicle.truck_id)
                # print(vehicle.order)
                # print('首发派车95装油')
                # print('正在装95油车数',depot_state["refueling_vehicles_count"][1])
                depot_state["refueling_vehicles_count"][1] += 1
                # print('派车直接在', depot.id, '95装油')
                # print('正在装95油车数',depot_state["refueling_vehicles_count"][1])
            else:
                shortest_queue = depot.queue_92 if depot.queue_92.qsize() < depot.queue_95.qsize() else depot.queue_95
                if shortest_queue == depot.queue_92:
                    vehicle.current_refuel_cabin1 = '92'
                    vehicle.current_refuel_cabin2 = '95'
                    vehicle_state["status"] = 2
                    depot_state["waiting_vehicles_count"][0] += 1
                    depot.queue_92.put(vehicle)
                else:
                    vehicle.current_refuel_cabin1 = '95'
                    vehicle.current_refuel_cabin2 = '92'
                    vehicle_state["status"] = 2
                    depot_state["waiting_vehicles_count"][1] += 1
                    depot.queue_95.put(vehicle)


            # print('派车完毕,车辆为:' + vehicle.truck_id)
            return True,vehicle_type
        elif order['oil_class'] == 'derv':  # 只需要柴油

            if vehicle_type == 'Single_diesel':
                depot_state['vehicle_types_count'][1] -= 1
                vehicle.current_refuel_cabin1 = 'derv'

            elif vehicle_type == 'Double_diesel':
                depot_state['vehicle_types_count'][3] -= 1
                vehicle.current_refuel_cabin1 = 'derv'
                vehicle.current_refuel_cabin2 = 'derv'

            vehicle.target = order['path']
            # 改动
            station_state = station_states[vehicle.target[1]]
            station_state['Vehicle dispatch count_derv'] += 1

            vehicle_state["target labeling"] += 1
            vehicle_state['distance_traveled'] = distance.alldistance[(order['path'][0],order['path'][1])]
            if depot_state["refueling_vehicles_count"][2] < tube:
                vehicle_state["status"] = 4
                depot_state["refueling_vehicles_count"][2] += 1
            else:
                vehicle_state["status"] = 2
                depot_state["waiting_vehicles_count"][2] += 1
                depot.queue_derv.put(vehicle)

            order['vehicle'] = vehicle.truck_id

            vehicle.order = order
            # print('派车完毕,车辆为:' + vehicle.truck_id)
            return True,vehicle_type
    # a=vehicle
    # return False,None


