
def choose_order_92(action_space, station, station_states,depot_id,step_num):

        for station_id,station_state in station_states.items():
            if station_id == station.station_id:
                station_state = station_state
                break
        if action_space == 0:
            # 无需配送
            return None
        elif action_space==1:

                order={}
                order['order_name']=f'92_single_{str(station.station_id)}_{step_num}' # 订单名字增加step数，防止订单名字重复
                order['station_name'] = station.station_id
                order['depot'] = depot_id
                order['time_to_empty'] = station_state['time_to_empty'][0]
                order['vehicle_type'] = 'Single_gasoline'
                order['oil_class'] = '92'
                order['time1'] = 0
                order['time2'] = 0
                order['is_combined'] = 0

                return order


        elif action_space==2:
            # vehicle_type='Double_gasoline'
            order = {}
            order['order_name'] = f'92_double_{str(station.station_id)}_{step_num}'
            order['station_name'] = station.station_id
            order['depot'] = depot_id
            order['time_to_empty'] = station_state['time_to_empty'][0]
            order['vehicle_type'] = 'Double_gasoline'
            order['oil_class'] = '92'
            order['time1'] = 0
            order['time2'] = 0
            order['is_combined'] = 0

            return order

def choose_order_95(action_space, station, station_states, depot_id, step_num):
    for station_id, station_state in station_states.items():
        if station_id == station.station_id:
            station_state = station_state
            break
    if action_space == 0:
        # 无需配送
        return None
    elif action_space == 1:
        order = {}
        order['order_name'] = f'95_single_{str(station.station_id)}_{step_num}'
        order['station_name'] = station.station_id
        order['depot'] = depot_id
        order['time_to_empty'] = station_state['time_to_empty'][1]
        order['vehicle_type'] = 'Single_gasoline'
        order['oil_class'] = '95'
        order['time1'] = 0
        order['time2'] = 0
        order['is_combined'] = 0

        return order
    elif action_space == 2:
            # vehicle_type='Double_gasoline'
            order = {}
            order['order_name'] = f'95_double_{str(station.station_id)}_{step_num}'
            order['station_name'] = station.station_id
            order['depot'] = depot_id
            order['time_to_empty'] = station_state['time_to_empty'][1]
            order['vehicle_type'] = 'Double_gasoline'
            order['oil_class'] = '95'
            order['time1'] = 0
            order['time2'] = 0
            order['is_combined'] = 0

            return order

def choose_order_derv(action_space, station, station_states, depot_id, step_num):
    for station_id, station_state in station_states.items():
        if station_id == station.station_id:
            station_state = station_state
            break

    if action_space == 0:
        # 无需配送
        return None

    elif action_space == 1:
        order = {}
        order['order_name'] = f'diesel_single_{str(station.station_id)}_{step_num}'
        order['station_name'] = station.station_id
        order['depot'] = depot_id
        order['time_to_empty'] = station_state['time_to_empty'][2]
        order['vehicle_type'] = 'Single_diesel'
        order['oil_class'] = 'derv'
        order['time1'] = 0
        order['time2'] = 0
        order['is_combined'] = 0
        # print('生成：',order)

        return order

    elif action_space == 2:

            # vehicle_type='Double_diesel'
            order = {}
            order['order_name'] = f'diesel_double_{str(station.station_id)}_{step_num}'
            order['station_name'] = station.station_id
            order['depot'] = depot_id
            order['time_to_empty'] = station_state['time_to_empty'][2]
            order['vehicle_type'] = 'Double_diesel'
            order['oil_class'] = 'derv'
            order['time1'] = 0
            order['time2'] = 0
            order['is_combined'] = 0

            return order



