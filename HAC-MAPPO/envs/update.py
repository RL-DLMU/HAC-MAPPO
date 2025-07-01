def update_all_order_tte(alldepots,station_states):
    for depot in alldepots:

        for order in depot.total_orders:

            if order['order_type'] == 'combined':

                i = {'92': 0, '95': 1}.get(order['oil_class_1'], 2)
                j = {'92': 0, '95': 1}.get(order['oil_class_2'], 2)
                order['time_to_empty'] = min(station_states[order['station_name1']]['time_to_empty'][i],station_states[order['station_name2']]['time_to_empty'][j])
            else:
                if order['oil_class'] == '92+95':

                    order['time_to_empty'] = min(station_states[order['station_name']]['time_to_empty'][0],station_states[order['station_name']]['time_to_empty'][1])
                else:

                    i = {'92': 0, '95': 1}.get(order['oil_class'], 2)
                    order['time_to_empty'] = station_states[order['station_name']]['time_to_empty'][i]


        for order in depot.gasoline_orders:

            i = {'92': 0, '95': 1}.get(order['oil_class'], 2)
            order['time_to_empty'] = station_states[order['station_name']]['time_to_empty'][i]


        for order in depot.diesel_orders:

            order['time_to_empty'] = station_states[order['station_name']]['time_to_empty'][2]


        for order in depot.combined_orders:

            i = {'92': 0, '95': 1}.get(order['oil_class_1'], 2)
            j = {'92': 0, '95': 1}.get(order['oil_class_2'], 2)
            order['time_to_empty'] = min(station_states[order['station_name1']]['time_to_empty'][i],
                                         station_states[order['station_name2']]['time_to_empty'][j])

