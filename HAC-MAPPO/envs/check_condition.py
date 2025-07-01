
import data

def check_condition(order,depot):

            # 提取列表和距离
            path=order['path']
            distance =order['distance']
            f=order['time_to_empty']
            vehicle_type=order['vehicle_type']
            depot_vehicle_list=getattr(data,path[0])
            if vehicle_type == 'Single_gasoline':
                vehicle_queue = depot.Single_gasoline
                speed = depot_vehicle_list[0]['Single_gasoline'][0]['speed']
            elif vehicle_type == 'Single_diesel':
                vehicle_queue = depot.Single_diesel
                speed = depot_vehicle_list[1]['Single_diesel'][0]['speed']
            elif vehicle_type == 'Double_gasoline':
                vehicle_queue = depot.Double_gasoline
                speed = depot_vehicle_list[2]['Double_gasoline'][0]['speed']
            else:
                vehicle_queue = depot.Double_diesel
                speed = depot_vehicle_list[3]['Double_diesel'][0]['speed']

            if not vehicle_queue.empty():
                vehicle = vehicle_queue.get()
                # print('派车成功')
                return vehicle
            else:
                # 处理队列为空的情况
                # print('派车失败')
                return None



