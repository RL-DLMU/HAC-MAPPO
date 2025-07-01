import math
import queue
import time

import numpy as np
from gym import spaces
import New_characters
from queue import Queue
import station_action
import cal_oil
import distance
import dispatch_vehicle
import update
import get_path
import random
import simulate

class OilDeliveryEnv1():
    def __init__(self):
        super(OilDeliveryEnv1, self).__init__()
        self.tube=6
        self.singe_dis = 0
        self.wait_time=0
        self.double_dis = 0
        self.safe_time=0
        self.dis_cost=0
        self.dis_time=0
        self.over_order=0
        self.safe_dis = 0
        self.empty_time=0
        self.full_time = 0
        self.dc=0
        self.empty92=0
        self.empty95 = 0
        self.emptyderv = 0
        self.full92 = 0
        self.full95 = 0
        self.fullderv = 0
        self.dis92 = 0
        self.dis95 = 0
        self.disderv = 0
        self.gas_single = 0
        self.gas_double = 0
        self.derv_single = 0
        self.derv_double = 0
        self.initial_state =[]
        self.next_state=[]
        self.step_minutes=5
        self.alldepots=New_characters.alldepots
        self.allstations=New_characters.allstations
        self.all_vehicle=New_characters.all_vehicle
        self.veh_num=len(self.all_vehicle)/8
        self.alldistance=distance.alldistance
        self.step_num=0
        self.refuel_time=10 # 装油的时间
        self.service_time=10# 服务的时间
        self.emco =10
        self.single_cost = 20
        self.double_cost = 40
        self.initial_state = []
        self.next_state = []
        self.station_states = {}
        self.depot_states = {}
        self.vehicle_states = {}
        self.terminate=False
        # self.reward=0
        # 定义油库状态空间
        self.oil_depot_space = spaces.Dict({
            "vehicle_types_count": spaces.MultiDiscrete([10] * 4),  # 4种类型车各自的数量
            "refueling_vehicles_count": spaces.MultiDiscrete([4] * 3),  # 92号、95号、柴油此时正在装油的车辆数
            "waiting_vehicles_count": spaces.MultiDiscrete([10] * 3)  # 92号、95号、柴油此时正在等待装油的车辆数
        })

        # 定义车辆状态空间
        self.vehicle_space = spaces.Dict({
            "distance_traveled": spaces.Discrete(1000),  # 距离目标剩余距离
            "status": spaces.Discrete(5),  # 车辆当前状态(未配送：0，行驶中：1，等待中：2，服务中：3，装油中：4)
            "total_time": spaces.Discrete(100),  # 路程总时间
            "time_elapsed": spaces.Discrete(100),  # 已行驶时间
            "service_time": spaces.Discrete(100),  # 已服务时间
            "wait_time": spaces.Discrete(100),  # 等待时间
            "refuel_time": spaces.Discrete(100),  # 已装油花费的时间
            "vehicle_type": spaces.Discrete(4),  # 车辆类型
            "oil_tank_empty_cabin1":spaces.Discrete(2),  # 只能取 0 或 1
            "oil_tank_empty_cabin2":spaces.Discrete(2),  # 只能取 0 或 1
            "target labeling":spaces.Discrete(3) #

        })

        # 定义加油站状态空间
        self.gas_station_space = spaces.Dict({
            "92_gas": spaces.Discrete(100),  # 92号油量
            "95_gas": spaces.Discrete(100),  # 95号油量
            "diesel": spaces.Discrete(100),  # 柴油量
            "vehicle_num": spaces.MultiDiscrete([100] * 3),  # 三种不同油品等待队列中油车的数量
            "service_status": spaces.MultiBinary(3),  # 三种不同油品队列是否有车在服务中（有车在服务中：1，没车在服务中：0）
            "time_to_empty": spaces.MultiDiscrete([100] * 3),  # 三种油品距离耗尽的剩余时间
            'Vehicle dispatch single_92': spaces.Discrete(10),
            'Vehicle dispatch single_95': spaces.Discrete(10),
            'Vehicle dispatch single_diesel': spaces.Discrete(10),
            'Vehicle dispatch double_92': spaces.Discrete(10),
            'Vehicle dispatch double_95': spaces.Discrete(10),
            'Vehicle dispatch double_diesel': spaces.Discrete(10),
            'Vehicle dispatch single_gasoline': spaces.Discrete(10),
            'Vehicle dispatch double_gasoline': spaces.Discrete(10),
            # 'Vehicle dispatch single_diesel': spaces.Discrete(10),
            # 'Vehicle dispatch double_diesel': spaces.Discrete(10),
            'Vehicle dispatch count_92':spaces.Discrete(10),
            'Vehicle dispatch count_95':spaces.Discrete(10),
            'Vehicle dispatch count_derv':spaces.Discrete(10),# 为该加油站配送的车辆数
            "oil_percentage": spaces.MultiBinary(3),
            # "service_progress_92": spaces.Discrete(101),  # 92号油装油进度（0到100）
            # "service_progress_95": spaces.Discrete(101),  # 95号油装油进度（0到100）
            # "service_progress_diesel": spaces.Discrete(101)  # 柴油装油进度（0到100）
        })
        self.action_space = []
        self.action_space1 = []
        self.action_space2 = []
        self.observation_space = []
        self.share_observation_space = []
        #定义动作空间
        self.agent_num=40
        self.obs_dim=9

        self.action_dim=3
        total_action_space = []
        # total_action_space1 = []
        total_action_space2 = []
        share_obs_dim = self.agent_num * self.obs_dim
        for agent_idx in range(self.agent_num):
            u_action_space = spaces.Discrete(3)  # 3个离散的动作
            total_action_space.append(u_action_space)
            self.action_space.append(total_action_space[agent_idx])


        # for agent_idx in range(self.agent_num):
        #     u_action_space1 = spaces.Discrete(2)  # 3个离散的动作
        #     total_action_space1.append(u_action_space1)
        #     self.action_space1.append(total_action_space1[agent_idx])

        for agent_idx in range(self.agent_num):
            u_action_space2 = spaces.Discrete(3)  # 3个离散的动作
            total_action_space2.append(u_action_space2)
            self.action_space2.append(total_action_space2[agent_idx])
        for _ in range(self.agent_num):
            self.observation_space.append(
                spaces.Box(
                    low=-np.inf,
                    high=+np.inf,
                    shape=(self.obs_dim,),
                    dtype=np.float32
                )
            )
        self.share_observation_space = [
            spaces.Box(low=-np.inf, high=+np.inf, shape=(share_obs_dim,), dtype=np.float32)
            for _ in range(self.agent_num)
        ]
        # self.action_space = spaces.Discrete(5)
        # self.action_space = [spaces.Discrete(3) for _ in range(6)]
        # self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        # self.observation_space = [spaces.Box(low=-np.inf, high=+np.inf, shape=(18,), dtype=np.float32) for _ in range(6)]
        # self.share_observation_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(36,), dtype=np.float32)

        # 每个变量的最高值




    def step(self, actions):
        for station in self.allstations:
            station.total_reward = 0
            station.emp_cost=0
        targ = True
        depot_states=self.initial_state[0]
        station_states = self.initial_state[1]
        vehicle_states = self.initial_state[2]
        for station_id,station_state in station_states.items():
            # for station in self.allstations:
            #     if station.emp_cost>1:
            #         print('错了')
            #
            # if targ:
            #     if station_state["oil_percentage"][0] > 95 or station_state["oil_percentage"][0] < 5 or \
            #             station_state["oil_percentage"][1] > 95 or station_state["oil_percentage"][1] < 5 or \
            #             station_state["oil_percentage"][2] > 95 or station_state["oil_percentage"][2] < 5:
            #             targ = False
            #
            #             for station in self.allstations:
            #                 station.emp_cost += 1
            tempstation=New_characters.find_station_by_id(station_id)

            current_92_gas = station_state['92_gas']
            station_state['92_gas'],consum_92 = cal_oil.cal_remain_oil(station_id,'92', current_92_gas, self.step_num, self.step_minutes)  # 计算剩余油量
            tempstation.oil_92,consum_92=cal_oil.cal_remain_oil(station_id, '92', current_92_gas, self.step_num, self.step_minutes)
            tempstation.cosm_92 += consum_92
            current_95_gas = station_state['95_gas']
            station_state['95_gas'],consum_95 = cal_oil.cal_remain_oil(station_id,'95', current_95_gas, self.step_num, self.step_minutes)  # 计算剩余油量
            tempstation.oil_95,consum_95 = cal_oil.cal_remain_oil(station_id, '95', current_95_gas, self.step_num,self.step_minutes)
            tempstation.cosm_95 += consum_95
            current_diesel = station_state['diesel']
            station_state['diesel'],consum_derv = cal_oil.cal_remain_oil(station_id,'diesel', current_diesel, self.step_num, self.step_minutes)  # 计算剩余油量
            tempstation.oil_derv,consum_derv= cal_oil.cal_remain_oil(station_id, 'diesel', current_diesel, self.step_num, self.step_minutes)
            tempstation.cosm_derv += consum_derv
            #改动
            station_state["oil_percentage"][0]= int(tempstation.oil_92) / tempstation.capacity*100
            station_state["oil_percentage"][1]= int(tempstation.oil_95)/ tempstation.capacity*100
            station_state["oil_percentage"][2] =int(tempstation.oil_derv)/ tempstation.capacity*100
            if station_state["92_gas"] == 0 or station_state["95_gas"]==0 or station_state["diesel"]==0 or station_state["92_gas"] ==100 or station_state["95_gas"]==100 or station_state["diesel"]==100:
                 self.dc+=1
            # if station_state["92_gas"] == 0:
            #     tempstation.total_reward -=1000000
            #     # tempstation.emp_cost += 1
            #     self.empty_time +=1
            # if station_state["95_gas"] == 0:
            #     tempstation.total_reward -=1000000
            #     # tempstation.emp_cost += 1
            #     self.empty_time += 1
            # if station_state["diesel"] == 0:
            #     tempstation.total_reward -=1000000
            #     # tempstation.emp_cost += 1
            #     self.empty_time += 1
            # if station_state["92_gas"] == 100:
            #     tempstation.total_reward -=1000000
            #     # tempstation.emp_cost += 1
            #     self.full_time+= 1
            # if station_state["95_gas"] == 100:
            #     tempstation.total_reward -=1000000
            #     # tempstation.emp_cost += 1
            #     self.full_time += 1
            # if station_state["diesel"] == 100:
            #     tempstation.total_reward -=1000000
            #     # tempstation.emp_cost += 1
            #     self.full_time += 1
            if station_state["oil_percentage"][0] <= 90 and station_state["oil_percentage"][0] >= 10 and station_state["oil_percentage"][1] <= 90 and station_state["oil_percentage"][1] >=10 and station_state["oil_percentage"][2] <= 90 and station_state["oil_percentage"][2] >=10:
                self.safe_time+=1
            if station_state["oil_percentage"][0] > 95 or station_state["oil_percentage"][0] < 5 or station_state["oil_percentage"][1] > 95 or station_state["oil_percentage"][1] < 5 or station_state["oil_percentage"][2] > 95 or station_state["oil_percentage"][2] < 5:
                tempstation.emp_cost += 1
            # if station_state["oil_percentage"][0] > 95:
            #         tempstation.total_reward -= 1000000
            #         # tempstation.total_reward -=0.5
            # if station_state["oil_percentage"][0] <5:
            #     tempstation.total_reward -= 1000000
            #     # tempstation.total_reward -= 0.5
            # if station_state["oil_percentage"][1] > 95:
            #         tempstation.total_reward -=1000000
            #         # tempstation.total_reward -= 0.5
            # if station_state["oil_percentage"][1] < 5:
            #         tempstation.total_reward -= 1000000
            #         # tempstation.total_reward -= 0.5
            # if station_state["oil_percentage"][2] > 95:
            #     # tempstation.total_reward -= 0.5
            #     tempstation.total_reward -= 1000000
            # if station_state["oil_percentage"][2] < 5:
            #         tempstation.total_reward -=1000000
            #         # tempstation.total_reward -= 0.5

            station_state['time_to_empty'][0] = cal_oil.estimate_time_until_empty(station_id,'92', station_state['92_gas'], self.step_num, self.step_minutes)
            station_state['time_to_empty'][1] = cal_oil.estimate_time_until_empty(station_id,'95', station_state['95_gas'], self.step_num, self.step_minutes)
            station_state['time_to_empty'][2] = cal_oil.estimate_time_until_empty(station_id,'diesel', station_state['diesel'], self.step_num, self.step_minutes)
        update.update_all_order_tte(self.alldepots, station_states)
        #新改
        for i in range(3):
            for index, station in enumerate(self.allstations):
                # print(station.queue_95)
                depot=get_path.get_path(station,self.alldepots)
                # print(depot.id)
                action=actions[i][index]
                selected_depot = "d2" if depot.id == "d1" else "d1"
                new_depot=New_characters.find_depot_by_id(selected_depot)
                station_state=station_states[station.station_id]
                depot_state = depot_states[depot.id]
                depot2_state = depot_states[new_depot.id]
                if i==0:
                    # station_action.station_action_92(action, station, station_states, depot, self.empty_cost,
                    #                                  self.single_cost, self.double_cost, self.alldistance,
                    #                                  self.step_num)
                        if action==1:
                            if np.argmin(station_state["oil_percentage"])==0:
                                self.safe_dis+=1
                            self.dis_time+=1
                            station_state['Vehicle dispatch single_92']+=1
                            station.total_reward-=10
                            self.dis_cost+=0.01
                           
                            if  depot_state["refueling_vehicles_count"][0]<self.tube or depot_state["waiting_vehicles_count"][0]<depot2_state["waiting_vehicles_count"][0]:
                                station_action.station_action_92(action, station, station_states, depot,self.emco, self.single_cost, self.double_cost, self.alldistance,self.step_num)
                            else:
                                station_action.station_action_92(action, station, station_states, new_depot,self.emco, self.single_cost, self.double_cost, self.alldistance,self.step_num)
                        elif action==2:
                            if np.argmin(station_state["oil_percentage"])==0:
                                self.safe_dis+=1
                            self.dis_time += 1
                            station_state['Vehicle dispatch double_92'] += 1
                            station.total_reward -= 10
                            self.dis_cost += 0.01
                            if  depot_state["refueling_vehicles_count"][0]<self.tube  or depot_state["waiting_vehicles_count"][0]<depot2_state["waiting_vehicles_count"][0]:
                                station_action.station_action_92(action, station, station_states, depot,self.emco, self.single_cost, self.double_cost, self.alldistance,self.step_num)
                            else:
                                station_action.station_action_92(action, station, station_states, new_depot,self.emco, self.single_cost, self.double_cost, self.alldistance,self.step_num)
                elif i==1:
                    # station_action.station_action_95(action, station, station_states, depot, self.empty_cost,
                    #                                  self.single_cost, self.double_cost, self.alldistance,
                    #                                  self.step_num)
                    # if action == 0:
                    #     station.total_reward += 0.5
                        if action==1:
                            if np.argmin(station_state["oil_percentage"])==1:
                                self.safe_dis+=1
                            self.dis_time += 1
                            station_state['Vehicle dispatch single_95'] += 1
                            station.total_reward -=10
                            self.dis_cost += 0.01
                            if  depot_state["refueling_vehicles_count"][1]<self.tube  or depot_state["waiting_vehicles_count"][1]<depot2_state["waiting_vehicles_count"][1]:
                                station_action.station_action_95(action, station, station_states, depot,self.emco, self.single_cost, self.double_cost, self.alldistance,self.step_num)
                            else:
                                station_action.station_action_95(action, station, station_states, new_depot,self.emco, self.single_cost, self.double_cost, self.alldistance,self.step_num)
                        elif action==2:
                            if np.argmin(station_state["oil_percentage"])==1:
                                self.safe_dis+=1
                            self.dis_time += 1
                            station_state['Vehicle dispatch double_95'] += 1
                            station.total_reward -= 10
                            self.dis_cost += 0.01
                            if  depot_state["refueling_vehicles_count"][1]<self.tube  or depot_state["waiting_vehicles_count"][1]<depot2_state["waiting_vehicles_count"][1]:
                                station_action.station_action_95(action, station, station_states, depot,self.emco, self.single_cost, self.double_cost, self.alldistance,self.step_num)
                            else:
                                station_action.station_action_95(action, station, station_states, new_depot,self.emco, self.single_cost, self.double_cost, self.alldistance,self.step_num)
                elif i == 2:
                    # station_action.station_action_derv(action, station, station_states, depot, self.empty_cost,
                    #                                    self.single_cost, self.double_cost, self.alldistance,
                    #                                    self.step_num)
                    # if action == 0:
                    #     station.total_reward += 0.5

                    if action == 1:
                        if np.argmin(station_state["oil_percentage"]) == 2:
                            self.safe_dis += 1
                        self.dis_time += 1
                        station_state['Vehicle dispatch single_diesel'] += 1
                        station.total_reward -=10
                        self.dis_cost += 0.01
                        if depot_state["refueling_vehicles_count"][2] < self.tube  or depot_state["waiting_vehicles_count"][2] < \
                                depot2_state["waiting_vehicles_count"][2]:

                            station_action.station_action_derv(action, station, station_states, depot, self.emco,
                                                               self.single_cost, self.double_cost, self.alldistance,
                                                               self.step_num)
                        else :
                            station_action.station_action_derv(action, station, station_states, new_depot, self.emco,
                                                               self.single_cost, self.double_cost, self.alldistance,
                                                               self.step_num)
                    elif action == 2:
                        if np.argmin(station_state["oil_percentage"]) == 2:
                            self.safe_dis += 1
                        self.dis_time += 1
                        station_state['Vehicle dispatch double_diesel'] += 1
                        station.total_reward -= 10
                        self.dis_cost += 0.01
                        if depot_state["refueling_vehicles_count"][2] < self.tube  or depot_state["waiting_vehicles_count"][2] < \
                                depot2_state["waiting_vehicles_count"][2]:
                            station_action.station_action_derv(action, station, station_states, depot, self.emco,
                                                               self.single_cost, self.double_cost, self.alldistance,
                                                               self.step_num)
                        else:
                            station_action.station_action_derv(action, station, station_states, new_depot, self.emco,
                                                               self.single_cost, self.double_cost, self.alldistance,
                                                               self.step_num)

        #订单合成
        for d in self.alldepots:
            if len(d.gasoline_orders) != 0 and len(d.gasoline_orders) != 1:
                # 进入该 if 语句
                # print('测试:',d.gasoline_orders)
                best_route,best_cost = simulate.tabu_search(d.id, d.gasoline_orders, 100, 10)
                sta = best_route[1]
                for i in range(len(sta)):
                    #新改
                    if len(sta[i]) == 1:
                        for h in range(len(d.gasoline_orders)):
                            if d.gasoline_orders[h]['station_name'] == sta[i][0]:
                                if d.gasoline_orders[h]['is_combined']==1:
                                    for old_combined_order in d.combined_orders:
                                        if old_combined_order['original_order1'] == d.gasoline_orders[h]['order_name']:
                                            d.gasoline_orders[h]['is_combined'] = 0
                                            d.total_orders.append(d.gasoline_orders[h])
                                            for order in d.gasoline_orders:
                                                # print(d.gasoline_orders[h], '之前合成')
                                                if order['order_name'] == old_combined_order['original_order2']:
                                                    order['is_combined'] = 0  # 将恢复的订单变为未合成状态
                                                    d.total_orders.append(order)
                                                    # print('去除另一个子订单并将其重新添加至总订单中：', order)
                                                    break
                                            # 删除先前的合并订单
                                            for l in d.total_orders:
                                                if l['order_name'] == old_combined_order['order_name']:
                                                    d.total_orders.remove(l)
                                                    break
                                            for k in d.combined_orders:
                                                if k['order_name'] == old_combined_order['order_name']:
                                                    d.combined_orders.remove(k)
                                                    break
                                            # 删除总订单列表中的一个子订单

                                        elif old_combined_order['original_order2'] == d.gasoline_orders[h][
                                            'order_name']:
                                            d.gasoline_orders[h]['is_combined'] = 0
                                            d.total_orders.append(d.gasoline_orders[h])
                                            for order in d.gasoline_orders:
                                                # print(d.gasoline_orders[h], '之前合成')
                                                if order['order_name'] == old_combined_order['original_order1']:
                                                    order['is_combined'] = 0  # 将恢复的订单变为未合成状态
                                                    d.total_orders.append(order)
                                                    # print('去除另一个子订单并将其重新添加至总订单中：', order)
                                                    break
                                            # 删除先前的合并订单
                                            for l in d.total_orders:
                                                if l['order_name'] == old_combined_order['order_name']:
                                                    d.total_orders.remove(l)
                                            for k in d.combined_orders:
                                                if k['order_name'] == old_combined_order['order_name']:
                                                    d.combined_orders.remove(k)
                                            # 删除总订单列表中的一个子订单

                                        # 若是子订单2被重新合成，找出先前合成订单的子订单1，并恢复到总订单里
                                        elif old_combined_order['original_order1'] == d.gasoline_orders[h][
                                            'order_name']:
                                            d.gasoline_orders[h]['is_combined'] = 0
                                            d.total_orders.append(d.gasoline_orders[h])
                                            for order in d.gasoline_orders:
                                                # print(d.gasoline_orders[h], '之前合成')
                                                if order['order_name'] == old_combined_order['original_order2']:
                                                    order['is_combined'] = 0  # 将恢复的订单变为未合成状态
                                                    d.total_orders.append(order)
                                                    # print('去除另一个子订单并将其重新添加至总订单中：', order)
                                                    break
                                            # 删除先前的合并订单
                                            for l in d.total_orders:
                                                if l['order_name'] == old_combined_order['order_name']:
                                                    d.total_orders.remove(l)
                                            for k in d.combined_orders:
                                                if k['order_name'] == old_combined_order['order_name']:
                                                    d.combined_orders.remove(k)
                                            # 删除总订单列表中的一个子订单

                                        elif old_combined_order['original_order2'] == d.gasoline_orders[h][
                                            'order_name']:
                                            d.gasoline_orders[h]['is_combined'] = 0
                                            d.total_orders.append(d.gasoline_orders[h])
                                            # print(d.gasoline_orders[h], '之前合成')
                                            for order in d.gasoline_orders:
                                                if order['order_name'] == old_combined_order['original_order1']:
                                                    order['is_combined'] = 0  # 将恢复的订单变为未合成状态
                                                    d.total_orders.append(order)
                                                    # print('去除另一个子订单并将其重新添加至总订单中：', order)
                                                    break
                                            # 删除先前的合并订单
                                            for l in d.total_orders:
                                                if l['order_name'] == old_combined_order['order_name']:
                                                    d.total_orders.remove(l)
                                            for k in d.combined_orders:
                                                if k['order_name'] == old_combined_order['order_name']:
                                                    d.combined_orders.remove(k)
                                            # 删除总订单列表中的一个子订单
                                            # if d.gasoline_orders[m] not in d.total_orders:
                                            #     print(1)

                                d.gasoline_orders[h]['cost']=best_cost
                    #新改
                    if len(sta[i]) == 2:
                        m = 0
                        j = 0
                        for h in range(len(d.gasoline_orders)):
                            if d.gasoline_orders[h]['station_name'] == sta[i][0]:
                                m = h
                            if d.gasoline_orders[h]['station_name'] == sta[i][1]:
                                j = h
                        if d.gasoline_orders[m]["oil_class"] == d.gasoline_orders[j]["oil_class"]:
                            oil_class = d.gasoline_orders[m]['oil_class']
                        else:
                            oil_class = f'{d.gasoline_orders[m]["oil_class"]}+{d.gasoline_orders[j]["oil_class"]}'
                        if d.gasoline_orders[m]['is_combined'] == 0 and d.gasoline_orders[j]['is_combined'] == 0:
                            combined_order = {
                                'order_name': f'{d.gasoline_orders[m]["order_name"]}-{d.gasoline_orders[j]["order_name"]}',
                                # 合成订单的名字
                                'station_name1': d.gasoline_orders[m]['station_name'],
                                'station_name2': d.gasoline_orders[j]['station_name'],
                                'original_order1': d.gasoline_orders[m]['order_name'],
                                'original_order2': d.gasoline_orders[j]['order_name'],
                                'depot': d.gasoline_orders[m]['depot'],
                                'time_to_empty': min(d.gasoline_orders[m]['time_to_empty'],
                                                     d.gasoline_orders[j]['time_to_empty']),
                                'vehicle_type': 'Double_gasoline',
                                'oil_class_1': d.gasoline_orders[m]['oil_class'],
                                'oil_class_2': d.gasoline_orders[j]['oil_class'],
                                'path': [d.gasoline_orders[m]['depot'], sta[i][0], sta[i][1],
                                         d.gasoline_orders[m]['depot']],
                                'distance': distance.get_path_distance(distance.alldistance,
                                                                       [d.gasoline_orders[m]['depot'], sta[i][0], sta[i][1],
                                                                        d.gasoline_orders[m]['depot']], ),
                                'cost': best_cost,
                                # 订单等待
                                'time1': d.gasoline_orders[m]['time1'],
                                'time2': d.gasoline_orders[j]['time1'],
                                'class': 'gasoline',
                                # 订单等待

                                'order_type': 'combined'
                            }
                            # 删除总订单列表中的两个子订单

                            d.total_orders.remove(d.gasoline_orders[m])
                            # print('两个子订单之前均未合成')
                            # print('总订单删除：',d.gasoline_orders[m])
                            d.total_orders.remove(d.gasoline_orders[j])
                            # print('总订单删除：', d.gasoline_orders[j])
                            d.gasoline_orders[m]['is_combined'] = 1  # 是否被合成 1代表被合成
                            d.gasoline_orders[j]['is_combined'] = 1

                            if combined_order['order_name'] not in d.total_orders:
                                # 将合并后的订单添加到总订单列表以及合成列表中
                                d.total_orders.append(combined_order)
                                # print("加入到合成订单里")
                                d.combined_orders.append(combined_order)
                        elif (d.gasoline_orders[m]['is_combined'] == 1 and d.gasoline_orders[j]['is_combined'] == 0) or \
                                (d.gasoline_orders[m]['is_combined'] == 0 and d.gasoline_orders[j]['is_combined'] == 1):
                            # print('两个子订单其中一个是合成另一个之前未合成')
                            for old_combined_order in d.combined_orders:
                                if old_combined_order['original_order1'] == d.gasoline_orders[m]['order_name'] and \
                                        d.gasoline_orders[m]['is_combined'] == 1:
                                    for order in d.gasoline_orders:
                                        # print(d.gasoline_orders[m],'之前合成')
                                        if order['order_name'] == old_combined_order['original_order2']:
                                            order['is_combined'] = 0  # 将恢复的订单变为未合成状态
                                            d.total_orders.append(order)
                                            # print('去除另一个子订单并将其重新添加至总订单中：',order)
                                            break
                                    # 删除先前的合并订单
                                    for l in d.total_orders:
                                        if l['order_name'] == old_combined_order['order_name']:
                                            d.total_orders.remove(l)
                                            break
                                    for k in d.combined_orders:
                                        if k['order_name'] == old_combined_order['order_name']:
                                            d.combined_orders.remove(k)
                                            break
                                    # 删除总订单列表中的一个子订单
                                    # print('将', d.gasoline_orders[j], '从总订单中删除')
                                    d.total_orders.remove(d.gasoline_orders[j])

                                    d.gasoline_orders[j]['is_combined'] = 1
                                elif old_combined_order['original_order2'] == d.gasoline_orders[m]['order_name'] and \
                                        d.gasoline_orders[m]['is_combined'] == 1:
                                    for order in d.gasoline_orders:
                                        # print(d.gasoline_orders[m], '之前合成')
                                        if order['order_name'] == old_combined_order['original_order1']:
                                            order['is_combined'] = 0  # 将恢复的订单变为未合成状态
                                            d.total_orders.append(order)
                                            # print('去除另一个子订单并将其重新添加至总订单中：',order)
                                            break
                                    # 删除先前的合并订单
                                    for l in d.total_orders:
                                        if l['order_name'] == old_combined_order['order_name']:
                                            d.total_orders.remove(l)
                                    for k in d.combined_orders:
                                        if k['order_name'] == old_combined_order['order_name']:
                                            d.combined_orders.remove(k)
                                    # 删除总订单列表中的一个子订单
                                    d.total_orders.remove(d.gasoline_orders[j])
                                    # print('将',d.gasoline_orders[j],'从总订单中删除')
                                    d.gasoline_orders[j]['is_combined'] = 1
                                # 若是子订单2被重新合成，找出先前合成订单的子订单1，并恢复到总订单里
                                elif old_combined_order['original_order1'] == d.gasoline_orders[j]['order_name'] and \
                                        d.gasoline_orders[j]['is_combined'] == 1:
                                    for order in d.gasoline_orders:
                                        # print(d.gasoline_orders[j], '之前合成')
                                        if order['order_name'] == old_combined_order['original_order2']:
                                            order['is_combined'] = 0  # 将恢复的订单变为未合成状态
                                            d.total_orders.append(order)
                                            # print('去除另一个子订单并将其重新添加至总订单中：',order)
                                            break
                                    # 删除先前的合并订单
                                    for l in d.total_orders:
                                        if l['order_name'] == old_combined_order['order_name']:
                                            d.total_orders.remove(l)
                                    for k in d.combined_orders:
                                        if k['order_name'] == old_combined_order['order_name']:
                                            d.combined_orders.remove(k)
                                    # 删除总订单列表中的一个子订单
                                    d.total_orders.remove(d.gasoline_orders[m])
                                    # print('将',d.gasoline_orders[m],'从总订单中删除')
                                    d.gasoline_orders[m]['is_combined'] = 1
                                elif old_combined_order['original_order2'] == d.gasoline_orders[j]['order_name'] and \
                                        d.gasoline_orders[j]['is_combined'] == 1:
                                    # print(d.gasoline_orders[j], '之前合成')
                                    for order in d.gasoline_orders:
                                        if order['order_name'] == old_combined_order['original_order1']:
                                            order['is_combined'] = 0  # 将恢复的订单变为未合成状态
                                            d.total_orders.append(order)
                                            # print('去除另一个子订单并将其重新添加至总订单中：',order)
                                            break
                                    # 删除先前的合并订单
                                    for l in d.total_orders:
                                        if l['order_name'] == old_combined_order['order_name']:
                                            d.total_orders.remove(l)
                                    for k in d.combined_orders:
                                        if k['order_name'] == old_combined_order['order_name']:
                                            d.combined_orders.remove(k)
                                    # 删除总订单列表中的一个子订单
                                    # if d.gasoline_orders[m] not in d.total_orders:
                                    #     print(1)
                                    d.total_orders.remove(d.gasoline_orders[m])
                                    # print('将',d.gasoline_orders[m],'从总订单中删除')
                                    d.gasoline_orders[m]['is_combined'] = 1
                            # 两个订单原先均为合成状态情况
                            combined_order = {
                                'order_name': f'{d.gasoline_orders[m]["order_name"]}-{d.gasoline_orders[j]["order_name"]}',
                                # 合成订单的名字
                                'station_name1': d.gasoline_orders[m]['station_name'],
                                'station_name2': d.gasoline_orders[j]['station_name'],
                                'original_order1': d.gasoline_orders[m]['order_name'],
                                'original_order2': d.gasoline_orders[j]['order_name'],
                                'depot': d.gasoline_orders[m]['depot'],
                                'time_to_empty': min(d.gasoline_orders[m]['time_to_empty'],
                                                     d.gasoline_orders[j]['time_to_empty']),
                                'vehicle_type': 'Double_gasoline',
                                'oil_class_1': d.gasoline_orders[m]['oil_class'],
                                'oil_class_2': d.gasoline_orders[j]['oil_class'],
                                'path': [d.gasoline_orders[m]['depot'], sta[i][0], sta[i][1],
                                         d.gasoline_orders[m]['depot']],
                                # 订单等待
                                'class': 'gasoline',
                                'time1': d.gasoline_orders[m]['time1'],
                                'time2': d.gasoline_orders[j]['time1'],
                                # 订单等待
                                'distance': distance.get_path_distance(distance.alldistance,
                                                                       [d.gasoline_orders[m]['depot'], sta[i][0],
                                                                        sta[i][1],
                                                                        d.gasoline_orders[m]['depot']], ),
                                'cost': best_cost,

                                'order_type': 'combined'
                            }
                            if combined_order['order_name'] not in d.total_orders:
                                # 将合并后的订单添加到总订单列表中
                                d.total_orders.append(combined_order)
                                d.combined_orders.append(combined_order)
                        # elif d.gasoline_orders[m]['is_combined'] == 1 and d.gasoline_orders[j]['is_combined'] == 1:
                        #     print('两个子订之前均已合成')
                        #     for old_combined_order in d.combined_orders:
                        #         if (old_combined_order['original_order1'] == d.gasoline_orders[m]['order_name'] and \
                        #                 old_combined_order['original_order2'] == d.gasoline_orders[j]['order_name']) or \
                        #                 (old_combined_order['original_order2'] == d.gasoline_orders[m]['order_name'] and \
                        #                 old_combined_order['original_order1'] == d.gasoline_orders[j]['order_name']):
                        #             # 删除先前的合并订单
                        #             print('两个子订之前已为同一个合成订单')
                        #             # d.total_orders.remove(old_combined_order)
                        #             # d.combined_orders.remove(old_combined_order)
                        #             break
                        #         else:
                        #             print('两个子订单之前不是同一个合成订单')
                        #             if old_combined_order['original_order1'] == d.gasoline_orders[m][
                        #                 'order_name']:
                        #                 print(old_combined_order, '之前的合成订单分解')
                        #                 for order in d.gasoline_orders:
                        #                     if order['order_name'] == old_combined_order['original_order2']:
                        #                         order['is_combined'] = 0  # 将恢复的订单变为未合成状态
                        #                         d.total_orders.append(order)
                        #                         print('去除另一个子订单并将其重新添加至总订单中：',order)
                        #                         break
                        #                 # 删除先前的合并订单
                        #                 for l in d.total_orders:
                        #                     if l['order_name'] == old_combined_order['order_name']:
                        #                         d.total_orders.remove(l)
                        #                 for k in d.combined_orders:
                        #                     if k['order_name'] == old_combined_order['order_name']:
                        #                         d.combined_orders.remove(k)
                        #                 # 删除总订单列表中的一个子订单
                        #
                        #             if old_combined_order['original_order2'] == d.gasoline_orders[m][
                        #                 'order_name']:
                        #                 print(old_combined_order, '之前的合成订单分解')
                        #                 for order in d.gasoline_orders:
                        #                     if order['order_name'] == old_combined_order['original_order1']:
                        #                         order['is_combined'] = 0  # 将恢复的订单变为未合成状态
                        #                         d.total_orders.append(order)
                        #                         print('去除另一个子订单并将其重新添加至总订单中：',order)
                        #                         break
                        #                 # 删除先前的合并订单
                        #                 for l in d.total_orders:
                        #                     if l['order_name'] == old_combined_order['order_name']:
                        #                         d.total_orders.remove(l)
                        #                 for k in d.combined_orders:
                        #                     if k['order_name'] == old_combined_order['order_name']:
                        #                         d.combined_orders.remove(k)
                        #                 # 删除总订单列表中的一个子订单
                        #
                        #             # 若是子订单2被重新合成，找出先前合成订单的子订单1，并恢复到总订单里
                        #             if old_combined_order['original_order1'] == d.gasoline_orders[j][
                        #                 'order_name']:
                        #                 print(old_combined_order, '之前的合成订单分解')
                        #                 for order in d.gasoline_orders:
                        #                     if order['order_name'] == old_combined_order['original_order2']:
                        #                         order['is_combined'] = 0  # 将恢复的订单变为未合成状态
                        #                         d.total_orders.append(order)
                        #                         print('去除另一个子订单并将其重新添加至总订单中：',order)
                        #                         break
                        #                 # 删除先前的合并订单
                        #                 for l in d.total_orders:
                        #                     if l['order_name'] == old_combined_order['order_name']:
                        #                         d.total_orders.remove(l)
                        #                 for k in d.combined_orders:
                        #                     if k['order_name'] == old_combined_order['order_name']:
                        #                         d.combined_orders.remove(k)
                        #                 # 删除总订单列表中的一个子订单
                        #
                        #             if old_combined_order['original_order2'] == d.gasoline_orders[j][
                        #                 'order_name']:
                        #                 print(old_combined_order, '之前的合成订单分解')
                        #                 for order in d.gasoline_orders:
                        #                     if order['order_name'] == old_combined_order['original_order1']:
                        #                         order['is_combined'] = 0  # 将恢复的订单变为未合成状态
                        #                         d.total_orders.append(order)
                        #                         print('去除另一个子订单并将其重新添加至总订单中：',order)
                        #                         break
                        #                 # 删除先前的合并订单
                        #                 for l in d.total_orders:
                        #                     if l['order_name'] == old_combined_order['order_name']:
                        #                         d.total_orders.remove(l)
                        #                 for k in d.combined_orders:
                        #                     if k['order_name'] == old_combined_order['order_name']:
                        #                         d.combined_orders.remove(k)
                        #                 # 删除总订单列表中的一个子订单
                        #
                        #
                        #
                        #
                        #         # 若是子订单1被重新合成，找出先前合成订单的子订单2，并恢复到总订单里
                        #
                        #
                        #     # 合成新的合成订单
                        #     combined_order = {
                        #         'order_name': f'{d.gasoline_orders[m]["order_name"]}-{d.gasoline_orders[j]["order_name"]}',
                        #         # 合成订单的名字
                        #         'station_name1': d.gasoline_orders[m]['station_name'],
                        #         'station_name2': d.gasoline_orders[j]['station_name'],
                        #         'original_order1': d.gasoline_orders[m]['order_name'],
                        #         'original_order2': d.gasoline_orders[j]['order_name'],
                        #         'depot': d.gasoline_orders[m]['depot'],
                        #         'time_to_empty': min(d.gasoline_orders[m]['time_to_empty'],
                        #                              d.gasoline_orders[j]['time_to_empty']),
                        #         'vehicle_type': 'Double_gasoline',
                        #         'oil_class_1': d.gasoline_orders[m]['oil_class'],
                        #         'oil_class_2': d.gasoline_orders[j]['oil_class'],
                        #         'path': [d.gasoline_orders[m]['depot'], sta[i][0], sta[i][1],
                        #                  d.gasoline_orders[m]['depot']],
                        #         # 订单等待
                        #         'class': 'gasoline',
                        #         'time1': d.gasoline_orders[m]['time1'],
                        #         'time2': d.gasoline_orders[j]['time1'],
                        #         # 订单等待
                        #         'distance': distance.get_path_distance(distance.alldistance,
                        #                                                [d.gasoline_orders[m]['depot'], sta[i][0], sta[i][1],
                        #                                                 d.gasoline_orders[m]['depot']], ),
                        #         'cost': best_cost,
                        #
                        #         'order_type': 'combined'
                        #     }
                        #     if combined_order['order_name'] not in d.total_orders:
                        #         # 将合并后的订单添加到总订单列表中
                        #         d.total_orders.append(combined_order)
                        #         d.combined_orders.append(combined_order)

                        # print('---------')
                        # print(d.gasoline_orders[m])
                        # print(d.gasoline_orders[j])
                        # print('gasoline_orders:', d.gasoline_orders)
                        # print('total_orders:', d.total_orders)
                        # print('---------')

            if len(d.diesel_orders) != 0 and len(d.diesel_orders) != 1:
                best_route,best_derv_cost = simulate.tabu_search(d.id, d.diesel_orders, 100, 10)
                sta=best_route[1]
                for i in range(len(sta)):
                    if len(sta[i]) == 1:
                        for h in range(len(d.diesel_orders)):
                            if d.diesel_orders[h]['station_name'] == sta[i][0]:
                                if d.diesel_orders[h]['station_name'] == sta[i][0]:
                                    if d.diesel_orders[h]['is_combined'] == 1:
                                        for old_combined_order in d.combined_orders:
                                            if old_combined_order['original_order1'] == d.diesel_orders[h][
                                                'order_name']:
                                                d.diesel_orders[h]['is_combined'] = 0
                                                d.total_orders.append(d.diesel_orders[h])
                                                for order in d.diesel_orders:
                                                    # print(d.diesel_orders[h], '之前合成')
                                                    if order['order_name'] == old_combined_order['original_order2']:
                                                        order['is_combined'] = 0  # 将恢复的订单变为未合成状态
                                                        d.total_orders.append(order)
                                                        # print('去除另一个子订单并将其重新添加至总订单中：', order)
                                                        break
                                                # 删除先前的合并订单
                                                for l in d.total_orders:
                                                    if l['order_name'] == old_combined_order['order_name']:
                                                        d.total_orders.remove(l)
                                                        break
                                                for k in d.combined_orders:
                                                    if k['order_name'] == old_combined_order['order_name']:
                                                        d.combined_orders.remove(k)
                                                        break
                                                # 删除总订单列表中的一个子订单

                                            elif old_combined_order['original_order2'] == d.diesel_orders[h][
                                                'order_name']:
                                                d.diesel_orders[h]['is_combined'] = 0
                                                d.total_orders.append(d.diesel_orders[h])
                                                for order in d.diesel_orders:
                                                    # print(d.diesel_orders[h], '之前合成')
                                                    if order['order_name'] == old_combined_order['original_order1']:
                                                        order['is_combined'] = 0  # 将恢复的订单变为未合成状态
                                                        d.total_orders.append(order)
                                                        # print('去除另一个子订单并将其重新添加至总订单中：', order)
                                                        break
                                                # 删除先前的合并订单
                                                for l in d.total_orders:
                                                    if l['order_name'] == old_combined_order['order_name']:
                                                        d.total_orders.remove(l)
                                                for k in d.combined_orders:
                                                    if k['order_name'] == old_combined_order['order_name']:
                                                        d.combined_orders.remove(k)
                                                # 删除总订单列表中的一个子订单

                                            # 若是子订单2被重新合成，找出先前合成订单的子订单1，并恢复到总订单里
                                            elif old_combined_order['original_order1'] == d.diesel_orders[h][
                                                'order_name']:
                                                d.diesel_orders[h]['is_combined'] = 0
                                                d.total_orders.append(d.diesel_orders[h])
                                                for order in d.diesel_orders:
                                                    # print(d.diesel_orders[h], '之前合成')
                                                    if order['order_name'] == old_combined_order['original_order2']:
                                                        order['is_combined'] = 0  # 将恢复的订单变为未合成状态
                                                        d.total_orders.append(order)
                                                        # print('去除另一个子订单并将其重新添加至总订单中：', order)
                                                        break
                                                # 删除先前的合并订单
                                                for l in d.total_orders:
                                                    if l['order_name'] == old_combined_order['order_name']:
                                                        d.total_orders.remove(l)
                                                for k in d.combined_orders:
                                                    if k['order_name'] == old_combined_order['order_name']:
                                                        d.combined_orders.remove(k)
                                                # 删除总订单列表中的一个子订单

                                            elif old_combined_order['original_order2'] == d.diesel_orders[h][
                                                'order_name']:
                                                d.diesel_orders[h]['is_combined'] = 0
                                                d.total_orders.append(d.diesel_orders[h])
                                                # print(d.diesel_orders[h], '之前合成')
                                                for order in d.diesel_orders:
                                                    if order['order_name'] == old_combined_order['original_order1']:
                                                        order['is_combined'] = 0  # 将恢复的订单变为未合成状态
                                                        d.total_orders.append(order)
                                                        # print('去除另一个子订单并将其重新添加至总订单中：', order)
                                                        break
                                                # 删除先前的合并订单
                                                for l in d.total_orders:
                                                    if l['order_name'] == old_combined_order['order_name']:
                                                        d.total_orders.remove(l)
                                                for k in d.combined_orders:
                                                    if k['order_name'] == old_combined_order['order_name']:
                                                        d.combined_orders.remove(k)
                                                # 删除总订单列表中的一个子订单
                                                # if d.gasoline_orders[m] not in d.total_orders:
                                                #     print(1)
                                d.diesel_orders[h]['cost']=best_derv_cost
                    if len(sta[i])==2:
                        m=0
                        j =0
                        for h in range(len(d.diesel_orders)):
                            if d.diesel_orders[h]['station_name']==sta[i][0]:
                                m =h
                            if d.diesel_orders[h]['station_name'] == sta[i][1]:
                                j= h

                        if d.diesel_orders[m]['is_combined'] == 0 and d.diesel_orders[j]['is_combined'] == 0:
                            combined_order = {
                                'order_name': f'{d.diesel_orders[m]["order_name"]}-{d.diesel_orders[j]["order_name"]}',  # 合成订单的名字
                                'station_name1': d.diesel_orders[m]['station_name'],
                                'station_name2': d.diesel_orders[j]['station_name'],
                                'original_order1': d.diesel_orders[m]['order_name'],
                                'original_order2': d.diesel_orders[j]['order_name'],
                                'depot': d.diesel_orders[m]['depot'],
                                'time_to_empty': min(d.diesel_orders[m]['time_to_empty'], d.diesel_orders[j]['time_to_empty']),
                                'vehicle_type': 'Double_diesel',
                                'oil_class_1': d.diesel_orders[m]['oil_class'],
                                'oil_class_2': d.diesel_orders[j]['oil_class'],
                                'path': [d.diesel_orders[m]['depot'],sta[i][0],sta[i][1],d.diesel_orders[m]['depot']],
                                'distance': distance.get_path_distance(distance.alldistance,[d.diesel_orders[m]['depot'],sta[i][0],sta[i][1],d.diesel_orders[m]['depot']],),
                                'cost': best_derv_cost,
                                # 订单等待
                                'time1': d.diesel_orders[m]['time1'],
                                'time2': d.diesel_orders[j]['time1'],
                                'class': 'derv',
                                # 订单等待

                                'order_type': 'combined'
                            }
                            # 删除总订单列表中的两个子订单

                            d.total_orders.remove(d.diesel_orders[m])
                            d.total_orders.remove(d.diesel_orders[j])
                            d.diesel_orders[m]['is_combined'] = 1  # 是否被合成 1代表被合成
                            d.diesel_orders[j]['is_combined'] = 1

                            if combined_order['order_name'] not in d.total_orders:
                                # 将合并后的订单添加到总订单列表以及合成列表中
                                d.total_orders.append(combined_order)
                                # print("加入到合成订单里")
                                d.combined_orders.append(combined_order)
                        elif (d.diesel_orders[m]['is_combined'] == 1 and d.diesel_orders[j][
                                    'is_combined'] == 0) or (d.diesel_orders[m]['is_combined'] == 0 and d.diesel_orders[j]['is_combined'] == 1):
                            for old_combined_order in d.combined_orders:
                                # 若是子订单1被重新合成，找出先前合成订单的子订单2，并恢复到总订单里
                                if old_combined_order['original_order1'] == d.diesel_orders[m]['order_name'] and d.diesel_orders[m][
                                    'is_combined'] == 1:
                                    for order in d.diesel_orders:
                                        if order['order_name'] == old_combined_order['original_order2']:
                                            order['is_combined'] = 0  # 将恢复的订单变为未合成状态
                                            d.total_orders.append(order)
                                    # 删除先前的合并订单
                                    d.total_orders.remove(old_combined_order)
                                    d.combined_orders.remove(old_combined_order)
                                    # 删除总订单列表中的一个子订单
                                    d.total_orders.remove(d.diesel_orders[j])
                                    d.diesel_orders[j]['is_combined'] = 1
                                elif old_combined_order['original_order2'] == d.diesel_orders[m]['order_name'] and d.diesel_orders[m][
                                    'is_combined'] == 1:
                                    for order in d.diesel_orders:
                                        if order['order_name'] == old_combined_order['original_order1']:
                                            order['is_combined'] = 0  # 将恢复的订单变为未合成状态
                                            d.total_orders.append(order)
                                    # 删除先前的合并订单
                                    d.total_orders.remove(old_combined_order)
                                    d.combined_orders.remove(old_combined_order)
                                    # 删除总订单列表中的一个子订单
                                    d.total_orders.remove(d.diesel_orders[j])
                                    d.diesel_orders[j]['is_combined'] = 1
                                # 若是子订单2被重新合成，找出先前合成订单的子订单1，并恢复到总订单里
                                elif old_combined_order['original_order1'] == d.diesel_orders[j]['order_name'] and d.diesel_orders[j][
                                    'is_combined'] == 1:
                                    for order in d.diesel_orders:
                                        if order['order_name'] == old_combined_order['original_order2']:
                                            order['is_combined'] = 0  # 将恢复的订单变为未合成状态
                                            d.total_orders.append(order)
                                    # 删除先前的合并订单
                                    d.total_orders.remove(old_combined_order)
                                    d.combined_orders.remove(old_combined_order)
                                    # 删除总订单列表中的一个子订单
                                    if d.diesel_orders[m] not in d.total_orders:
                                        print(1)
                                    d.total_orders.remove(d.diesel_orders[m])
                                    d.diesel_orders[m]['is_combined'] = 1
                                elif old_combined_order['original_order2'] == d.diesel_orders[j]['order_name'] and d.diesel_orders[j][
                                    'is_combined'] == 1:
                                    for order in d.diesel_orders:
                                        if order['order_name'] == old_combined_order['original_order1']:
                                            order['is_combined'] = 0  # 将恢复的订单变为未合成状态
                                            d.total_orders.append(order)
                                    # 删除先前的合并订单
                                    d.total_orders.remove(old_combined_order)
                                    d.combined_orders.remove(old_combined_order)
                                    # 删除总订单列表中的一个子订单
                                    d.total_orders.remove(d.diesel_orders[m])
                                    d.diesel_orders[m]['is_combined'] = 1
                            # 合成新的合成订单
                            combined_order = {
                                'order_name': f'{d.diesel_orders[m]["order_name"]}-{d.diesel_orders[j]["order_name"]}',  # 合成订单的名字
                                'station_name1': d.diesel_orders[m]['station_name'],
                                'station_name2': d.diesel_orders[j]['station_name'],
                                'original_order1': d.diesel_orders[m]['order_name'],
                                'original_order2': d.diesel_orders[j]['order_name'],
                                'depot': d.diesel_orders[m]['depot'],
                                'time_to_empty': min(d.diesel_orders[m]['time_to_empty'], d.diesel_orders[j]['time_to_empty']),
                                'vehicle_type': 'Double_diesel',
                                'oil_class_1': d.diesel_orders[m]['oil_class'],
                                'oil_class_2': d.diesel_orders[j]['oil_class'],
                                'path': [d.diesel_orders[m]['depot'],sta[i][0],sta[i][1],d.diesel_orders[m]['depot']],
                                # 订单等待
                                'class': 'derv',
                                'time1': d.diesel_orders[m]['time1'],
                                'time2': d.diesel_orders[j]['time1'],
                                # 订单等待
                                'distance': distance.get_path_distance(distance.alldistance, [d.diesel_orders[m]['depot'],sta[i][0],sta[i][1],d.diesel_orders[m]['depot']],),
                                'cost': best_derv_cost,

                                'order_type': 'combined'
                            }
                            if combined_order['order_name'] not in d.total_orders:
                                # 将合并后的订单添加到总订单列表中
                                d.total_orders.append(combined_order)
                                d.combined_orders.append(combined_order)

                        # print('---------')
                        # print(d.gasoline_orders[m])
                        # print(d.gasoline_orders[j])
                        # print('gasoline_orders:', d.gasoline_orders)
                        # print('total_orders:', d.total_orders)
                        # print('---------')
        #订单合成
        for d in self.alldepots:
            if len(d.total_orders) != 0:

                for o in d.total_orders[:]:
                    #订单等待
                    if o['order_type']=='combined':
                        if o['class']=='derv':
                            matching_order1 = next(order for order in d.diesel_orders if order['order_name'] == o['original_order1'])
                            matching_order2 = next(order for order in d.diesel_orders if order['order_name'] == o['original_order2'])
                            matching_order1['time1']+=self.step_minutes
                            matching_order2['time1'] += self.step_minutes
                        else:
                            # print(o)
                            # print(o)
                            # # print(o['class'])
                            # print('子订单1：',o['original_order1'])
                            # print('子订单2：', o['original_order2'])
                            matching_order1 = next(order for order in d.gasoline_orders if order['order_name'] == o['original_order1'])
                            matching_order2 = next(order for order in d.gasoline_orders if order['order_name'] == o['original_order2'])
                            matching_order1['time1'] += self.step_minutes
                            matching_order2['time1'] += self.step_minutes
                        o['time1']+=self.step_minutes
                        o['time2'] += self.step_minutes

                    else:
                        o['time1'] += self.step_minutes
                    #订单等待

                    result = dispatch_vehicle.dispatch_vehicle(o, depot_states, vehicle_states, d, station_states,self.tube)
                    # if result is None:
                    #     raise ValueError("dispatch_vehicle.dispatch_vehicle returned None for order: {}".format(o))

                    is_dispatch, vtype = result


                    # 改动
                    if is_dispatch:

                            if o['order_type']!='combined':
                                ostation=New_characters.find_station_by_id(o['station_name'])

                                # ostation.total_reward -= o['cost']/150
                                # if 'gasoline' in vtype:
                                if o['oil_class']=='92':

                                    # ostation.total_reward -= o['cost']
                                    if 'Single' in vtype:
                                        self.singe_dis += 1
                                        # ostation.total_reward -=2
                                        o['time_deliver'] =\
                                            self.service_time / self.step_minutes + self.refuel_time / self.step_minutes + math.ceil(
                                                (distance.get_path_distance(distance.alldistance,
                                                                            [o['path'][0],
                                                                             o['path'][1]]) / 5) / self.step_minutes)+1
                                        # station_states[o['station_name']]['Vehicle dispatch single_gasoline']+=1

                                        # ostation.single_92_reward -= 50
                                    elif 'Double' in vtype:
                                        self.double_dis += 1
                                        # ostation.total_reward -=4
                                        o['time_deliver'] = \
                                            2*self.service_time / self.step_minutes + 2*self.refuel_time / self.step_minutes + math.ceil(
                                                (distance.get_path_distance(distance.alldistance,
                                                                            [o['path'][0],
                                                                             o['path'][1]]) / 5) / self.step_minutes)+1
                                        # station_states[o['station_name']]['Vehicle dispatch double_gasoline'] += 1
                                        # ostation.single_92_reward -= 100
                                elif o['oil_class']=='95':
                                    # ostation.single_95_reward -= o['cost']
                                    if 'Single' in vtype:
                                        self.singe_dis += 1
                                        # ostation.total_reward -=2
                                        o['time_deliver'] = \
                                            self.service_time / self.step_minutes + self.refuel_time / self.step_minutes + math.ceil(
                                                (distance.get_path_distance(distance.alldistance,
                                                                            [o['path'][0],
                                                                             o['path'][1]]) / 5) / self.step_minutes)+1
                                        # station_states[o['station_name']]['Vehicle dispatch single_gasoline'] += 1
                                        # ostation.single_95_reward -= 50
                                    elif 'Double' in vtype:
                                        self.double_dis += 1
                                        # ostation.total_reward -=4
                                        o['time_deliver'] = \
                                            2 * self.service_time / self.step_minutes + 2 * self.refuel_time / self.step_minutes + math.ceil(
                                                (distance.get_path_distance(distance.alldistance,
                                                                            [o['path'][0],
                                                                             o['path'][1]]) / 5) / self.step_minutes)+1
                                        # station_states[o['station_name']]['Vehicle dispatch double_gasoline'] += 1
                                        # ostation.single_95_reward -= 100
                                elif o['oil_class']=='derv':
                                    # ostation.single_derv_reward -= o['cost']
                                    if 'Single' in vtype:
                                        self.singe_dis += 1
                                        # ostation.total_reward -=2
                                        o['time_deliver'] = \
                                            self.service_time / self.step_minutes + self.refuel_time / self.step_minutes + math.ceil(
                                                (distance.get_path_distance(distance.alldistance,
                                                                            [o['path'][0],
                                                                             o['path'][1]]) / 5) / self.step_minutes)+1
                                        # station_states[o['station_name']]['Vehicle dispatch single_diesel'] += 1
                                        # ostation.single_derv_reward -= 50
                                    elif 'Double' in vtype:
                                        self.double_dis += 1
                                        # ostation.total_reward -=4
                                        o['time_deliver'] = \
                                            2 * self.service_time / self.step_minutes + 2 * self.refuel_time / self.step_minutes + math.ceil(
                                                (distance.get_path_distance(distance.alldistance,
                                                                            [o['path'][0],
                                                                             o['path'][1]]) / 5) / self.step_minutes)+1
                                        # station_states[o['station_name']]['Vehicle dispatch double_diesel'] += 1
                                        # ostation.single_derv_reward -= 100
                                elif o['oil_class']=='95+92' or o['oil_class']=='92+95':
                                    # ostation.total_reward -=4
                                    self.double_dis += 1
                                    o['time_deliver'] = \
                                        2 * self.service_time / self.step_minutes + 2 * self.refuel_time / self.step_minutes + math.ceil(
                                            (distance.get_path_distance(distance.alldistance,
                                                                        [o['path'][0],
                                                                         o['path'][1]]) / 5) / self.step_minutes)+1
                                    # station_states[o['station_name']]['Vehicle dispatch double_gasoline'] += 1
                                    # ostation.single_92_reward -= o['cost']
                                    # ostation.single_95_reward -= o['cost']
#                                     ostation.single_92_reward -= 100
#                                     ostation.single_95_reward -= 100

                            else:
                                self.double_dis += 1
                                ostation1 = New_characters.find_station_by_id(o['station_name1'])
                                ostation2 = New_characters.find_station_by_id(o['station_name2'])
                                o['time_deliver1'] =self.service_time / self.step_minutes + 2*self.refuel_time / self.step_minutes + math.ceil(
                                    (distance.get_path_distance(distance.alldistance,
                                                                [o['path'][0], o['path'][1]]) / 5) / self.step_minutes)+1
                                o['time_deliver2'] = 2*self.service_time / self.step_minutes + 2*self.refuel_time / self.step_minutes + math.ceil(
                                    (distance.get_path_distance(distance.alldistance,
                                                                [o['path'][0], o['path'][1],o['path'][2]]) / 5) / self.step_minutes)+1
                                # ostation1.total_reward -= 4
                                # ostation2.total_reward -= 4
                                if o['oil_class_1']==o['oil_class_2']:
                                    if o['oil_class_1']=='92':
                                        # station_states[o['station_name1']]['Vehicle dispatch double_gasoline'] += 1
                                        # station_states[o['station_name2']]['Vehicle dispatch double_gasoline'] += 1
                                        ostation1 = New_characters.find_station_by_id(o['station_name1'])
                                        ostation2 = New_characters.find_station_by_id(o['station_name2'])
#                                         ostation1.single_92_reward -= o['cost']
#                                         ostation2.single_92_reward -= o['cost']
#                                         ostation1.single_92_reward -= 100
#                                         ostation2.single_92_reward -= 100
                                    elif o['oil_class_1']=='95':
                                        # station_states[o['station_name1']]['Vehicle dispatch double_gasoline'] += 1
                                        # station_states[o['station_name2']]['Vehicle dispatch double_gasoline'] += 1
                                        ostation1 = New_characters.find_station_by_id(o['station_name1'])
                                        ostation2 = New_characters.find_station_by_id(o['station_name2'])
#                                         ostation1.single_95_reward -= o['cost']
#                                         ostation2.single_95_reward -= o['cost']
#                                         ostation1.single_95_reward -= 100
#                                         ostation2.single_95_reward -= 100
                                    elif o['oil_class_1']=='derv':
                                        # station_states[o['station_name1']]['Vehicle dispatch double_diesel'] += 1
                                        # station_states[o['station_name2']]['Vehicle dispatch double_diesel'] += 1
                                        ostation1 = New_characters.find_station_by_id(o['station_name1'])
                                        ostation2 = New_characters.find_station_by_id(o['station_name2'])
#                                         ostation1.single_derv_reward -= o['cost']
#                                         ostation2.single_derv_reward -= o['cost']
#                                         ostation1.single_derv_reward -= 100
#                                         ostation2.single_derv_reward -= 100
                                elif o['oil_class_1']!=o['oil_class_2']:
                                    # station_states[o['station_name1']]['Vehicle dispatch double_gasoline'] += 1
                                    # station_states[o['station_name2']]['Vehicle dispatch double_gasoline'] += 1
                                    if o['oil_class_1'] == '92':
                                        ostation1 = New_characters.find_station_by_id(o['station_name1'])
#                                         ostation1.single_92_reward -= o['cost']
#                                         ostation1.single_92_reward -= 100
                                    elif o['oil_class_1'] == '95':
                                        ostation1 = New_characters.find_station_by_id(o['station_name1'])
#                                         ostation1.single_95_reward -= o['cost']
#                                         ostation1.single_95_reward -= 100

                                    if o['oil_class_2'] == '92':
                                        ostation2 = New_characters.find_station_by_id(o['station_name2'])
#                                         ostation2.single_92_reward -= o['cost']
#                                         ostation2.single_92_reward -= 100
                                    elif o['oil_class_2'] == '95':
                                        ostation2 = New_characters.find_station_by_id(o['station_name2'])

                    if is_dispatch:
                        # 改动，移除派送订单

                        if o['order_type'] == 'combined':
                            if 'gasoline' in o['vehicle_type']:
                                # print('派出订单：', o)
                                for order1 in d.gasoline_orders:
                                    if o['original_order1'] == order1['order_name']:
                                        d.gasoline_orders.remove(order1)
                                        # print('汽油订单列表去除合成订单：', order1)
                                for order2 in d.gasoline_orders:
                                    if o['original_order2'] == order2['order_name']:
                                        d.gasoline_orders.remove(order2)
                                        # print('汽油订单列表去除合成订单：', order2)
                            else:
                                for order1 in d.diesel_orders:
                                    if o['original_order1'] == order1['order_name']:
                                        d.diesel_orders.remove(order1)
                                for order2 in d.diesel_orders:
                                    if o['original_order2'] == order2['order_name']:
                                        d.diesel_orders.remove(order2)
                            d.combined_orders.remove(o)

                            d.total_orders.remove(o)
                            # print('总订单列表去除订单：', o)
                            # print('派出合成柴油订单：', o)
                        else:
                            if 'single' in o['order_name']:
                                if 'gasoline' in o['vehicle_type']:
                                    if o not in d.gasoline_orders:
                                        print(1)
                                    # print('汽油订单列表去除：', o)
                                    d.gasoline_orders.remove(o)
                                else:
                                    # print('当前油库:', d.id, '的所有单舱柴油订单：', d.diesel_orders)
                                    # print('派出非合成柴油订单：', o)
                                    d.diesel_orders.remove(o)

                            d.total_orders.remove(o)
                            # print('总订单列表去除：', o)

        # 新改

        # 无论出车or不出车均检查各车状态
        for vehicle_id, vehicle_state in vehicle_states.items():

            current_status = vehicle_state['status']
            vehicle_type = vehicle_state['vehicle_type']
            current_vehicle = New_characters.find_vehicle_by_id(vehicle_id)
            #订单等待
            if current_status != 0:
                if current_vehicle.order_type == 'combined':
                    current_vehicle.order['time1'] += self.step_minutes
                    current_vehicle.order['time2'] += self.step_minutes
                else:
                    current_vehicle.order['time1'] += self.step_minutes
            #订单等待

            depot_id=vehicle_id[:2]
            current_depot = New_characters.find_depot_by_id(depot_id)
            current_order = current_vehicle.order

            if current_status == 0:
                continue
            # 行驶中
            elif current_status == 1:
                # print(vehicle_id)
                # print(current_vehicle.current_refuel_cabin1)
                # print(current_vehicle.current_refuel_cabin2)
                # print(current_vehicle.oil_class_1)
                # print(current_vehicle.oil_class_2)
                # print(current_vehicle.order)
                # print(current_vehicle.order_type)

                # print(vehicle_state)
                vehicle_state['time_elapsed'] += self.step_minutes
                vehicle_state['total_time'] += self.step_minutes
                vehicle_state['distance_traveled'] -= current_vehicle.speed*self.step_minutes
                if vehicle_state['distance_traveled'] <= 0:
                    i = vehicle_state["target labeling"]
                    current_location = current_vehicle.target[i]

                    vehicle_state["distance_traveled"] = 0
                    if current_location.startswith("s"):  # 下一目的地为油站
                        vehicle_state["target labeling"] += 1

                        current_station = New_characters.find_station_by_id(current_location)
                        station_state = station_states[current_station.station_id]
                        if current_vehicle.order_type == 'not_combined':

                            if vehicle_type == 0:  # 如果为第一类型车，则只为单舱，且只可加92和95的汽油
                                # oil_type = current_vehicle.current_refuel_cabin1
                                if current_vehicle.current_refuel_cabin1 == '92':
                                    if station_states[current_station.station_id ]['service_status'][0] == 0:
                                        vehicle_state["status"] = 3
                                        station_states[current_station.station_id]['service_status'][0] = 1

                                    else:
                                        vehicle_state["status"] = 2
                                        station_states[current_station.station_id]['vehicle_num'][0] += 1
                                        # print(f"Adding vehicle {current_vehicle.truck_id} to queue_92")
                                        current_station.queue_92.put(current_vehicle)
                                        # station_state['vehicle_num'][0] += 1
                                        # print(f"Successfully added vehicle {current_vehicle.truck_id} to queue_92")

                                elif current_vehicle.current_refuel_cabin1 == '95':
                                    # print(vehicle_id)
                                    # print(current_vehicle.order)
                                    # print(current_vehicle.order_type)
                                    if station_states[current_station.station_id ]['service_status'][1] == 0:
                                        # print(vehicle_id,'+准备服务95',current_vehicle.order)
                                        # print('当前服务状态:',
                                        #       station_states[current_station.station_id]['service_status'][1])
                                        vehicle_state["status"] = 3
                                        station_states[current_station.station_id]['service_status'][1] += 1
                                        # print('当前服务状态:',
                                        #       station_states[current_station.station_id]['service_status'][1])

                                    else:
                                        # print(vehicle_id, '+等待服务95', current_vehicle.order)
                                        vehicle_state["status"] = 2
                                        station_states[current_station.station_id]['vehicle_num'][1] += 1
                                        # print(f"Adding vehicle {current_vehicle.truck_id} to queue_95")
                                        current_station.queue_95.put(current_vehicle)
                                        # station_state['vehicle_num'][1] += 1
                                        # print(f"Successfully added vehicle {current_vehicle.truck_id} to queue_95")


                            elif vehicle_type == 1:  # 如果为第二类型车，则只为单舱，且只可加柴油
                                    # print('非合成车行驶到')
                                    if station_states[current_station.station_id]['service_status'][2] == 0:
                                        vehicle_state["status"] = 3
                                        station_states[current_station.station_id]['service_status'][2] = 1

                                    else:
                                        vehicle_state["status"] = 2
                                        station_states[current_station.station_id]['vehicle_num'][2] += 1
                                        # print(f"Adding vehicle {current_vehicle.truck_id} to queue_derv")
                                        current_station.queue_derv.put(current_vehicle)
                                        # station_state['vehicle_num'][2] += 1
                                        # print(f"Successfully added vehicle {current_vehicle.truck_id} to queue_derv")


                            elif vehicle_type == 2:
                                oil_type1 = current_vehicle.current_refuel_cabin1

                                oil_type2 = current_vehicle.current_refuel_cabin2
                                # print(oil_type1,oil_type2)
                                if oil_type1==oil_type2:
                                        if oil_type1=='92':
                                            if station_states[current_station.station_id]['service_status'][0] == 0:
                                                vehicle_state["status"] = 3
                                                station_states[current_station.station_id]['service_status'][0] = 1

                                            else:
                                                vehicle_state["status"] = 2
                                                station_states[current_station.station_id]['vehicle_num'][0] += 1
                                                # print(f"Adding vehicle {current_vehicle.truck_id} to queue_92")
                                                current_station.queue_92.put(current_vehicle)
                                                # station_state['vehicle_num'][0] += 1
                                                # print(
                                                #     f"Successfully added vehicle {current_vehicle.truck_id} to queue_92")
                                        elif oil_type1 == '95':
                                            if station_states[current_station.station_id]['service_status'][1] == 0:
                                                # print(vehicle_id, '+先准备服务95', current_vehicle.order)
                                                vehicle_state["status"] = 3
                                                # print('当前服务状态:',
                                                #       station_states[current_station.station_id]['service_status'][1])
                                                station_states[current_station.station_id]['service_status'][1] += 1
                                                # print('当前服务状态:',station_states[current_station.station_id]['service_status'][1])
                                            else:
                                                # print(vehicle_id, '+等待服务95', current_vehicle.order)
                                                # print('当前服务状态:',station_states[current_station.station_id]['service_status'][1])
                                                vehicle_state["status"] = 2
                                                station_states[current_station.station_id]['vehicle_num'][1] += 1
                                                # print(f"Adding vehicle {current_vehicle.truck_id} to queue_95")
                                                current_station.queue_95.put(current_vehicle)
                                                # station_state['vehicle_num'][1] += 1
                                                # print(
                                                #     f"Successfully added vehicle {current_vehicle.truck_id} to queue_95")
                                else:
                                    if station_states[current_station.station_id]['service_status'][0] == 0:
                                        vehicle_state["status"] = 3
                                        station_states[current_station.station_id]['service_status'][0] = 1
                                        current_vehicle.current_refuel_cabin1='92'
                                        current_vehicle.current_refuel_cabin2 = '95'

                                    elif station_states[current_station.station_id]['service_status'][1] == 0:
                                                current_vehicle.current_refuel_cabin1 = '95'
                                                current_vehicle.current_refuel_cabin2 = '92'
                                                # print(vehicle_id, '+先准备服务95', current_vehicle.order)
                                                # print('当前服务状态:',
                                                #       station_states[current_station.station_id]['service_status'][1])
                                                vehicle_state["status"] = 3
                                                station_states[current_station.station_id]['service_status'][1] += 1
                                                # print('当前服务状态:',station_states[current_station.station_id]['service_status'][1])

                                    else:
                                        shortest_queue = current_station.queue_92 if current_station.queue_92.qsize() < current_station.queue_95.qsize() else current_station.queue_95
                                        if shortest_queue == current_station.queue_92:
                                            current_vehicle.current_refuel_cabin1 = '92'
                                            current_vehicle.current_refuel_cabin2 = '95'
                                            vehicle_state["status"] = 2
                                            station_state['vehicle_num'][0] += 1
                                            # print(f"Adding vehicle {current_vehicle.truck_id} to queue_92")
                                            current_station.queue_92.put(current_vehicle)
                                            # station_state['vehicle_num'][0] += 1
                                            # print(f"Successfully added vehicle {current_vehicle.truck_id} to queue_92")
                                        else:
                                            current_vehicle.current_refuel_cabin1 = '95'
                                            current_vehicle.current_refuel_cabin2 = '92'
                                            # print(vehicle_id, '+等待服务95', current_vehicle.order)
                                            # print('当前服务状态:',station_states[current_station.station_id]['service_status'][1])
                                            vehicle_state["status"] = 2
                                            station_state['vehicle_num'][1] += 1
                                            # print(f"Adding vehicle {current_vehicle.truck_id} to queue_95")
                                            current_station.queue_95.put(current_vehicle)
                                            # station_state['vehicle_num'][1] += 1
                                            # print(f"Successfully added vehicle {current_vehicle.truck_id} to queue_95")

                            elif vehicle_type == 3:
                                # print('非合成车行驶到')
                                if station_states[current_station.station_id]['service_status'][2] == 0:
                                    vehicle_state["status"] = 3
                                    station_states[current_station.station_id]['service_status'][2] = 1

                                else:
                                    vehicle_state["status"] = 2
                                    station_states[current_station.station_id]['vehicle_num'][2] += 1
                                    # print(f"Adding vehicle {current_vehicle.truck_id} to queue_derv")
                                    current_station.queue_derv.put(current_vehicle)
                                    # station_state['vehicle_num'][2] += 1
                                    # print(f"Successfully added vehicle {current_vehicle.truck_id} to queue_derv")
                        elif current_vehicle.order_type == 'combined':
                            if current_location==current_vehicle.target[1]:
                                # print('配送双仓订单第一站')
                                oil_type = current_vehicle.oil_class_1
                                if oil_type == '92':
                                    if station_states[current_station.station_id]['service_status'][0] == 0:
                                        vehicle_state["status"] = 3
                                        station_states[current_station.station_id]['service_status'][0] = 1

                                    else:
                                        vehicle_state["status"] = 2
                                        station_states[current_station.station_id]['vehicle_num'][0] += 1
                                        # print(f"Adding vehicle {current_vehicle.truck_id} to queue_92")
                                        current_station.queue_92.put(current_vehicle)
                                        # station_state['vehicle_num'][0] += 1
                                        # print(f"Successfully added vehicle {current_vehicle.truck_id} to queue_92")
                                elif oil_type == '95':
                                    if station_states[current_station.station_id]['service_status'][1] == 0:
                                        # print('当前服务状态:',
                                        #       station_states[current_station.station_id]['service_status'][1])
                                        # print(vehicle_id, '+准备服务95', current_vehicle.order)
                                        vehicle_state["status"] = 3
                                        station_states[current_station.station_id]['service_status'][1] += 1
                                        # print('当前服务状态:',
                                        #       station_states[current_station.station_id]['service_status'][1])

                                    else:
                                        # print(vehicle_id, '+等待服务95', current_vehicle.order)
                                        vehicle_state["status"] = 2
                                        station_states[current_station.station_id]['vehicle_num'][1] += 1
                                        # print(f"Adding vehicle {current_vehicle.truck_id} to queue_95")
                                        current_station.queue_95.put(current_vehicle)
                                        # station_state['vehicle_num'][1] += 1
                                        # print(f"Successfully added vehicle {current_vehicle.truck_id} to queue_95")
                                if oil_type == 'derv':
                                    # print('合成车行驶到')
                                    if station_states[current_station.station_id]['service_status'][2] == 0:
                                        vehicle_state["status"] = 3
                                        station_states[current_station.station_id]['service_status'][2] = 1

                                    else:
                                        vehicle_state["status"] = 2
                                        station_states[current_station.station_id]['vehicle_num'][2] += 1
                                        # print(f"Adding vehicle {current_vehicle.truck_id} to queue_derv")
                                        current_station.queue_derv.put(current_vehicle)
                                        # station_state['vehicle_num'][2] += 1
                                        # print(f"Successfully added vehicle {current_vehicle.truck_id} to queue_derv")
                            if current_location==current_vehicle.target[2]:
                                # print('配送双仓订单第二站')
                                oil_type = current_vehicle.oil_class_2
                                if oil_type == '92':
                                    if station_states[current_station.station_id]['service_status'][0] == 0:
                                        vehicle_state["status"] = 3
                                        station_states[current_station.station_id]['service_status'][0] = 1

                                    else:
                                        vehicle_state["status"] = 2
                                        station_states[current_station.station_id]['vehicle_num'][0] += 1
                                        # print(f"Adding vehicle {current_vehicle.truck_id} to queue_92")
                                        current_station.queue_92.put(current_vehicle)
                                        # station_state['vehicle_num'][0] += 1
                                        # print(f"Successfully added vehicle {current_vehicle.truck_id} to queue_92")
                                if oil_type == '95':
                                    if station_states[current_station.station_id]['service_status'][1] == 0:
                                        # print(vehicle_id, '+准备服务95', current_vehicle.order)
                                        # print('当前服务状态:',
                                        #       station_states[current_station.station_id]['service_status'][1])
                                        vehicle_state["status"] = 3
                                        station_states[current_station.station_id]['service_status'][1] += 1
                                        # print('当前服务状态:',
                                        #       station_states[current_station.station_id]['service_status'][1])

                                    else:
                                        # print(vehicle_id, '+等待服务95', current_vehicle.order)
                                        vehicle_state["status"] = 2
                                        station_states[current_station.station_id]['vehicle_num'][1] += 1
                                        # print(f"Adding vehicle {current_vehicle.truck_id} to queue_95")
                                        current_station.queue_95.put(current_vehicle)
                                        # station_state['vehicle_num'][1] += 1
                                        # print(f"Successfully added vehicle {current_vehicle.truck_id} to queue_95")
                                if oil_type == 'derv':
                                    if station_states[current_station.station_id]['service_status'][2] == 0:
                                        vehicle_state["status"] = 3
                                        station_states[current_station.station_id]['service_status'][2] = 1

                                    else:
                                        vehicle_state["status"] = 2
                                        station_states[current_station.station_id]['vehicle_num'][2] += 1
                                        # print(f"Adding vehicle {current_vehicle.truck_id} to queue_derv")
                                        current_station.queue_derv.put(current_vehicle)
                                        # station_state['vehicle_num'][2] += 1
                                        # print(f"Successfully added vehicle {current_vehicle.truck_id} to queue_derv")

                    if current_vehicle.target[i].startswith("d"):  # 下一目的地为油库


                        # if current_vehicle.order_type  == 'not_combined':
                        #     #改动
                        #     current_depot.total_orders.remove(current_order)
                        #     print(f'订单{current_order["order_name"]}完成')
                        # else:
                        #     current_depot.total_orders.remove(current_order)
                        #     print(f'订单{current_order["order_name"]}完成1')
                        #需要
                        # if current_vehicle.order_type == 'not_combined':
                        #     # print('非合成完成')
                        #     current_station = New_characters.find_station_by_id(current_vehicle.target[1])
                        #     current_station.single_reward -= vehicle_state["total_time"] * 10
#                         #     current_station.total_reward -= vehicle_state["total_time"] * 10
                        #
                        # else:
                        #     # print('合成完成')
                        #     current_station1 = New_characters.find_station_by_id(current_vehicle.target[1])
                        #     current_station2 = New_characters.find_station_by_id(current_vehicle.target[2])
                        #     current_station1.single_reward -= vehicle_state["total_time"] * 10
#                         #     current_station1.total_reward -= vehicle_state["total_time"] * 10
                        #     current_station2.single_reward -= vehicle_state["total_time"] * 10
#                         #     current_station2.total_reward -= vehicle_state["total_time"] * 10




                        vehicle_state["total_time"] = 0
                        vehicle_state["time_elapsed"] = 0
                        vehicle_state["target labeling"] = 0
                        vehicle_state["distance_traveled"] = 0
                        current_vehicle.current_refuel_cabin1=None
                        current_vehicle.current_refuel_cabin2 = None
                        current_vehicle.oil_class_1=None
                        current_vehicle.oil_class_2 = None

                        current_vehicle.order_type=None
                        vehicle_state["status"]=0
                        end_depot=New_characters.find_depot_by_id(current_vehicle.target[-1])
                        current_ds=depot_states[end_depot.id]
                        current_vehicle.target = []
                        if vehicle_state["vehicle_type"] == 0:
                            current_ds["vehicle_types_count"][0]+=1
                            end_depot.Single_gasoline.put(current_vehicle)
                        if vehicle_state["vehicle_type"] == 1:
                            end_depot.Single_diesel.put(current_vehicle)
                            current_ds["vehicle_types_count"][1] += 1
                        if vehicle_state["vehicle_type"] == 2:
                            end_depot.Double_gasoline.put(current_vehicle)
                            current_ds["vehicle_types_count"][2] += 1
                        if vehicle_state["vehicle_type"] == 3:
                            end_depot.Double_diesel.put(current_vehicle)
                            current_ds["vehicle_types_count"][3] += 1
                        # print('完成任务')

            # 等待中
            elif current_status == 2:
                #改动

                # print(vehicle_id)
                # print(current_vehicle.current_refuel_cabin1)
                # print(current_vehicle.current_refuel_cabin2)
                # print(current_vehicle.oil_class_1)
                # print(current_vehicle.oil_class_2)
                # print(current_vehicle.order)
                # print(current_vehicle.order_type)

                # print(vehicle_state)
                vehicle_state['total_time'] += self.step_minutes
                vehicle_state["wait_time"]+=self.step_minutes
                self.wait_time += 1
            # 服务中
            elif current_status == 3:
                # print(vehicle_id)
                # print(current_vehicle.current_refuel_cabin1)
                # print(current_vehicle.current_refuel_cabin2)
                # print(current_vehicle.oil_class_1)
                # print(current_vehicle.oil_class_2)
                # print(current_vehicle.order)
                # print(current_vehicle.order_type)
                #
                # print(vehicle_state)
                vehicle_state['total_time'] += self.step_minutes
                vehicle_state['service_time'] += self.step_minutes
                # 若服务完成
                if vehicle_state['service_time'] >= self.service_time:

                    i=vehicle_state["target labeling"]-1
                    current_location = current_vehicle.target[i]

                    current_station = New_characters.find_station_by_id(current_location)

                    station_current_state =station_states[current_station.station_id]
                    if current_vehicle.order_type == 'not_combined':
                        #改动
                        # 汽油 单舱
                        if vehicle_state['vehicle_type'] == 0:

                            # station_current_state['Vehicle dispatch single_gasoline'] -= 1
                            # if (current_vehicle.order['time1'] / self.step_minutes)<=current_vehicle.order['time_deliver']:
                            #     current_station.total_reward +=2
                            if current_vehicle.order['time1'] / self.step_minutes > current_vehicle.order['time_deliver']:
                                current_station.total_reward -=30
                                self.over_order +=1
                            # current_station.total_reward -= (current_vehicle.order['time1'] / self.step_minutes - current_vehicle.order['time_deliver'] )*0.05
                            #订单等待
                            # if current_vehicle.order['oil_class']=='92':
                            #     current_station.single_92_reward-=(current_vehicle.order['time1']/self.step_minutes-current_station.time_deliver)*10
                            # elif current_vehicle.order['oil_class']=='95':
                            #     current_station.single_95_reward -= (current_vehicle.order['time1']/self.step_minutes-current_station.time_deliver)*10
                            current_vehicle.order['time1']=0
                            #订单等待
                            vehicle_state['oil_tank_empty_cabin1'] = 0
                            vehicle_state['status'] = 1
                            vehicle_state['service_time'] = 0
                            # 若服务的为92号汽油
                            if current_vehicle.current_refuel_cabin1 == '92':
                                station_current_state['Vehicle dispatch single_92'] -= 1
                                station_current_state['Vehicle dispatch count_92'] -= 1
                                # 油站的92剩余油量增加一舱的油量
                                station_current_state['92_gas'] += current_vehicle.capacity
                                current_station.oil_92+= current_vehicle.capacity
#                                 # print('奖励')
                                #满仓
                                if station_current_state['92_gas'] >= current_station.capacity:
                                    station_current_state['92_gas'] = current_station.capacity

                                if station_current_state['vehicle_num'][0] != 0:
                                    try:
                                        # print("Attempting to get vehicle from queue_92")
                                        vehicle_wait = current_station.queue_92.get(timeout=2)  # 设置超时为5秒
                                        # print(f"Successfully got vehicle: {vehicle_wait}")
                                        # 处理车辆等待的逻辑
                                    except queue.Empty:
                                        print("92Queue is empty, could not retrieve vehicle within timeout period")

                                    vehicle_wait_state = vehicle_states[vehicle_wait.truck_id]
                                    vehicle_wait_state['status'] = 3



                                    #等待奖励
                                    # if vehicle_wait.order_type == 'not_combined':
                                    #     current_station = New_characters.find_station_by_id(vehicle_wait.target[1])
                                    #     current_station.single_reward -= vehicle_wait_state["wait_time"] * 15
#                                     #     current_station.total_reward -= vehicle_wait_state["wait_time"] * 15
                                    #
                                    #     # print('小惩罚')
                                    # else:
                                    #     current_station1 = New_characters.find_station_by_id(vehicle_wait.target[1])
                                    #     current_station2 = New_characters.find_station_by_id(vehicle_wait.target[2])
                                    #     current_station1.single_reward -= vehicle_wait_state["wait_time"] * 15
#                                     #     current_station1.total_reward -= vehicle_wait_state["wait_time"] * 15
                                    #     current_station2.single_reward -= vehicle_wait_state["wait_time"] * 15
#                                     #     current_station2.total_reward -= vehicle_wait_state["wait_time"] * 15
#                                         print('小惩罚')

                                    vehicle_wait_state["wait_time"] = 0
                                    station_current_state['vehicle_num'][0] -= 1
                                else:
                                    station_current_state['service_status'][0]=0
                            # 若服务的为95号汽油
                            else:
                                # 油站的95剩余油量增加一舱的油量
                                station_current_state['95_gas'] += current_vehicle.capacity
                                current_station.oil_95 += current_vehicle.capacity
                                station_current_state['Vehicle dispatch single_95'] -= 1
                                station_current_state['Vehicle dispatch count_95'] -= 1
# #                                 print('奖励')
                                #满仓
                                if station_current_state['95_gas'] >= current_station.capacity:
                                    station_current_state['95_gas'] = current_station.capacity
                                if station_current_state['vehicle_num'][1] != 0:

                                    try:
#                                         print("Attempting to get vehicle from queue_95")
                                        vehicle_wait = current_station.queue_95.get(timeout=2)  # 设置超时为5秒
#                                         print(f"Successfully got vehicle: {vehicle_wait}")
                                        # 处理车辆等待的逻辑
                                    except queue.Empty:
                                        print("95Queue is empty, could not retrieve vehicle within timeout period")

                                    vehicle_wait_state = vehicle_states[vehicle_wait.truck_id]
                                    # print(vehicle_id, '+服务完毕有等待车', current_vehicle.order)
                                    # print('当前服务状态:',
                                    #       station_states[current_station.station_id]['service_status'][1])
                                    vehicle_wait_state['status'] = 3
                                    # print(vehicle_wait.truck_id, '+等待车准备服务95', vehicle_wait.order)
                                    # print('当前服务状态:',
                                    #       station_states[current_station.station_id]['service_status'][1])
                                    #等待
#                                     if vehicle_wait.order_type == 'not_combined':
#                                         current_station = New_characters.find_station_by_id(vehicle_wait.target[1])
#                                         current_station.single_reward -= vehicle_wait_state["wait_time"] * 15
# #                                         current_station.total_reward -= vehicle_wait_state["wait_time"] * 15
#
# #                                         print('小惩罚')
#                                     else:
#                                         current_station1 = New_characters.find_station_by_id(vehicle_wait.target[1])
#                                         current_station2 = New_characters.find_station_by_id(vehicle_wait.target[2])
#                                         current_station1.single_reward -= vehicle_wait_state["wait_time"] * 15
# #                                         current_station1.total_reward -= vehicle_wait_state["wait_time"] * 15
#                                         current_station2.single_reward -= vehicle_wait_state["wait_time"] * 15
# #                                         current_station2.total_reward -= vehicle_wait_state["wait_time"] * 15
#                                         print('小惩罚')
                                    vehicle_wait_state["wait_time"] = 0
                                    station_current_state['vehicle_num'][1] -= 1
                                else:
                                    # print(vehicle_id, '+服务完毕且无等待车', current_vehicle.order)
                                    # print('当前服务状态:',
                                    #       station_states[current_station.station_id]['service_status'][1])
                                    station_current_state['service_status'][1]-=1
                                    # print('当前服务状态:',
                                    #       station_states[current_station.station_id]['service_status'][1])
                        # 柴油 单舱
                        elif vehicle_state['vehicle_type'] == 1:
                            # if (current_vehicle.order['time1'] / self.step_minutes)<=current_vehicle.order['time_deliver']:
                            #     current_station.total_reward +=2
                            if current_vehicle.order['time1'] / self.step_minutes >current_vehicle.order['time_deliver']:
                            # current_station.total_reward -= (current_vehicle.order['time1'] / self.step_minutes - current_vehicle.order['time_deliver'])*0.05
                                current_station.total_reward -= 30
                                self.over_order += 1
                            station_current_state['Vehicle dispatch single_diesel'] -= 1

                            #订单等待
                            station_current_state['Vehicle dispatch count_derv'] -= 1
                            # current_station.single_derv_reward -= (current_vehicle.order['time1']/self.step_minutes-current_station.time_deliver)*10
                            current_vehicle.order['time1'] = 0
                            #订单等待
                            vehicle_state['oil_tank_empty_cabin1'] = 0
                            vehicle_state['status'] = 1
                            vehicle_state['service_time'] = 0
                            station_current_state['diesel'] += current_vehicle.capacity
                            current_station.oil_derv += current_vehicle.capacity
# #                             print('奖励')
                            #满仓
                            if station_current_state['diesel']>=current_station.capacity:
                                station_current_state['diesel']=current_station.capacity
                            if station_current_state['vehicle_num'][2] != 0:
                                try:
#                                     print("Attempting to get vehicle from queue_derv")
                                    vehicle_wait = current_station.queue_derv.get(timeout=2)  # 设置超时为5秒
#                                     print(f"Successfully got vehicle: {vehicle_wait}")
                                    # 处理车辆等待的逻辑
                                except queue.Empty:
                                    print("dervQueue is empty, could not retrieve vehicle within timeout period")

                                vehicle_wait_state = vehicle_states[vehicle_wait.truck_id]
                                vehicle_wait_state['status'] = 3
                                #等待奖励
#                                 if vehicle_wait.order_type == 'not_combined':
#                                     current_station = New_characters.find_station_by_id(vehicle_wait.target[1])
#                                     current_station.single_reward -= vehicle_wait_state["wait_time"] * 15
# #                                     current_station.total_reward -= vehicle_wait_state["wait_time"] * 15
#
# #                                     print('小惩罚')
#                                 else:
#                                     current_station1 = New_characters.find_station_by_id(vehicle_wait.target[1])
#                                     current_station2 = New_characters.find_station_by_id(vehicle_wait.target[2])
#                                     current_station1.single_reward -= vehicle_wait_state["wait_time"] * 15
# #                                     current_station1.total_reward -= vehicle_wait_state["wait_time"] * 15
#                                     current_station2.single_reward -= vehicle_wait_state["wait_time"] * 15
# #                                     current_station2.total_reward -= vehicle_wait_state["wait_time"] * 15
#                                     print('小惩罚')
                                vehicle_wait_state["wait_time"] = 0
                                station_current_state['vehicle_num'][2] -= 1
                            else:
                                station_current_state['service_status'][2] = 0
                        # 汽油 双舱
                        elif vehicle_state['vehicle_type'] == 2:

                            # print('第一仓：', current_vehicle.current_refuel_cabin1)
                            # print('第二仓：', current_vehicle.current_refuel_cabin2)
                            # 两舱装的是否为同一种汽油
                            if current_vehicle.current_refuel_cabin2 == current_vehicle.current_refuel_cabin1:
                                # 两舱装的都为92
                                if current_vehicle.current_refuel_cabin1 == '92':
                                    # 判断第一舱是否为空(1为不空），若为不空，则代表为加油站装的是第一舱的油，赋值为0（空）
                                    if vehicle_state['oil_tank_empty_cabin1'] == 1:
                                        vehicle_state['oil_tank_empty_cabin1'] = 0

                                        vehicle_state['service_time'] = 0
                                        station_current_state['92_gas'] += current_vehicle.capacity
                                        current_station.oil_92 += current_vehicle.capacity
# #                                         print('奖励')
                                        #满仓
                                        if station_current_state['92_gas'] >= current_station.capacity:
                                            station_current_state['92_gas'] = current_station.capacity
                                    # 若第一舱为空，则代表为加油站装的是第二舱的油，赋值为0（空），同时target labeling加1
                                    else:
                                        #订单等待
                                        station_current_state['Vehicle dispatch double_92'] -= 1
                                        # station_current_state['Vehicle dispatch double_gasoline'] -= 1
                                        # if (current_vehicle.order['time1'] / self.step_minutes) <= \
                                        #         current_vehicle.order['time_deliver']:
                                        #     current_station.total_reward +=2
                                        if current_vehicle.order['time1'] / self.step_minutes > current_vehicle.order[
                                            'time_deliver']:
                                            current_station.total_reward -= 30
                                            self.over_order += 1
                                        # current_station.total_reward-= (current_vehicle.order['time1'] / self.step_minutes - current_vehicle.order['time_deliver'])*0.05
                                        # current_station.single_92_reward -= (current_vehicle.order['time1']/self.step_minutes-current_station.time_deliver)*10
                                        current_vehicle.order['time1'] = 0
                                        #订单等待
                                        vehicle_state['oil_tank_empty_cabin2'] = 0
                                        station_current_state['Vehicle dispatch count_92'] -= 1
                                        vehicle_state['status'] = 1
                                        vehicle_state['service_time'] = 0
                                        station_current_state['92_gas'] += current_vehicle.capacity
                                        current_station.oil_92 += current_vehicle.capacity
# #                                         print('奖励')
                                        #满仓
                                        if station_current_state['92_gas'] >= current_station.capacity:
                                            station_current_state['92_gas'] = current_station.capacity
                                        if station_current_state['vehicle_num'][0] != 0:
                                            try:
#                                                 print("Attempting to get vehicle from queue_92")
                                                vehicle_wait = current_station.queue_92.get(timeout=2)  # 设置超时为5秒
#                                                 print(f"Successfully got vehicle: {vehicle_wait}")
                                                # 处理车辆等待的逻辑
                                            except queue.Empty:
                                                print(
                                                    "92Queue is empty, could not retrieve vehicle within timeout period")

                                            vehicle_wait_state = vehicle_states[vehicle_wait.truck_id]
                                            vehicle_wait_state['status'] = 3
                                            #等待奖励
#                                             if vehicle_wait.order_type == 'not_combined':
#                                                 current_station = New_characters.find_station_by_id(
#                                                     vehicle_wait.target[1])
#                                                 current_station.single_reward -= vehicle_wait_state["wait_time"] * 15
# #                                                 current_station.total_reward -= vehicle_wait_state["wait_time"] * 15
# #                                                 print('小惩罚')
#                                             else:
#                                                 current_station1 = New_characters.find_station_by_id(
#                                                     vehicle_wait.target[1])
#                                                 current_station2 = New_characters.find_station_by_id(
#                                                     vehicle_wait.target[2])
#                                                 current_station1.single_reward -= vehicle_wait_state["wait_time"] * 15
# #                                                 current_station1.total_reward -= vehicle_wait_state["wait_time"] * 15
#                                                 current_station2.single_reward -= vehicle_wait_state["wait_time"] * 15
# #                                                 current_station2.total_reward -= vehicle_wait_state["wait_time"] * 15
#                                                 print('小惩罚')
                                            vehicle_wait_state["wait_time"] = 0
                                            station_current_state['vehicle_num'][0] -= 1
                                        else:
                                            station_current_state['service_status'][0] = 0
                                # 两舱装的都为95
                                else:
                                    if vehicle_state['oil_tank_empty_cabin1'] == 1:
                                        vehicle_state['oil_tank_empty_cabin1'] = 0
                                        station_current_state['95_gas'] += current_vehicle.capacity
                                        current_station.oil_95 += current_vehicle.capacity
# #                                         print('奖励')
                                        #满仓
                                        if station_current_state['95_gas'] >= current_station.capacity:
                                            station_current_state['95_gas'] = current_station.capacity
                                        vehicle_state['service_time'] = 0
                                    else:
                                        #订单等待
                                        station_current_state['Vehicle dispatch double_95'] -= 1
                                        # station_current_state['Vehicle dispatch double_gasoline'] -= 1
                                        # if (current_vehicle.order['time1'] / self.step_minutes) <= \
                                        #         current_vehicle.order['time_deliver']:
                                        #     current_station.total_reward += 2
                                        # current_station.total_reward-= (current_vehicle.order['time1'] / self.step_minutes - current_vehicle.order['time_deliver'])*0.5
                                        # current_station.single_95_reward -= (current_vehicle.order['time1']/self.step_minutes-current_station.time_deliver)*10
                                        current_vehicle.order['time1'] = 0
                                        #订单等待
                                        vehicle_state['oil_tank_empty_cabin2'] = 0
                                        station_current_state['Vehicle dispatch count_95'] -= 1
                                        vehicle_state['status'] = 1
                                        vehicle_state['service_time'] = 0

                                        if station_current_state['vehicle_num'][1] != 0:
                                            try:
#                                                 print("Attempting to get vehicle from queue_95")
                                                vehicle_wait = current_station.queue_95.get(timeout=2)  # 设置超时为5秒
#                                                 print(f"Successfully got vehicle: {vehicle_wait}")
                                                # 处理车辆等待的逻辑
                                            except queue.Empty:
                                                print(
                                                    "95Queue is empty, could not retrieve vehicle within timeout period")
                                            vehicle_wait_state = vehicle_states[vehicle_wait.truck_id]
                                            vehicle_wait_state['status'] = 3
                                            # print(vehicle_wait.truck_id, '+等待车准备服务95', vehicle_wait.order)
                                            # print('当前服务状态:',
                                            #       station_states[current_station.station_id]['service_status'][1])
                                            #等待奖励
                                            # if vehicle_wait.order_type == 'not_combined':
                                            #     current_station = New_characters.find_station_by_id(
                                            #         vehicle_wait.target[1])
                                            #     current_station.single_reward -= vehicle_wait_state["wait_time"] * 15
#                                             #     current_station.total_reward -= vehicle_wait_state["wait_time"] * 15
                                            # else:
                                            #     current_station1 = New_characters.find_station_by_id(
                                            #         vehicle_wait.target[1])
                                            #     current_station2 = New_characters.find_station_by_id(
                                            #         vehicle_wait.target[2])
                                            #     current_station1.single_reward -= vehicle_wait_state["wait_time"] * 15
#                                             #     current_station1.total_reward -= vehicle_wait_state["wait_time"] * 15
                                            #     current_station2.single_reward -= vehicle_wait_state["wait_time"] * 15
#                                             #     current_station2.total_reward -= vehicle_wait_state["wait_time"] * 15
#                                                 print('小惩罚')
                                            vehicle_wait_state["wait_time"] = 0
                                            station_current_state['vehicle_num'][1] -= 1
                                            # print(vehicle_id, '+服务完毕有等待车', current_vehicle.order)
                                        else:
                                            # print(vehicle_id, '+服务完毕且无等待车', current_vehicle.order)
                                            # print('当前服务状态:',
                                            #       station_states[current_station.station_id]['service_status'][1])
                                            station_current_state['service_status'][1]-=1
                                            # print('当前服务状态:',
                                            #       station_states[current_station.station_id]['service_status'][1])

                                            # station_current_state['service_status'][1] = 0
                            # 两舱装的不为同一种汽油
                            else:

                                if vehicle_state['oil_tank_empty_cabin1'] == 1:  # 判断第一舱是否为空
                                    vehicle_state['oil_tank_empty_cabin1'] = 0

                                    vehicle_state['service_time'] = 0
                                    # print('第一仓：',current_vehicle.current_refuel_cabin1)
                                    # print('第二仓：', current_vehicle.current_refuel_cabin2)
                                    # 该车刚服务完95号汽油，需要去服务92号汽油
                                    if current_vehicle.current_refuel_cabin2 == '92':
                                        # current_station.single_95_reward -= (current_vehicle.order['time1']/self.step_minutes-current_station.time_deliver)*10
                                        station_current_state['95_gas'] += current_vehicle.capacity
                                        current_station.oil_95 += current_vehicle.capacity
                                        station_current_state['Vehicle dispatch single_95'] -= 1
                                        station_current_state['Vehicle dispatch count_95'] -= 1
# #                                         print('奖励')
                                        #满仓
                                        if station_current_state['95_gas'] >= current_station.capacity:
                                            station_current_state['95_gas'] = current_station.capacity
                                        if station_current_state['vehicle_num'][1] != 0:
                                            try:
#                                                 print("Attempting to get vehicle from queue_95")
                                                vehicle_wait = current_station.queue_95.get(timeout=2)  # 设置超时为5秒
#                                                 print(f"Successfully got vehicle: {vehicle_wait}")
                                                # 处理车辆等待的逻辑
                                            except queue.Empty:
                                                print(
                                                    "95Queue is empty, could not retrieve vehicle within timeout period")
                                            vehicle_wait_state = vehicle_states[vehicle_wait.truck_id]
                                            vehicle_wait_state['status'] = 3
                                            # print(vehicle_wait.truck_id, '+等待车准备服务95', vehicle_wait.order)
                                            # print('当前服务状态:',
                                            #       station_states[current_station.station_id]['service_status'][1])
                                            #等待奖励
#                                             if vehicle_wait.order_type == 'not_combined':
#                                                 current_station = New_characters.find_station_by_id(vehicle_wait.target[1])
#                                                 current_station.single_reward -= vehicle_wait_state["wait_time"] * 15
# #                                                 current_station.total_reward -= vehicle_wait_state["wait_time"] * 15
# #                                                 print('小惩罚')
#                                             else:
#                                                 current_station1 = New_characters.find_station_by_id(
#                                                     vehicle_wait.target[1])
#                                                 current_station2 = New_characters.find_station_by_id(
#                                                     vehicle_wait.target[2])
#                                                 current_station1.single_reward -= vehicle_wait_state["wait_time"] * 15
# #                                                 current_station1.total_reward -= vehicle_wait_state["wait_time"] * 15
#                                                 current_station2.single_reward -= vehicle_wait_state["wait_time"] * 15
# #                                                 current_station2.total_reward -= vehicle_wait_state["wait_time"] * 15
#                                                 print('小惩罚')
                                            vehicle_wait_state["wait_time"] = 0
                                            station_current_state['vehicle_num'][1] -= 1
                                            # print(vehicle_id, '+服务完95去92有等待车', current_vehicle.order)
                                        else:
                                            # print('当前服务状态:',
                                            #       station_states[current_station.station_id]['service_status'][1])
                                            station_current_state['service_status'][1] -= 1
                                            # print(vehicle_id, '+服务完95去92无等待车', current_vehicle.order)
                                            # print('当前服务状态:',
                                            #       station_states[current_station.station_id]['service_status'][1])
                                            # station_current_state['service_status'][1] = 0
                                        if station_current_state['service_status'][0] == 0:
                                            vehicle_state["status"] = 3
                                            station_current_state['service_status'][0] = 1

                                        else:
                                            vehicle_state["status"] = 2
                                            current_station.queue_92.put(current_vehicle)
                                            station_current_state['vehicle_num'][0] += 1

                                    # 该车刚服务完92号汽油，需要去服务95号汽油
                                    else:
                                        # current_station.single_92_reward -= (current_vehicle.order['time1']/self.step_minutes-current_station.time_deliver)*10
                                        station_current_state['92_gas'] += current_vehicle.capacity
                                        current_station.oil_92 += current_vehicle.capacity
                                        station_current_state['Vehicle dispatch count_92'] -= 1
                                        station_current_state['Vehicle dispatch single_92'] -= 1
# #                                         print('奖励')
                                        #满仓
                                        if station_current_state['92_gas'] >= current_station.capacity:
                                            station_current_state['92_gas'] = current_station.capacity
                                        if station_current_state['vehicle_num'][0] != 0:
                                            try:
#                                                 print("Attempting to get vehicle from queue_92")
                                                vehicle_wait = current_station.queue_92.get(timeout=2)  # 设置超时为5秒
#                                                 print(f"Successfully got vehicle: {vehicle_wait}")
                                                # 处理车辆等待的逻辑
                                            except queue.Empty:
                                                print(
                                                    "92Queue is empty, could not retrieve vehicle within timeout period")
                                            vehicle_wait_state = vehicle_states[vehicle_wait.truck_id]
                                            vehicle_wait_state['status'] = 3
                                            # 等待奖励
                                            # if vehicle_wait.order_type == 'not_combined':
                                            #     current_station = New_characters.find_station_by_id(
                                            #         vehicle_wait.target[1])
                                            #     current_station.single_reward -= vehicle_wait_state["wait_time"] * 15
#                                             #     current_station.total_reward -= vehicle_wait_state["wait_time"] * 15
                                            # else:
                                            #     current_station1 = New_characters.find_station_by_id(
                                            #         vehicle_wait.target[1])
                                            #     current_station2 = New_characters.find_station_by_id(
                                            #         vehicle_wait.target[2])
                                            #     current_station1.single_reward -= vehicle_wait_state["wait_time"] * 15
#                                             #     current_station1.total_reward -= vehicle_wait_state["wait_time"] * 15
                                            #     current_station2.single_reward -= vehicle_wait_state["wait_time"] * 15
#                                             #     current_station2.total_reward -= vehicle_wait_state["wait_time"] * 15
                                            vehicle_wait_state["wait_time"] = 0
                                            station_current_state['vehicle_num'][0] -= 1
                                        else:
                                            station_current_state['service_status'][0] = 0
                                        if station_current_state['service_status'][1] == 0:
                                            # print(vehicle_id, '+服务完92去95', current_vehicle.order)
                                            # print('当前服务状态:',
                                            #       station_states[current_station.station_id]['service_status'][1])
                                            vehicle_state["status"] = 3
                                            station_current_state['service_status'][1] += 1
                                            # print('当前服务状态:',
                                            #       station_states[current_station.station_id]['service_status'][1])
                                        else:
                                            # print(vehicle_id, '+服务完92去95等待', current_vehicle.order)
                                            vehicle_state["status"] = 2
                                            current_station.queue_95.put(current_vehicle)
                                            station_current_state['vehicle_num'][1] += 1

                                else:
                                    #订单等待
                                    station_current_state['Vehicle dispatch double_gasoline'] -= 1
                                    # if (current_vehicle.order['time1'] / self.step_minutes) <= current_vehicle.order[
                                    #     'time_deliver']:
                                    #     current_station.total_reward += 2
                                    if current_vehicle.order['time1'] / self.step_minutes > current_vehicle.order[
                                        'time_deliver']:
                                        current_station.total_reward -= 30
                                        self.over_order += 1
                                    # current_station.total_reward -= (current_vehicle.order['time1'] / self.step_minutes - current_vehicle.order['time_deliver'])*0.05
                                    # if current_vehicle.current_refuel_cabin2 == '92':
                                    #     current_station.single_92_reward -= (current_vehicle.order['time1']/self.step_minutes-current_station.time_deliver)*10
                                    # elif current_vehicle.current_refuel_cabin2 == '95':
                                    #     current_station.single_95_reward -= (current_vehicle.order['time1']/self.step_minutes-current_station.time_deliver)*10
                                    current_vehicle.order['time1'] = 0
                                    #订单等待

                                    vehicle_state['oil_tank_empty_cabin2'] = 0

                                    vehicle_state['status'] = 1
                                    vehicle_state['service_time'] = 0

                                    if current_vehicle.current_refuel_cabin2 == '92':
                                        # station_current_state['Vehicle dispatch count_92'] -= 1
                                        station_current_state['Vehicle dispatch single_92'] -= 1
                                        station_current_state['92_gas'] += current_vehicle.capacity
                                        current_station.oil_92 += current_vehicle.capacity
# #                                         print('奖励')
                                        #满仓
                                        if station_current_state['92_gas'] >= current_station.capacity:
                                            station_current_state['92_gas'] = current_station.capacity
                                        if station_current_state['vehicle_num'][0] != 0:
                                            try:
#                                                 print("Attempting to get vehicle from queue_92")
                                                vehicle_wait = current_station.queue_92.get(timeout=2)  # 设置超时为5秒
#                                                 print(f"Successfully got vehicle: {vehicle_wait}")
                                                # 处理车辆等待的逻辑
                                            except queue.Empty:
                                                print(
                                                    "92Queue is empty, could not retrieve vehicle within timeout period")
                                            vehicle_wait_state = vehicle_states[vehicle_wait.truck_id]
                                            vehicle_wait_state['status'] =3
                                            #等待奖励
#                                             if vehicle_wait.order_type == 'not_combined':
#                                                 current_station = New_characters.find_station_by_id(
#                                                     vehicle_wait.target[1])
#                                                 current_station.single_reward -= vehicle_wait_state["wait_time"] * 15
# #                                                 current_station.total_reward -= vehicle_wait_state["wait_time"] * 15
# #                                                 print('小惩罚')
#                                             else:
#                                                 current_station1 = New_characters.find_station_by_id(
#                                                     vehicle_wait.target[1])
#                                                 current_station2 = New_characters.find_station_by_id(
#                                                     vehicle_wait.target[2])
#                                                 current_station1.single_reward -= vehicle_wait_state["wait_time"] * 15
# #                                                 current_station1.total_reward -= vehicle_wait_state["wait_time"] * 15
#                                                 current_station2.single_reward -= vehicle_wait_state["wait_time"] * 15
# #                                                 current_station2.total_reward -= vehicle_wait_state["wait_time"] * 15
#                                                 print('小惩罚')
                                            vehicle_wait_state["wait_time"] = 0
                                            station_current_state['vehicle_num'][0] -= 1
                                        else:
                                            station_current_state['service_status'][0] = 0

                                    else:
                                        station_current_state['Vehicle dispatch single_95'] -= 1
                                        # station_current_state['Vehicle dispatch count_95'] -= 1
                                        station_current_state['95_gas'] += current_vehicle.capacity
                                        current_station.oil_95 += current_vehicle.capacity
# #                                         print('奖励')
                                        #满仓
                                        if station_current_state['95_gas'] >= current_station.capacity:
                                            station_current_state['95_gas'] = current_station.capacity
                                        if station_current_state['vehicle_num'][1] != 0:
                                            try:
#                                                 print("Attempting to get vehicle from queue_95")
                                                vehicle_wait = current_station.queue_95.get(timeout=2)  # 设置超时为5秒
#                                                 print(f"Successfully got vehicle: {vehicle_wait}")
                                                # 处理车辆等待的逻辑
                                            except queue.Empty:
                                                print(
                                                    "95Queue is empty, could not retrieve vehicle within timeout period")
                                            vehicle_wait_state =vehicle_states[vehicle_wait.truck_id]
                                            vehicle_wait_state['status'] = 3
                                            # print(vehicle_wait.truck_id, '+等待车准备服务95', vehicle_wait.order)
                                            # print('当前服务状态:',
                                            #       station_states[current_station.station_id]['service_status'][1])
                                            #等待奖励
#                                             if vehicle_wait.order_type == 'not_combined':
#                                                 current_station = New_characters.find_station_by_id(
#                                                     vehicle_wait.target[1])
#                                                 current_station.single_reward -= vehicle_wait_state["wait_time"] * 15
# #                                                 current_station.total_reward -= vehicle_wait_state["wait_time"] * 15
# #                                                 print('小惩罚')
#                                             else:
#                                                 current_station1 = New_characters.find_station_by_id(
#                                                     vehicle_wait.target[1])
#                                                 current_station2 = New_characters.find_station_by_id(
#                                                     vehicle_wait.target[2])
#                                                 current_station1.single_reward -= vehicle_wait_state["wait_time"] * 15
# #                                                 current_station1.total_reward -= vehicle_wait_state["wait_time"] * 15
#                                                 current_station2.single_reward -= vehicle_wait_state["wait_time"] * 15
# #                                                 current_station2.total_reward -= vehicle_wait_state["wait_time"] * 15
#                                                 print('小惩罚')
                                            vehicle_wait_state["wait_time"] = 0
                                            station_current_state['vehicle_num'][1] -= 1
                                            # print(vehicle_id, '+服务完毕有等待车', current_vehicle.order)

                                        else:
                                            # print(vehicle_id, '+服务完毕且无等待车', current_vehicle.order)
                                            # print('当前服务状态:',
                                            #       station_states[current_station.station_id]['service_status'][1])
                                            station_current_state['service_status'][1] -= 1
                                            # print('当前服务状态:',
                                            #       station_states[current_station.station_id]['service_status'][1])

                        # 柴油 双舱
                        else:

                            if vehicle_state['oil_tank_empty_cabin1'] == 1:  # 判断第一舱是否为空
                                station_current_state['diesel'] += current_vehicle.capacity
                                current_station.oil_derv += current_vehicle.capacity
                                station_current_state['Vehicle dispatch count_derv'] -= 1
# #                                 print('奖励')
                                #满仓
                                if station_current_state['diesel'] >= current_station.capacity:
                                    station_current_state['diesel'] = current_station.capacity
                                vehicle_state['oil_tank_empty_cabin1'] = 0
                                vehicle_state['service_time'] = 0
                            else:
                                #订单等待
                                station_current_state['Vehicle dispatch double_diesel'] -= 1

                                # if (current_vehicle.order['time1'] / self.step_minutes) <= current_vehicle.order[
                                #     'time_deliver']:
                                #     current_station.total_reward +=2
                                if current_vehicle.order['time1'] / self.step_minutes > current_vehicle.order[
                                    'time_deliver']:
                                    current_station.total_reward -= 30
                                    self.over_order += 1
                                # current_station.total_reward -= (current_vehicle.order['time1'] / self.step_minutes - current_vehicle.order['time_deliver'])*0.05

                                # current_station.single_derv_reward -= (current_vehicle.order['time1']/self.step_minutes-current_station.time_deliver)*10
                                current_vehicle.order['time1'] = 0
                                #订单等待
                                station_current_state['diesel'] += current_vehicle.capacity
                                current_station.oil_derv += current_vehicle.capacity
# #                                 print('奖励')
                                if station_current_state['diesel'] >= current_station.capacity:
                                    station_current_state['diesel'] = current_station.capacity
                                vehicle_state['oil_tank_empty_cabin2'] = 0

                                vehicle_state['status'] = 1
                                vehicle_state['service_time'] = 0

                                if station_current_state['vehicle_num'][2] != 0:
                                    try:
#                                         print("Attempting to get vehicle from queue_derv")
                                        vehicle_wait = current_station.queue_derv.get(timeout=2)  # 设置超时为5秒
#                                         print(f"Successfully got vehicle: {vehicle_wait}")
                                        # 处理车辆等待的逻辑
                                    except queue.Empty:
                                        print("dervQueue is empty, could not retrieve vehicle within timeout period")
                                    vehicle_wait_state = vehicle_states[vehicle_wait.truck_id]
                                    vehicle_wait_state['status'] = 3
                                    #等待奖励
#                                     if vehicle_wait.order_type == 'not_combined':
#                                         current_station = New_characters.find_station_by_id(vehicle_wait.target[1])
#                                         current_station.single_reward -= vehicle_wait_state["wait_time"] * 15
# #                                         current_station.total_reward -= vehicle_wait_state["wait_time"] * 15
# #                                         print('小惩罚')
#                                     else:
#                                         current_station1 = New_characters.find_station_by_id(vehicle_wait.target[1])
#                                         current_station2 = New_characters.find_station_by_id(vehicle_wait.target[2])
#                                         current_station1.single_reward -= vehicle_wait_state["wait_time"] * 15
# #                                         current_station1.total_reward -= vehicle_wait_state["wait_time"] * 15
#                                         current_station2.single_reward -= vehicle_wait_state["wait_time"] * 15
# #                                         current_station2.total_reward -= vehicle_wait_state["wait_time"] * 15
#                                         print('小惩罚')
                                    vehicle_wait_state["wait_time"] = 0
                                    station_current_state['vehicle_num'][2] -= 1
                                else:
                                    station_current_state['service_status'][2]=0
                    elif current_vehicle.order_type == 'combined':

                        # print('服务双仓订单')
                        if current_location == current_vehicle.target[1]:
                            # if (current_vehicle.order['time1'] / self.step_minutes)<=current_vehicle.order['time_deliver1']:
                            #     current_station.total_reward +=0.5
                            if current_vehicle.order['time1'] / self.step_minutes > current_vehicle.order['time_deliver1']:
                                current_station.total_reward -= 30
                                self.over_order += 1
                            # current_station.total_reward -= (current_vehicle.order['time1'] / self.step_minutes - current_vehicle.order['time_deliver1'])*0.05

                            #改动
                            # print('服务双仓订单第一站')
                            oil_type = current_vehicle.oil_class_1
                            vehicle_state['oil_tank_empty_cabin1'] = 0
                            vehicle_state['service_time'] = 0
                            vehicle_state['status'] = 1
                            vehicle_state["distance_traveled"] = distance.alldistance[(current_vehicle.target[1],current_vehicle.target[2])]
                            if oil_type == '92':
                                station_current_state['Vehicle dispatch single_92'] -= 1
                                # station_current_state['Vehicle dispatch double_gasoline'] -= 1
                                station_current_state['Vehicle dispatch count_92'] -= 1
                                #订单等待
                                # current_station.single_92_reward -= (current_vehicle.order['time1']/self.step_minutes-current_station.time_deliver)*10
                                current_vehicle.order['time1'] = 0
                                #订单等待
                                station_current_state['92_gas'] += current_vehicle.capacity
                                current_station.oil_92 += current_vehicle.capacity
# #                                 print('奖励')
                                #满仓
                                if station_current_state['92_gas'] >= current_station.capacity:
                                    station_current_state['92_gas'] = current_station.capacity
                                if station_current_state['vehicle_num'][0] != 0:
                                    try:
#                                         print("Attempting to get vehicle from queue_92")
                                        vehicle_wait = current_station.queue_92.get(timeout=2)  # 设置超时为5秒
#                                         print(f"Successfully got vehicle: {vehicle_wait}")
                                        # 处理车辆等待的逻辑
                                    except queue.Empty:
                                        print("92Queue is empty, could not retrieve vehicle within timeout period")
                                    vehicle_wait_state = vehicle_states[vehicle_wait.truck_id]
                                    vehicle_wait_state['status'] = 3
                                    #等待奖励
#                                     if vehicle_wait.order_type == 'not_combined':
#                                         current_station = New_characters.find_station_by_id(vehicle_wait.target[1])
#                                         current_station.single_reward -= vehicle_wait_state["wait_time"] * 15
# #                                         current_station.total_reward -= vehicle_wait_state["wait_time"] * 15
# #                                         print('小惩罚')
#                                     else:
#                                         current_station1 = New_characters.find_station_by_id(vehicle_wait.target[1])
#                                         current_station2 = New_characters.find_station_by_id(vehicle_wait.target[2])
#                                         current_station1.single_reward -= vehicle_wait_state["wait_time"] * 15
# #                                         current_station1.total_reward -= vehicle_wait_state["wait_time"] * 15
#                                         current_station2.single_reward -= vehicle_wait_state["wait_time"] * 15
# #                                         current_station2.total_reward -= vehicle_wait_state["wait_time"] * 15
#                                         print('小惩罚')
                                    vehicle_wait_state["wait_time"] = 0
                                    station_current_state['vehicle_num'][0] -= 1
                                else:
                                    station_current_state['service_status'][0] = 0
                            if oil_type == '95':
                                station_current_state['Vehicle dispatch single_95'] -= 1
                                # station_current_state['Vehicle dispatch double_gasoline'] -= 1
                                #订单等待
                                # current_station.single_95_reward -= (current_vehicle.order['time1']/self.step_minutes-current_station.time_deliver)*10
                                current_vehicle.order['time1'] = 0
                                #订单等待
                                station_current_state['Vehicle dispatch count_95'] -= 1
                                station_current_state['95_gas'] += current_vehicle.capacity
                                current_station.oil_95 += current_vehicle.capacity
# #                                 print('奖励')
                                #满仓
                                if station_current_state['95_gas'] >= current_station.capacity:
                                    station_current_state['95_gas'] = current_station.capacity
                                if station_current_state['vehicle_num'][1] != 0:
                                    try:
#                                         print("Attempting to get vehicle from queue_95")
                                        vehicle_wait = current_station.queue_95.get(timeout=2)  # 设置超时为5秒
#                                         print(f"Successfully got vehicle: {vehicle_wait}")
                                        # 处理车辆等待的逻辑
                                    except queue.Empty:
                                        print("95Queue is empty, could not retrieve vehicle within timeout period")
                                    vehicle_wait_state = vehicle_states[vehicle_wait.truck_id]
                                    vehicle_wait_state['status'] = 3
                                    # print(vehicle_wait.truck_id, '+等待车准备服务95', vehicle_wait.order)
                                    # print('当前服务状态:',
                                    #       station_states[current_station.station_id]['service_status'][1])
                                    #等待奖励
#                                     if vehicle_wait.order_type == 'not_combined':
#                                         current_station = New_characters.find_station_by_id(vehicle_wait.target[1])
#                                         current_station.single_reward -= vehicle_wait_state["wait_time"] * 15
# #                                         current_station.total_reward -= vehicle_wait_state["wait_time"] * 15
# #                                         print('小惩罚')
#                                     else:
#                                         current_station1 = New_characters.find_station_by_id(vehicle_wait.target[1])
#                                         current_station2 = New_characters.find_station_by_id(vehicle_wait.target[2])
#                                         current_station1.single_reward -= vehicle_wait_state["wait_time"] * 15
# #                                         current_station1.total_reward -= vehicle_wait_state["wait_time"] * 15
#                                         current_station2.single_reward -= vehicle_wait_state["wait_time"] * 15
# #                                         current_station2.total_reward -= vehicle_wait_state["wait_time"] * 15
#                                         print('小惩罚')
                                    vehicle_wait_state["wait_time"] = 0
                                    station_current_state['vehicle_num'][1] -= 1
                                    # print(vehicle_id, '+服务完毕有等待车', current_vehicle.order)
                                else:
                                    # print(vehicle_id, '+服务完毕且无等待车', current_vehicle.order)
                                    # print('当前服务状态:',
                                    #       station_states[current_station.station_id]['service_status'][1])
                                    station_current_state['service_status'][1] -= 1
                                    # print('当前服务状态:',
                                    #       station_states[current_station.station_id]['service_status'][1])
                            if oil_type == 'derv':
                                station_current_state['Vehicle dispatch single_diesel'] -= 1
                                # station_current_state['Vehicle dispatch double_diesel'] -= 1
                                #订单等待
                                # current_station.single_derv_reward -= (current_vehicle.order['time1']/self.step_minutes-current_station.time_deliver)*10
                                current_vehicle.order['time1'] = 0
                                #订单等待
                                station_current_state['Vehicle dispatch count_derv'] -= 1
                                station_current_state['diesel'] += current_vehicle.capacity
                                current_station.oil_derv += current_vehicle.capacity
# #                                 print('奖励')
                                #满仓
                                if station_current_state['diesel'] >= current_station.capacity:
                                    station_current_state['diesel'] = current_station.capacity
                                if station_current_state['vehicle_num'][2] != 0:
                                    try:
#                                         print("Attempting to get vehicle from queue_derv")
                                        vehicle_wait = current_station.queue_derv.get(timeout=2)  # 设置超时为5秒
#                                         print(f"Successfully got vehicle: {vehicle_wait}")
                                        # 处理车辆等待的逻辑
                                    except queue.Empty:
                                        print("dervQueue is empty, could not retrieve vehicle within timeout period")
                                    vehicle_wait_state = vehicle_states[vehicle_wait.truck_id]
                                    vehicle_wait_state['status'] = 3
                                    #等待奖励
#                                     if vehicle_wait.order_type == 'not_combined':
#                                         current_station = New_characters.find_station_by_id(vehicle_wait.target[1])
#                                         current_station.single_reward -= vehicle_wait_state["wait_time"] * 15
# #                                         current_station.total_reward -= vehicle_wait_state["wait_time"] * 15
# #                                         print('小惩罚')
#                                     else:
#                                         current_station1 = New_characters.find_station_by_id(vehicle_wait.target[1])
#                                         current_station2 = New_characters.find_station_by_id(vehicle_wait.target[2])
#                                         current_station1.single_reward -= vehicle_wait_state["wait_time"] * 15
# #                                         current_station1.total_reward -= vehicle_wait_state["wait_time"] * 15
#                                         current_station2.single_reward -= vehicle_wait_state["wait_time"] * 15
# #                                         current_station2.total_reward -= vehicle_wait_state["wait_time"] * 15
#                                         print('小惩罚')
                                    vehicle_wait_state["wait_time"] = 0
                                    station_current_state['vehicle_num'][2] -= 1
                                else:
                                    station_current_state['service_status'][2] = 0
                        if current_location == current_vehicle.target[2]:
                            #改动
                            # print('服务双仓订单第二站')
                            # if current_vehicle.order['time2'] / self.step_minutes>current_vehicle.order['time_deliver2']:
                            #     print(1)
                            # if (current_vehicle.order['time2'] / self.step_minutes)<=current_vehicle.order['time_deliver2']:
                            #     current_station.total_reward +=0.5
                            if current_vehicle.order['time2'] / self.step_minutes > current_vehicle.order['time_deliver2']:
                                current_station.total_reward -= 30
                                self.over_order += 1
                            # current_station.total_reward -= (current_vehicle.order['time2'] / self.step_minutes - current_vehicle.order['time_deliver2'])*0.05

                            oil_type = current_vehicle.oil_class_2
                            vehicle_state['oil_tank_empty_cabin2'] = 0
                            vehicle_state['service_time'] = 0
                            vehicle_state['status'] = 1
                            vehicle_state["distance_traveled"]=distance.alldistance[(current_vehicle.target[2],current_vehicle.target[3])]
                            if oil_type == '92':
                                station_current_state['Vehicle dispatch single_92'] -= 1
                                # station_current_state['Vehicle dispatch double_gasoline'] -= 1
                                #订单等待
                                # current_station.single_92_reward -= (current_vehicle.order['time2']/self.step_minutes-current_station.time_deliver)*10
                                current_vehicle.order['time2'] = 0
                                #订单等待
                                station_current_state['Vehicle dispatch count_92'] -= 1
                                station_current_state['92_gas'] += current_vehicle.capacity
                                current_station.oil_92 += current_vehicle.capacity
# #                                 print('奖励')
                                #满仓
                                if station_current_state['92_gas'] >= current_station.capacity:
                                    station_current_state['92_gas'] = current_station.capacity
                                if station_current_state['vehicle_num'][0] != 0:
                                    try:
#                                         print("Attempting to get vehicle from queue_92")
                                        vehicle_wait = current_station.queue_92.get(timeout=2)  # 设置超时为5秒
#                                         print(f"Successfully got vehicle: {vehicle_wait}")
                                        # 处理车辆等待的逻辑
                                    except queue.Empty:
                                        print("92Queue is empty, could not retrieve vehicle within timeout period")
                                    vehicle_wait_state = vehicle_states[vehicle_wait.truck_id]
                                    vehicle_wait_state['status'] = 3
                                    # 等待奖励
#                                     if vehicle_wait.order_type == 'not_combined':
#                                         current_station = New_characters.find_station_by_id(vehicle_wait.target[1])
#                                         current_station.single_reward -= vehicle_wait_state["wait_time"] * 15
# #                                         current_station.total_reward -= vehicle_wait_state["wait_time"] * 15
# #                                         print('小惩罚')
#                                     else:
#                                         current_station1 = New_characters.find_station_by_id(vehicle_wait.target[1])
#                                         current_station2 = New_characters.find_station_by_id(vehicle_wait.target[2])
#                                         current_station1.single_reward -= vehicle_wait_state["wait_time"] * 15
# #                                         current_station1.total_reward -= vehicle_wait_state["wait_time"] * 15
#                                         current_station2.single_reward -= vehicle_wait_state["wait_time"] * 15
# #                                         current_station2.total_reward -= vehicle_wait_state["wait_time"] * 15
#                                         print('小惩罚')
                                    vehicle_wait_state["wait_time"] = 0
                                    station_current_state['vehicle_num'][0] -= 1
                                else:
                                    station_current_state['service_status'][0] = 0
                            if oil_type == '95':
                                station_current_state['Vehicle dispatch single_95'] -= 1
                                # station_current_state['Vehicle dispatch double_gasoline'] -= 1
                                #订单等待
                                # current_station.single_95_reward -= (current_vehicle.order['time2']/self.step_minutes-current_station.time_deliver)*10
                                current_vehicle.order['time2'] = 0
                                #订单等待
                                station_current_state['Vehicle dispatch count_95'] -= 1
                                station_current_state['95_gas'] += current_vehicle.capacity
                                current_station.oil_95 += current_vehicle.capacity
# #                                 print('奖励')
                                #满仓
                                if station_current_state['95_gas'] >= current_station.capacity:
                                    station_current_state['95_gas'] = current_station.capacity
                                if station_current_state['vehicle_num'][1] != 0:
                                    try:
#                                         print("Attempting to get vehicle from queue_95")
                                        vehicle_wait = current_station.queue_95.get(timeout=2)  # 设置超时为5秒
#                                         print(f"Successfully got vehicle: {vehicle_wait}")
                                        # 处理车辆等待的逻辑
                                    except queue.Empty:
                                        print("95Queue is empty, could not retrieve vehicle within timeout period")
                                    vehicle_wait_state = vehicle_states[vehicle_wait.truck_id]
                                    vehicle_wait_state['status'] = 3
                                    # print(vehicle_wait.truck_id, '+等待车准备服务95', vehicle_wait.order)
                                    # print('当前服务状态:',
                                    #       station_states[current_station.station_id]['service_status'][1])
                                    #等待奖励
#                                     if vehicle_wait.order_type == 'not_combined':
#                                         current_station = New_characters.find_station_by_id(vehicle_wait.target[1])
#                                         current_station.single_reward -= vehicle_wait_state["wait_time"] * 15
# #                                         current_station.total_reward -= vehicle_wait_state["wait_time"] * 15
# #                                         print('小惩罚')
#                                     else:
#                                         current_station1 = New_characters.find_station_by_id(vehicle_wait.target[1])
#                                         current_station2 = New_characters.find_station_by_id(vehicle_wait.target[2])
#                                         current_station1.single_reward -= vehicle_wait_state["wait_time"] * 15
# #                                         current_station1.total_reward -= vehicle_wait_state["wait_time"] * 15
#                                         current_station2.single_reward -= vehicle_wait_state["wait_time"] * 15
# #                                         current_station2.total_reward -= vehicle_wait_state["wait_time"] * 15
#                                         print('小惩罚')
                                    vehicle_wait_state["wait_time"] = 0
                                    station_current_state['vehicle_num'][1] -= 1
                                    # print(vehicle_id, '+服务完毕有等待车', current_vehicle.order)
                                else:
                                    # print(vehicle_id, '+服务完毕且无等待车', current_vehicle.order)
                                    # print('当前服务状态:',
                                    #       station_states[current_station.station_id]['service_status'][1])
                                    station_current_state['service_status'][1] -= 1
                                    # print('当前服务状态:',
                                    #       station_states[current_station.station_id]['service_status'][1])
                                    # station_current_state['service_status'][1] = 0
                            if oil_type == 'derv':
                                station_current_state['Vehicle dispatch single_diesel'] -= 1
                                # station_current_state['Vehicle dispatch double_diesel'] -= 1
                                #订单等待
                                # current_station.single_derv_reward -= (current_vehicle.order['time2']/self.step_minutes-current_station.time_deliver)*10
                                current_vehicle.order['time2'] = 0
                                #订单等待
                                station_current_state['Vehicle dispatch count_derv'] -= 1
                                station_current_state['diesel'] += current_vehicle.capacity
                                current_station.oil_derv += current_vehicle.capacity
                                # print('奖励')
                                #满仓
                                if station_current_state['diesel'] >= current_station.capacity:
                                    station_current_state['diesel'] = current_station.capacity
                                if station_current_state['vehicle_num'][2] != 0:
                                    try:
#                                         print("Attempting to get vehicle from queue_derv")
                                        vehicle_wait = current_station.queue_derv.get(timeout=2)  # 设置超时为5秒
#                                         print(f"Successfully got vehicle: {vehicle_wait}")
                                        # 处理车辆等待的逻辑
                                    except queue.Empty:
                                        print("92Queue is empty, could not retrieve vehicle within timeout period")
                                    vehicle_wait_state = vehicle_states[vehicle_wait.truck_id]
                                    vehicle_wait_state['status'] = 3
                                    #等待奖励
#                                     if vehicle_wait.order_type == 'not_combined':
#                                         current_station = New_characters.find_station_by_id(vehicle_wait.target[1])
#                                         current_station.single_reward -= vehicle_wait_state["wait_time"] * 15
# #                                         current_station.total_reward -= vehicle_wait_state["wait_time"] * 15
# #                                         print('小惩罚')
#                                     else:
#                                         current_station1 = New_characters.find_station_by_id(vehicle_wait.target[1])
#                                         current_station2 = New_characters.find_station_by_id(vehicle_wait.target[2])
#                                         current_station1.single_reward -= vehicle_wait_state["wait_time"] * 15
# #                                         current_station1.total_reward -= vehicle_wait_state["wait_time"] * 15
#                                         current_station2.single_reward -= vehicle_wait_state["wait_time"] * 15
# #                                         current_station2.total_reward -= vehicle_wait_state["wait_time"] * 15
#                                         print('小惩罚')
                                    vehicle_wait_state["wait_time"] = 0
                                    station_current_state['vehicle_num'][2] -= 1
                                else:
                                    station_current_state['service_status'][2] = 0

            # 装油中
            elif current_status == 4:

                # print(vehicle_id)
                # print(current_vehicle.current_refuel_cabin1)
                # print(current_vehicle.current_refuel_cabin2)
                # print(current_vehicle.oil_class_1)
                # print(current_vehicle.oil_class_2)
                # print(current_vehicle.order)
                # print(current_vehicle.order_type)
                #
                # print(vehicle_state)
                vehicle_state['total_time'] += self.step_minutes
                vehicle_state['refuel_time'] += self.step_minutes
                current_location = current_vehicle.target[0]

                if vehicle_state['refuel_time'] >= self.refuel_time:
                    depot_str = current_vehicle.target[0]
                    current_depot = New_characters.find_depot_by_id(depot_str)
                    depot_state = depot_states[current_depot.id]

                    # print(current_vehicle.truck_id)
                    # print('step_num',self.step_num)
                    # print('正在',{depot_str},'装95油车数量',depot_state['refueling_vehicles_count'][1])
                    # print('正在', {depot_str}, '等待95油车数量', depot_state['waiting_vehicles_count'][1])
                    # print(current_vehicle.order)

                    # 车辆类型为 汽油 单舱
                    if vehicle_state['vehicle_type'] == 0:
                        vehicle_state['oil_tank_empty_cabin1'] = 1
                        vehicle_state['status'] = 1
                        vehicle_state['refuel_time'] = 0

                        if current_vehicle.current_refuel_cabin1 == '92':
                            if depot_state['waiting_vehicles_count'][0] > 0:
                                try:
#                                     print("Attempting to get vehicle from queue_92")
                                    vehicle_wait = current_depot.queue_92.get(timeout=2)  # 设置超时为5秒
#                                     print(f"Successfully got vehicle: {vehicle_wait}")
                                    # 处理车辆等待的逻辑
                                except queue.Empty:
                                    print("92Queue is empty, could not retrieve vehicle within timeout period")
                                vehicle_wait_state = vehicle_states[vehicle_wait.truck_id]
                                vehicle_wait_state['status'] = 4
                                #改动
#                                 if vehicle_wait.order_type == 'not_combined':
#                                     current_station = New_characters.find_station_by_id(vehicle_wait.target[1])
#                                     current_station.single_reward -= vehicle_wait_state["wait_time"] * 15
# #                                     current_station.total_reward -= vehicle_wait_state["wait_time"] * 15
# #                                     print('小惩罚')
#                                 else:
#                                     current_station1 = New_characters.find_station_by_id(vehicle_wait.target[1])
#                                     current_station2 = New_characters.find_station_by_id(vehicle_wait.target[2])
#                                     current_station1.single_reward -= vehicle_wait_state["wait_time"] * 15
# #                                     current_station1.total_reward -= vehicle_wait_state["wait_time"] * 15
#                                     current_station2.single_reward -= vehicle_wait_state["wait_time"] * 15
# #                                     current_station2.total_reward -= vehicle_wait_state["wait_time"] * 15
#                                     print('小惩罚')
                                vehicle_wait_state["wait_time"]=0
                                depot_state['waiting_vehicles_count'][0] -= 1

                            else:
                                if depot_state['refueling_vehicles_count'][0]==0:
                                    print('92错误1')
                                depot_state['refueling_vehicles_count'][0] -= 1

                        else:
                            if depot_state['waiting_vehicles_count'][1] > 0:
                                try:
#                                     print("Attempting to get vehicle from queue_95")
                                    vehicle_wait = current_depot.queue_95.get(timeout=2)  # 设置超时为5秒
#                                     print(f"Successfully got vehicle: {vehicle_wait}")
                                    # 处理车辆等待的逻辑
                                except queue.Empty:
                                    print("95Queue is empty, could not retrieve vehicle within timeout period")
                                vehicle_wait_state = vehicle_states[vehicle_wait.truck_id]
                                vehicle_wait_state['status'] = 4
                                #等待奖励
#                                 if vehicle_wait.order_type == 'not_combined':
#                                     current_station = New_characters.find_station_by_id(vehicle_wait.target[1])
#                                     current_station.single_reward -= vehicle_wait_state["wait_time"] * 15
# #                                     current_station.total_reward -= vehicle_wait_state["wait_time"] * 15
# #                                     print('小惩罚')
#                                 else:
#                                     current_station1 = New_characters.find_station_by_id(vehicle_wait.target[1])
#                                     current_station2 = New_characters.find_station_by_id(vehicle_wait.target[2])
#                                     current_station1.single_reward -= vehicle_wait_state["wait_time"] * 15
# #                                     current_station1.total_reward -= vehicle_wait_state["wait_time"] * 15
#                                     current_station2.single_reward -= vehicle_wait_state["wait_time"] * 15
# #                                     current_station2.total_reward -= vehicle_wait_state["wait_time"] * 15
#                                     print('小惩罚')
                                vehicle_wait_state["wait_time"] = 0
                                # print('汽油单仓加95')
                                # print('等待车数量', depot_state['waiting_vehicles_count'][1])
                                depot_state['waiting_vehicles_count'][1] -= 1
                                # print('等待车数量', depot_state['waiting_vehicles_count'][1])
                                # print('不变')
                                # print(vehicle_wait .truck_id,'出来')
                                # print(vehicle_wait.order)


                            else:
                                if depot_state['refueling_vehicles_count'][1]==0:
                                    print('错误1')
                                # print('汽油单仓加95')
                                # print('正在装95车数量', depot_state['refueling_vehicles_count'][1])
                                depot_state['refueling_vehicles_count'][1] -= 1
                                # print('正在装95车数量', depot_state['refueling_vehicles_count'][1])
                                # print('等待车数量', depot_state['waiting_vehicles_count'][1])
                                # print('装油完成且无等待车')
                                # print('正在装油数量减一')

                    # 车辆类型为 柴油 单舱
                    elif vehicle_state['vehicle_type'] == 1:
                        vehicle_state['oil_tank_empty_cabin1'] = 1

                        vehicle_state['status'] = 1
                        vehicle_state['refuel_time'] = 0

                        if depot_state['waiting_vehicles_count'][2] > 0:
                            try:
#                                 print("Attempting to get vehicle from queue_derv")
                                vehicle_wait = current_depot.queue_derv.get(timeout=2)  # 设置超时为5秒
#                                 print(f"Successfully got vehicle: {vehicle_wait}")
                                # 处理车辆等待的逻辑
                            except queue.Empty:
                                print("dervQueue is empty, could not retrieve vehicle within timeout period")
                            vehicle_wait_state = vehicle_states[vehicle_wait.truck_id]
                            vehicle_wait_state['status'] = 4
                            #等待奖励
#                             if vehicle_wait.order_type == 'not_combined':
#                                 current_station = New_characters.find_station_by_id(vehicle_wait.target[1])
#                                 current_station.single_reward -= vehicle_wait_state["wait_time"] * 15
# #                                 current_station.total_reward -= vehicle_wait_state["wait_time"] * 15
# #                                 print('小惩罚')
#                             else:
#                                 current_station1 = New_characters.find_station_by_id(vehicle_wait.target[1])
#                                 current_station2 = New_characters.find_station_by_id(vehicle_wait.target[2])
#                                 current_station1.single_reward -= vehicle_wait_state["wait_time"] * 15
# #                                 current_station1.total_reward -= vehicle_wait_state["wait_time"] * 15
#                                 current_station2.single_reward -= vehicle_wait_state["wait_time"] * 15
# #                                 current_station2.total_reward -= vehicle_wait_state["wait_time"] * 15
#                                 print('小惩罚')
                            vehicle_wait_state["wait_time"] = 0
                            depot_state['waiting_vehicles_count'][2] -= 1
                            # depot_state['refueling_vehicles_count'][2] += 1
                        else:

                            depot_state['refueling_vehicles_count'][2] -= 1
                    # 车辆类型为 汽油 双舱
                    elif vehicle_state['vehicle_type'] == 2:
                        # 两舱装的是否为同一种汽油
                        if current_vehicle.current_refuel_cabin2 == current_vehicle.current_refuel_cabin1:
                            # 两舱装的都为92
                            if current_vehicle.current_refuel_cabin1 == '92':
                                # 判断第一舱是否为空
                                if vehicle_state['oil_tank_empty_cabin1'] == 0:
                                    vehicle_state['oil_tank_empty_cabin1'] = 1

                                    vehicle_state['refuel_time'] = 0
                                else:
                                    vehicle_state['oil_tank_empty_cabin2'] = 1

                                    vehicle_state['status'] = 1
                                    vehicle_state['refuel_time'] = 0

                                    if depot_state['waiting_vehicles_count'][0] >0:
                                        try:
#                                             print("Attempting to get vehicle from queue_92")
                                            vehicle_wait = current_depot.queue_92.get(timeout=2)  # 设置超时为5秒
#                                             print(f"Successfully got vehicle: {vehicle_wait}")
                                            # 处理车辆等待的逻辑
                                        except queue.Empty:
                                            print("92Queue is empty, could not retrieve vehicle within timeout period")
                                        vehicle_wait_state = vehicle_states[vehicle_wait.truck_id]
                                        vehicle_wait_state['status'] = 4
                                        #等待奖励
#                                         if vehicle_wait.order_type == 'not_combined':
#                                             current_station = New_characters.find_station_by_id(vehicle_wait.target[1])
#                                             current_station.single_reward -= vehicle_wait_state["wait_time"] * 15
# #                                             current_station.total_reward -= vehicle_wait_state["wait_time"] * 15
# #                                             print('小惩罚')
#                                         else:
#                                             current_station1 = New_characters.find_station_by_id(vehicle_wait.target[1])
#                                             current_station2 = New_characters.find_station_by_id(vehicle_wait.target[2])
#                                             current_station1.single_reward -= vehicle_wait_state["wait_time"] * 15
# #                                             current_station1.total_reward -= vehicle_wait_state["wait_time"] * 15
#                                             current_station2.single_reward -= vehicle_wait_state["wait_time"] * 15
# #                                             current_station2.total_reward -= vehicle_wait_state["wait_time"] * 15
#                                             print('小惩罚')
                                        vehicle_wait_state["wait_time"] = 0
                                        depot_state['waiting_vehicles_count'][0] -= 1

                                    else:
                                        if depot_state['refueling_vehicles_count'][0]==0:
                                            print('92错误2')
                                        depot_state['refueling_vehicles_count'][0] -= 1

                            # 两舱装的都为95
                            else:
                                if vehicle_state['oil_tank_empty_cabin1'] == 0:  # 判断第一舱是否为空
                                    vehicle_state['oil_tank_empty_cabin1'] = 1

                                    vehicle_state['refuel_time'] = 0
                                else:
                                    vehicle_state['oil_tank_empty_cabin2'] = 1

                                    vehicle_state['status'] = 1
                                    vehicle_state['refuel_time'] = 0

                                    if depot_state['waiting_vehicles_count'][1] > 0:
                                        try:
#                                             print("Attempting to get vehicle from queue_95")
                                            vehicle_wait = current_depot.queue_95.get(timeout=2)  # 设置超时为5秒
#                                             print(f"Successfully got vehicle: {vehicle_wait}")
                                            # 处理车辆等待的逻辑
                                        except queue.Empty:
                                            print("95Queue is empty, could not retrieve vehicle within timeout period")
                                        vehicle_wait_state = vehicle_states[vehicle_wait.truck_id]
                                        vehicle_wait_state['status'] = 4
                                        #等待奖励
#                                         if vehicle_wait.order_type == 'not_combined':
#                                             current_station = New_characters.find_station_by_id(vehicle_wait.target[1])
#                                             current_station.single_reward -= vehicle_wait_state["wait_time"] * 15
# #                                             current_station.total_reward -= vehicle_wait_state["wait_time"] * 15
# #                                             print('小惩罚')
#                                         else:
#                                             current_station1 = New_characters.find_station_by_id(vehicle_wait.target[1])
#                                             current_station2 = New_characters.find_station_by_id(vehicle_wait.target[2])
#                                             current_station1.single_reward -= vehicle_wait_state["wait_time"] * 15
# #                                             current_station1.total_reward -= vehicle_wait_state["wait_time"] * 15
#                                             current_station2.single_reward -= vehicle_wait_state["wait_time"] * 15
# #                                             current_station2.total_reward -= vehicle_wait_state["wait_time"] * 15
#                                             print('小惩罚')
                                        vehicle_wait_state["wait_time"] = 0
                                        # print('汽油双仓两舱加95')
                                        # print('等待车数量', depot_state['waiting_vehicles_count'][1])
                                        depot_state['waiting_vehicles_count'][1] -= 1
                                        # print('等待车数量', depot_state['waiting_vehicles_count'][1])
                                        # print('不变')
                                        # print(vehicle_wait.truck_id, '出来')
                                        # print(vehicle_wait.order)

                                    else:
                                        if  depot_state['refueling_vehicles_count'][1]==0:
                                            print('错误2')
                                        # print('汽油双仓两舱加95')
                                        # print('正在装95车数量', depot_state['refueling_vehicles_count'][1])
                                        depot_state['refueling_vehicles_count'][1] -= 1
                                        # print('正在装95车数量', depot_state['refueling_vehicles_count'][1])
                                        # print('等待车数量', depot_state['waiting_vehicles_count'][1])
                                        # print('装油完成且无等待车')
                                        # print('正在装油数量减一')

                        # 两舱装的不为同一种汽油
                        else:
                            # 判断第一舱是否为空
                            if vehicle_state['oil_tank_empty_cabin1'] == 0:
                                vehicle_state['oil_tank_empty_cabin1'] = 1

                                vehicle_state['refuel_time'] = 0
                                depot_str = current_vehicle.target[0]
                                current_depot = New_characters.find_depot_by_id(depot_str)
                                depot_state = depot_states[current_depot.id]
                                # 该车刚装完95号汽油，需要去装92号汽油
                                if current_vehicle.current_refuel_cabin1 == '95':
                                    if depot_state['waiting_vehicles_count'][1] > 0:
                                        try:
#                                             print("Attempting to get vehicle from queue_95")
                                            vehicle_wait = current_depot.queue_95.get(timeout=2)  # 设置超时为5秒
#                                             print(f"Successfully got vehicle: {vehicle_wait}")
                                            # 处理车辆等待的逻辑
                                        except queue.Empty:
                                            print("95Queue is empty, could not retrieve vehicle within timeout period")
                                        vehicle_wait_state = vehicle_states[vehicle_wait.truck_id]
                                        vehicle_wait_state['status'] = 4
                                        # 改动
#                                         if vehicle_wait.order_type == 'not_combined':
#                                             current_station = New_characters.find_station_by_id(vehicle_wait.target[1])
#                                             current_station.single_reward -= vehicle_wait_state["wait_time"] * 15
# #                                             current_station.total_reward -= vehicle_wait_state["wait_time"] * 15
# #                                             print('小惩罚')
#                                         else:
#                                             current_station1 = New_characters.find_station_by_id(vehicle_wait.target[1])
#                                             current_station2 = New_characters.find_station_by_id(vehicle_wait.target[2])
#                                             current_station1.single_reward -= vehicle_wait_state["wait_time"] * 15
# #                                             current_station1.total_reward -= vehicle_wait_state["wait_time"] * 15
#                                             current_station2.single_reward -= vehicle_wait_state["wait_time"] * 15
# #                                             current_station2.total_reward -= vehicle_wait_state["wait_time"] * 15
#                                             print('小惩罚')
                                        vehicle_wait_state["wait_time"] = 0
                                        # print('汽油双仓不同先加完95')
                                        # print('等待车数量', depot_state['waiting_vehicles_count'][1])
                                        depot_state['waiting_vehicles_count'][1] -= 1
                                        # print('等待车数量', depot_state['waiting_vehicles_count'][1])
                                        # print('不变')
                                        # print(vehicle_wait.truck_id, '出来')
                                        # print(vehicle_wait.order)

                                    else:
                                        if  depot_state['refueling_vehicles_count'][1]==0:
                                            print('错误3')
                                        # print('汽油双仓不同先加完95')
                                        # print('正在装95车数量', depot_state['refueling_vehicles_count'][1])
                                        depot_state['refueling_vehicles_count'][1] -= 1
                                        # print('正在装95车数量', depot_state['refueling_vehicles_count'][1])
                                        # print('等待车数量', depot_state['waiting_vehicles_count'][1])
                                        # print('装油完成且无等待车')
                                        # print('正在装油数量减一')



                                    if depot_state["refueling_vehicles_count"][0] < self.tube:
                                        vehicle_state["status"] = 4

                                        depot_state['refueling_vehicles_count'][0] += 1

                                    else:
                                        vehicle_state["status"] = 2
                                        depot_state["waiting_vehicles_count"][0] += 1
                                        # print(f"Adding vehicle {current_vehicle.truck_id} to queue_92")
                                        current_depot.queue_92.put(current_vehicle)
                                        # print(f"Successfully added vehicle {current_vehicle.truck_id} to queue_92")


                                # 该车刚装完92号汽油，需要去装95号汽油
                                else:
                                    if depot_state['waiting_vehicles_count'][0] > 0:
                                        try:
#                                             print("Attempting to get vehicle from queue_92")
                                            vehicle_wait = current_depot.queue_92.get(timeout=2)  # 设置超时为5秒
#                                             print(f"Successfully got vehicle: {vehicle_wait}")
                                            # 处理车辆等待的逻辑
                                        except queue.Empty:
                                            print("92Queue is empty, could not retrieve vehicle within timeout period")
                                        vehicle_wait_state = vehicle_states[vehicle_wait.truck_id]
                                        vehicle_wait_state['status'] = 4
                                        # 改动
#                                         if vehicle_wait.order_type == 'not_combined':
#                                             current_station = New_characters.find_station_by_id(vehicle_wait.target[1])
#                                             current_station.single_reward -= vehicle_wait_state["wait_time"] * 15
# #                                             current_station.total_reward -= vehicle_wait_state["wait_time"] * 15
# #                                             print('小惩罚')
#                                         else:
#                                             current_station1 = New_characters.find_station_by_id(vehicle_wait.target[1])
#                                             current_station2 = New_characters.find_station_by_id(vehicle_wait.target[2])
#                                             current_station1.single_reward -= vehicle_wait_state["wait_time"] * 15
# #                                             current_station1.total_reward -= vehicle_wait_state["wait_time"] * 15
#                                             current_station2.single_reward -= vehicle_wait_state["wait_time"] * 15
# #                                             current_station2.total_reward -= vehicle_wait_state["wait_time"] * 15
#                                             print('小惩罚')
                                        vehicle_wait_state["wait_time"] = 0
                                        depot_state['waiting_vehicles_count'][0] -= 1

                                    else:
                                        if depot_state['refueling_vehicles_count'][0]==0:
                                            print('92错误3')
                                        depot_state['refueling_vehicles_count'][0] -= 1




                                    if depot_state["refueling_vehicles_count"][1] < self.tube:
                                        vehicle_state["status"] = 4
                                        # print('汽油双仓不同后去加95')
                                        # print('正在装95油车数', depot_state["refueling_vehicles_count"][1])
                                        if depot_state["refueling_vehicles_count"][1] >self.tube:
                                            print('大错误')
                                        depot_state["refueling_vehicles_count"][1] += 1
                                        # print('95正在装油车加一')
                                        # print('正在装95油车数',depot_state["refueling_vehicles_count"][1])

                                    else:

                                        vehicle_state["status"] = 2
                                        # print('汽油双仓不同后去加95')
                                        # print('等待车数量', depot_state['waiting_vehicles_count'][1])
                                        depot_state["waiting_vehicles_count"][1] += 1
                                        # print('等待车数量', depot_state['waiting_vehicles_count'][1])
                                        # print('等待')
                                        # print(f"Adding vehicle {current_vehicle.truck_id} to queue_95")
                                        current_depot.queue_95.put(current_vehicle)
                                        # print(f"Successfully added vehicle {current_vehicle.truck_id} to queue_95")


                            else:
                                vehicle_state['oil_tank_empty_cabin2'] = 1
                                vehicle_state['status'] = 1
                                vehicle_state['refuel_time'] = 0
                                depot_str = current_vehicle.target[0]
                                current_depot = New_characters.find_depot_by_id(depot_str)
                                depot_state = depot_states[current_depot.id]
                                if current_vehicle.current_refuel_cabin2 == '92':
                                    if depot_state['waiting_vehicles_count'][0] > 0:
                                        try:
#                                             print("Attempting to get vehicle from queue_92")
                                            vehicle_wait = current_depot.queue_92.get(timeout=2)  # 设置超时为5秒
#                                             print(f"Successfully got vehicle: {vehicle_wait}")
                                            # 处理车辆等待的逻辑
                                        except queue.Empty:
                                            print("92Queue is empty, could not retrieve vehicle within timeout period")
                                        vehicle_wait_state = vehicle_states[vehicle_wait.truck_id]
                                        vehicle_wait_state['status'] = 4
                                        #等待奖励
#                                         if vehicle_wait.order_type == 'not_combined':
#                                             current_station = New_characters.find_station_by_id(vehicle_wait.target[1])
#                                             current_station.single_reward -= vehicle_wait_state["wait_time"] * 15
# #                                             current_station.total_reward -= vehicle_wait_state["wait_time"] * 15
# #                                             print('小惩罚')
#                                         else:
#                                             current_station1 = New_characters.find_station_by_id(vehicle_wait.target[1])
#                                             current_station2 = New_characters.find_station_by_id(vehicle_wait.target[2])
#                                             current_station1.single_reward -= vehicle_wait_state["wait_time"] * 15
# #                                             current_station1.total_reward -= vehicle_wait_state["wait_time"] * 15
#                                             current_station2.single_reward -= vehicle_wait_state["wait_time"] * 15
# #                                             current_station2.total_reward -= vehicle_wait_state["wait_time"] * 15
#                                             print('小惩罚')
                                        vehicle_wait_state["wait_time"] = 0
                                        depot_state['waiting_vehicles_count'][0] -= 1

                                    else:
                                        if depot_state['refueling_vehicles_count'][0]==0:
                                            print('92错误4')
                                        depot_state['refueling_vehicles_count'][0] -= 1

                                else:
                                    if depot_state['waiting_vehicles_count'][1] > 0:
                                        try:
#                                             print("Attempting to get vehicle from queue_95")
                                            vehicle_wait = current_depot.queue_95.get(timeout=2)  # 设置超时为5秒
#                                             print(f"Successfully got vehicle: {vehicle_wait}")
                                            # 处理车辆等待的逻辑
                                        except queue.Empty:
                                            print("95Queue is empty, could not retrieve vehicle within timeout period")
                                        vehicle_wait_state = vehicle_states[vehicle_wait.truck_id]
                                        vehicle_wait_state['status'] = 4
                                        #等待奖励
#                                         if vehicle_wait.order_type == 'not_combined':
#                                             current_station = New_characters.find_station_by_id(vehicle_wait.target[1])
#                                             current_station.single_reward -= vehicle_wait_state["wait_time"] * 15
# #                                             current_station.total_reward -= vehicle_wait_state["wait_time"] * 15
# #                                             print('小惩罚')
#                                         else:
#                                             current_station1 = New_characters.find_station_by_id(vehicle_wait.target[1])
#                                             current_station2 = New_characters.find_station_by_id(vehicle_wait.target[2])
#                                             current_station1.single_reward -= vehicle_wait_state["wait_time"] * 15
# #                                             current_station1.total_reward -= vehicle_wait_state["wait_time"] * 15
#                                             current_station2.single_reward -= vehicle_wait_state["wait_time"] * 15
# #                                             current_station2.total_reward -= vehicle_wait_state["wait_time"] * 15
#                                             print('小惩罚')
                                        vehicle_wait_state["wait_time"] = 0
                                        # print('汽油双仓不同第二仓加完95')
                                        # print('等待车数量', depot_state['waiting_vehicles_count'][1])
                                        depot_state['waiting_vehicles_count'][1] -= 1
                                        # print('等待车数量', depot_state['waiting_vehicles_count'][1])
                                        # print('不变')
                                        # print(vehicle_wait.truck_id, '出来')
                                        # print(vehicle_wait.order)

                                    else:
                                        if  depot_state['refueling_vehicles_count'][1]==0:
                                            print('错误4')
                                        # print('汽油双仓不同第二仓加完95')
                                        # print('正在装95车数量', depot_state['refueling_vehicles_count'][1])
                                        depot_state['refueling_vehicles_count'][1] -= 1
                                        # print('正在装95车数量', depot_state['refueling_vehicles_count'][1])
                                        # print('等待车数量', depot_state['waiting_vehicles_count'][1])
                                        # print('装油完成且无等待车')
                                        # print('正在装油数量减一')



                    # 车辆类型为 柴油 双舱
                    else:
                        if vehicle_state['oil_tank_empty_cabin1'] == 0:  # 判断第一舱是否为空
                            vehicle_state['oil_tank_empty_cabin1'] = 1

                            vehicle_state['refuel_time'] = 0
                        else:
                            vehicle_state['oil_tank_empty_cabin2'] = 1

                            vehicle_state['status'] = 1
                            vehicle_state['refuel_time'] = 0

                            if depot_state['waiting_vehicles_count'][2] > 0:
                                try:
#                                     print("Attempting to get vehicle from queue_derv")
                                    vehicle_wait = current_depot.queue_derv.get(timeout=2)  # 设置超时为5秒
#                                     print(f"Successfully got vehicle: {vehicle_wait}")
                                    # 处理车辆等待的逻辑
                                except queue.Empty:
                                    print("dervQueue is empty, could not retrieve vehicle within timeout period")
                                vehicle_wait_state = vehicle_states[vehicle_wait.truck_id]
                                vehicle_wait_state['status'] = 4
                                #等待奖励
#                                 if vehicle_wait.order_type == 'not_combined':
#                                     current_station = New_characters.find_station_by_id(vehicle_wait.target[1])
#                                     current_station.single_reward -= vehicle_wait_state["wait_time"] * 15
# #                                     current_station.total_reward -= vehicle_wait_state["wait_time"] * 15
# #                                     print('小惩罚')
#                                 else:
#                                     current_station1 = New_characters.find_station_by_id(vehicle_wait.target[1])
#                                     current_station2 = New_characters.find_station_by_id(vehicle_wait.target[2])
#                                     current_station1.single_reward -= vehicle_wait_state["wait_time"] * 15
# #                                     current_station1.total_reward -= vehicle_wait_state["wait_time"] * 15
#                                     current_station2.single_reward -= vehicle_wait_state["wait_time"] * 15
# #                                     current_station2.total_reward -= vehicle_wait_state["wait_time"] * 15
#                                     print('小惩罚')
                                vehicle_wait_state["wait_time"] = 0
                                depot_state['waiting_vehicles_count'][2] -= 1

                            else:

                                depot_state['refueling_vehicles_count'][2]-=1

            # depot_name = 'd1'

        # #改动，选完动作后，油库订单为空
        # for d in New_characters.alldepots:
        #     print(d.total_orders)

        self.next_state=[]
        self.next_state.append(depot_states)
        self.next_state.append(station_states)
        self.next_state.append(vehicle_states)
        self.step_num+=1
        self.terminate = self.check_done(self.step_num)

        agent_obs = self.get_state()
        agent_reward = []
        agent_cost=[]
        for station in self.allstations:
            agent_reward.append(station.total_reward)
            agent_cost.append(station.emp_cost)



        # agent_reward.append(agent_92_reward)
        # agent_reward.append(agent_95_reward)
        # agent_reward.append(agent_derv_reward)

        # print(reward)
        # 返回观测、奖励、是否结束和额外信息
        return agent_obs, agent_reward, self.terminate,agent_cost


        # else:
        #     reward = self.calculate_reward()
        #     # print(reward)
        #     # 返回观测、奖励、是否结束和额外信息
        #     return self.next_state, reward, self.terminate, {}


    #改动
    def reset(self):
        self.tube = 8
        self.singe_dis = 0
        self.double_dis = 0
        self.dc=0
        self.wait_time = 0
        self.terminate=False
        self.over_order = 0
        self.empty_time = 0
        self.full_time = 0
        self.dis_time = 0
        self.safe_dis = 0
        self.dis_cost=0
        self.safe_time = 0

        self.step_num = 0
        self.empty92 = 0
        self.empty95 = 0
        self.emptyderv = 0
        self.full92 = 0
        self.full95 = 0
        self.fullderv = 0
        self.dis92 = 0
        self.dis95 = 0
        self.disderv = 0
        self.gas_single = 0
        self.gas_double = 0
        self.derv_single = 0
        self.derv_double = 0

        temp_queue = Queue()
        # 重置加油站的状态为初始状态
        for station in self.allstations:
            self.cosm_92 = 0
            self.cosm_95 = 0
            self.cosm_derv = 0
            station.emp_cost = 0
            station.full_cost = 0
            station.total_reward = 0
            station.empty_stock_count=0
            # station.single_92_reward = 0
            # station.single_95_reward = 0
            # station.single_derv_reward = 0
            depot = get_path.get_path(station, self.alldepots)

            # dis1=math.ceil((self.alldistance[(station.station_id,depot.id)]/5)/self.step_minutes)
            # dis2 =self.service_time/self.step_minutes
            # dis3 = self.refuel_time/self.step_minutes
            # station.time_deliver=self.service_time/self.step_minutes+self.refuel_time/self.step_minutes+ math.ceil((self.alldistance[(station.station_id,depot.id)]/5)/self.step_minutes)
            # station.time_deliver =0
#             station.total_reward = 0
            station_id = str(station.station_id)
            station_state = {
                "92_gas": station.initial_oil_92,
                "95_gas": station.initial_oil_95,
                "diesel": station.initial_oil_derv,
                "vehicle_num": [0, 0, 0],
                "service_status": [0, 0, 0],
                'time_to_empty': [cal_oil.estimate_time_until_empty(station_id, '92', station.initial_oil_92,0, self.step_minutes),cal_oil.estimate_time_until_empty(station_id, '95', station.initial_oil_95,0, self.step_minutes),cal_oil.estimate_time_until_empty(station_id, 'diesel', station.initial_oil_derv,0, self.step_minutes)] ,
                'Vehicle dispatch single_gasoline': 0,
            'Vehicle dispatch double_gasoline': 0,
                # 'Vehicle dispatch single_diesel': 0,
                # 'Vehicle dispatch double_diesel': 0,
                'Vehicle dispatch count_92':0,
                'Vehicle dispatch count_95': 0,
                'Vehicle dispatch count_derv': 0,
                'Vehicle dispatch single_92': 0,
                'Vehicle dispatch single_95': 0,
                'Vehicle dispatch single_diesel': 0,
                'Vehicle dispatch double_92': 0,
                'Vehicle dispatch double_95': 0,
                'Vehicle dispatch double_diesel': 0,
                #改动
                "oil_percentage": [(station.initial_oil_92/station.capacity)*100,(station.initial_oil_95/station.capacity)*100,(station.initial_oil_derv/station.capacity)*100]
            }
            self.station_states[station_id] = station_state
        # 改动 5.24 wang
            # 油库对象属性初始化
            station.oil_92 = station.initial_oil_92
            station.oil_95 = station.initial_oil_95
            station.oil_derv = station.initial_oil_derv
            # 清空油站的所有队列
            while not station.queue_92.empty():
                station.queue_92.get()
            while not station.queue_95.empty():
                station.queue_95.get()
            while not station.queue_derv.empty():
                station.queue_derv.get()
            while not station.queue_92.empty():
                station.queue_92.get()
        # 不是第一次reset
        if len(self.vehicle_states) != 0:
            # print(f'len(self.vehicle_states):{len(self.vehicle_states)}')
            for vehicle in self.all_vehicle:
                vehicle_id = str(vehicle.truck_id)

                # 将车辆送回原油库
                if self.vehicle_states[vehicle_id]['status'] != 0:
                    if 'd1' in vehicle_id:
                        if 'v1' in vehicle_id:
                            self.alldepots[0].Single_gasoline.put(vehicle)
                        elif 'v2' in vehicle_id:
                            self.alldepots[0].Single_diesel.put(vehicle)
                        elif 'v3' in vehicle_id:
                            self.alldepots[0].Double_gasoline.put(vehicle)
                        elif 'v4' in vehicle_id:
                            self.alldepots[0].Double_diesel.put(vehicle)

                    elif 'd2' in vehicle_id:
                        if 'v1' in vehicle_id:
                            self.alldepots[1].Single_gasoline.put(vehicle)
                        elif 'v2' in vehicle_id:
                            self.alldepots[1].Single_diesel.put(vehicle)
                        elif 'v3' in vehicle_id:
                            self.alldepots[1].Double_gasoline.put(vehicle)
                        elif 'v4' in vehicle_id:
                            self.alldepots[1].Double_diesel.put(vehicle)

                    elif 'd3' in vehicle_id:
                        if 'v1' in vehicle_id:
                            self.alldepots[2].Single_gasoline.put(vehicle)
                        elif 'v2' in vehicle_id:
                            self.alldepots[2].Single_diesel.put(vehicle)
                        elif 'v3' in vehicle_id:
                            self.alldepots[2].Double_gasoline.put(vehicle)
                        elif 'v4' in vehicle_id:
                            self.alldepots[2].Double_diesel.put(vehicle)

                    vehicle_type_mapping = {
                        "v1": 0,
                        "v2": 1,
                        "v3": 2,
                        "v4": 3,
                    }
                    vehicle_type_str = vehicle_id.split('-')[1]
                    vehicle_type = vehicle_type_mapping.get(vehicle_type_str, 0)
                    vehicle_state = {
                        "distance_traveled": 0,
                        "status": 0,
                        "total_time": 0,
                        "time_elapsed": 0,
                        "service_time": 0,
                        "refuel_time": 0,
                        "vehicle_type": vehicle_type,
                        "wait_time": 0,
                        "oil_tank_empty_cabin1": 0,  # 只能取 0 或 1
                        "oil_tank_empty_cabin2": 0,
                        "target labeling": 0
                    }
                    self.vehicle_states[vehicle_id] = vehicle_state
                    # 车辆对象属性初始化
                    vehicle.target = []
                    vehicle.order = {}
                    vehicle.order_type = None
                    vehicle.oil_class_1 = None
                    vehicle.oil_class_2 = None
                    vehicle.current_refuel_cabin1 = None
                    vehicle.current_refuel_cabin2 = None
        # 第一次reset
        else:
            for vehicle in self.all_vehicle:
                vehicle_id = str(vehicle.truck_id)
                vehicle_type_mapping = {
                    "v1": 0,
                    "v2": 1,
                    "v3": 2,
                    "v4": 3,
                }
                vehicle_type_str = vehicle_id.split('-')[1]
                vehicle_type = vehicle_type_mapping.get(vehicle_type_str, 0)
                vehicle_state = {
                    "distance_traveled": 0,
                    "status": 0,
                    "total_time": 0,
                    "time_elapsed": 0,
                    "service_time": 0,
                    "refuel_time": 0,
                    "vehicle_type": vehicle_type,
                    "wait_time": 0,
                    "oil_tank_empty_cabin1": 0,  # 只能取 0 或 1
                    "oil_tank_empty_cabin2": 0,
                    "target labeling": 0
                }
                self.vehicle_states[vehicle_id] = vehicle_state
                # 车辆对象属性初始化
                vehicle.target = []
                vehicle.order = {}
                vehicle.order_type = None
                vehicle.oil_class_1 = None
                vehicle.oil_class_2 = None
                vehicle.current_refuel_cabin1 = None
                vehicle.current_refuel_cabin2 = None
        # / 改动 5.24 wang

        # 重置油库的状态为初始状态
        for depot in self.alldepots:
            depot_id = str(depot.id)
            oil_depot_state = {
                "vehicle_types_count":[depot.Single_diesel.qsize(), depot.Single_diesel.qsize(), depot.Double_gasoline.qsize(),depot.Double_diesel.qsize()], # 4种类型车各自的数量
                "refueling_vehicles_count": [0, 0, 0],  # 95号、92号、柴油此时正在装油的车辆数
                "waiting_vehicles_count": [0,0, 0]  # 95号、92号、柴油此时正在等待装油的车辆数
            }
            self.depot_states[depot_id] = oil_depot_state

            # 改动 5.24 wang
            # 油库对象属性初始化
            while not depot.queue_92.empty():
                depot.queue_92.get()

            while not depot.queue_95.empty():
                depot.queue_95.get()

            while not depot.queue_derv.empty():
                depot.queue_derv.get()

            depot.total_orders = []
            depot.diesel_orders = []
            depot.gasoline_orders = []
            depot.combined_orders = []
            depot.flag = False
            # / 改动 5.24 wang
        if len(self.initial_state) == 0:
            self.initial_state.append(self.depot_states)
            self.initial_state.append(self.station_states)
            self.initial_state.append(self.vehicle_states)
        self.next_state=self.initial_state
        # return self.initial_state
        # 返回车辆的初始观察值
        agent_obs = self.get_state()
        # agent_share_obs = self.get_share_state()

        # agent_obs = []
        # agent_obs_92 = self.get_obs_92()
        # agent_obs.append(agent_obs_92)
        # agent_obs_95 = self.get_obs_95()
        # agent_obs.append(agent_obs_95)
        # agent_obs_derv = self.get_obs_derv()
        # agent_obs.append(agent_obs_derv)

        return agent_obs



    def check_done(self, step_num):
        # 检查游戏是否结束，这里可以根据具体条件来定义
        # terminate=False
        if step_num== 288:
            return True
        # 这里简化为配送进度是否全部完成
        return False

    def get_obs_92(self):
        obs = []
        station_states = self.next_state[1]
        depot_states = self.next_state[0]
        for station_id, station_state in station_states.items():
            station_idd = []

            station_idd.append(station_state["oil_percentage"][0])
            station_idd.append(station_state['Vehicle dispatch count_92'])
            station_idd.append(station_state["time_to_empty"][0])
            station = New_characters.find_station_by_id(station_id)
            depot = get_path.get_path(station, self.alldepots)
            station_idd.append(depot_states[depot.id]["vehicle_types_count"][0])
            station_idd.append(depot_states[depot.id]["vehicle_types_count"][2])
            # station_idd.append(station.time_deliver)
            obs.append(station_idd)
        return obs

    def get_obs_95(self):
        obs = []
        station_states = self.next_state[1]
        depot_states = self.next_state[0]
        for station_id, station_state in station_states.items():
            station_idd = []
            station_idd.append(station_state["oil_percentage"][1])
            station_idd.append(station_state['Vehicle dispatch count_95'])
            station_idd.append(station_state["time_to_empty"][1])
            station = New_characters.find_station_by_id(station_id)
            depot = get_path.get_path(station, self.alldepots)
            station_idd.append(depot_states[depot.id]["vehicle_types_count"][0])
            station_idd.append(depot_states[depot.id]["vehicle_types_count"][2])
            # station_idd.append(station.time_deliver)
            obs.append(station_idd)
        return obs

    def get_obs_derv(self):
        obs = []
        station_states = self.next_state[1]
        depot_states = self.next_state[0]
        for station_id, station_state in station_states.items():
            station_idd = []
            station_idd.append(station_state["oil_percentage"][2])
            station_idd.append(station_state['Vehicle dispatch count_derv'])
            station_idd.append(station_state["time_to_empty"][2])
            station = New_characters.find_station_by_id(station_id)
            depot = get_path.get_path(station, self.alldepots)
            station_idd.append(depot_states[depot.id]["vehicle_types_count"][1])
            station_idd.append(depot_states[depot.id]["vehicle_types_count"][3])
            # station_idd.append(station.time_deliver)
            obs.append(station_idd)
        return obs

    # def new_get_obs(self):
    def get_share_state(self):
        depot_states = self.next_state[0]
        station_states = self.next_state[1]
        state = []
        for station_id, station_state in station_states.items():
                state.append(station_state["92_gas"])
                state.append(station_state["95_gas"])
                state.append(station_state["diesel"])
                state.append(station_state["vehicle_num"][0])
                state.append(station_state["vehicle_num"][1])
                state.append(station_state["vehicle_num"][2])
                state.append(station_state["service_status"][0])
                state.append(station_state["service_status"][1])
                state.append(station_state["service_status"][2])
                state.append(station_state["time_to_empty"][0])
                state.append(station_state["time_to_empty"][1])
                state.append(station_state["time_to_empty"][2])

                # state.append(station_state["oil_percentage"][0])
                # state.append(station_state["oil_percentage"][1])
                # state.append(station_state["oil_percentage"][2])
                # state.append(station_state['Vehicle dispatch count_92'])
                # state.append(station_state['Vehicle dispatch count_95'])
                # state.append(station_state['Vehicle dispatch count_derv'])

                state.append(station_state['Vehicle dispatch single_gasoline'])
                state.append(station_state['Vehicle dispatch double_gasoline'])
                state.append(station_state['Vehicle dispatch single_diesel'])
                state.append(station_state['Vehicle dispatch double_diesel'])

        for depot_id, depot_state in depot_states.items():
            state.append(depot_state["refueling_vehicles_count"][0])
            state.append(depot_state["refueling_vehicles_count"][1])
            state.append(depot_state["refueling_vehicles_count"][2])
            state.append(depot_state["waiting_vehicles_count"][0])
            state.append(depot_state["waiting_vehicles_count"][1])
            state.append(depot_state["waiting_vehicles_count"][2])
            state.append(depot_state["vehicle_types_count"][0])
            state.append(depot_state["vehicle_types_count"][1])
            state.append(depot_state["vehicle_types_count"][2])
            state.append(depot_state["vehicle_types_count"][3])
        return state
    def get_state(self):

        station_states = self.next_state[1]
        depot_states = self.next_state[0]
        station_obs=[]
        for station_id, station_state in station_states.items():
                tempstation = New_characters.find_station_by_id(station_id)
                state = []
                # state.append(tempstation.cosm_92)
                # state.append(tempstation.cosm_95)
                # state.append(tempstation.cosm_derv)
                # state.append(station_state["92_gas"])
                # state.append(station_state["95_gas"])
                # state.append(station_state["diesel"])
                # state.append(station_state["oil_percentage"][0])
                # state.append(station_state["oil_percentage"][1])
                # state.append(station_state["oil_percentage"][2])
                state.append(station_state["oil_percentage"][0])
                state.append(station_state["oil_percentage"][1])
                state.append(station_state["oil_percentage"][2])
                state.append(station_state['Vehicle dispatch single_92'])
                state.append(station_state['Vehicle dispatch double_92'])
                state.append(station_state['Vehicle dispatch single_95'])
                state.append(station_state['Vehicle dispatch double_95'])
                state.append(station_state['Vehicle dispatch single_diesel'])
                state.append(station_state['Vehicle dispatch double_diesel'])
                # state.append(station_state["vehicle_num"][0])
                # state.append(station_state["vehicle_num"][1])
                # state.append(station_state["vehicle_num"][2])
                # state.append(station_state["service_status"][0])
                # state.append(station_state["service_status"][1])
                # state.append(station_state["service_status"][2])

                # state.append(station_state['Vehicle dispatch count_92'])
                # state.append(station_state['Vehicle dispatch count_95'])
                # state.append(station_state['Vehicle dispatch count_derv'])
                # for depot_id, depot_state in depot_states.items():
                #     state.append(depot_state["refueling_vehicles_count"][0])
                #     state.append(depot_state["refueling_vehicles_count"][1])
                #     state.append(depot_state["refueling_vehicles_count"][2])
                #     state.append(depot_state["waiting_vehicles_count"][0])
                #     state.append(depot_state["waiting_vehicles_count"][1])
                #     state.append(depot_state["waiting_vehicles_count"][2])
                        # state.append(depot_states[depot_id]["vehicle_types_count"][0])
                        # state.append(depot_states[depot_id]["vehicle_types_count"][1])
                        # state.append(depot_states[depot_id]["vehicle_types_count"][2])
                        # state.append(depot_states[depot_id]["vehicle_types_count"][3])
                station_obs.append(state)

        # 提取油库和车辆的状态信息（可根据需求添加）




        # vehicle_states = self.next_state[2]
        # for vehicle_id, vehicle_state in vehicle_states.items():
        #
        #
        #         state.append(vehicle_state["distance_traveled"])
        #         state.append(vehicle_state["total_time"])
        #         state.append(vehicle_state["time_elapsed"])
        #         state.append(vehicle_state["refuel_time"])
        #         state.append(vehicle_state["wait_time"])
        #         state.append(vehicle_state["service_time"])
        #         state.append(vehicle_state["oil_tank_empty_cabin1"])
        #         state.append(vehicle_state["oil_tank_empty_cabin2"])
        #         state.append(vehicle_state["target labeling"])
        #         state.append(vehicle_state["vehicle_type"])
        #         state.append(vehicle_state["status"])


        return station_obs



    def get_avail_agent_actions(self):
        avail_action=[1,1,1,1,1]
        # avail_action = [0, 1, 2, 3, 4]
        return avail_action

    def close(self):
        # 关闭游戏环境
        pass

# env=OilDeliveryEnv1()
# env.reset()
# episode_reward=0
# episode_cost=0

# episode_reward=[]
# while not env.terminate:
#
#     rewards_avg=0
#     cost_avg=0
#     actionss = []
#     actions_92 = []
#     actions_95 = []
#     actions_derv = []
#     for station_index in range(6):
#         random.seed(time.time())
#         a = random.randint(0, 2)
#         # b=random.randint(0, 1)
#         c=random.randint(0, 2)
#         if a==0:
#             actions_92.append(0)
#             actions_95.append(0)
#             actions_derv.append(0)
#         elif a==1:
#
#                 if c==0:
#                     actions_92.append(1)
#                     actions_95.append(0)
#                     actions_derv.append(0)
#                 elif c == 1:
#                     actions_92.append(0)
#                     actions_95.append(1)
#                     actions_derv.append(0)
#                 elif c == 2:
#                     actions_92.append(0)
#                     actions_95.append(0)
#                     actions_derv.append(1)
#
#         elif a == 2:
#
#                 if c==0:
#                     actions_92.append(2)
#                     actions_95.append(0)
#                     actions_derv.append(0)
#                 elif c == 1:
#                     actions_92.append(0)
#                     actions_95.append(2)
#                     actions_derv.append(0)
#                 elif c == 2:
#                     actions_92.append(0)
#                     actions_95.append(0)
#                     actions_derv.append(2)
#     actionss.append(actions_92)
#     actionss.append(actions_95)
#     actionss.append(actions_derv)
#
#     nextstate, reward, terminate,cost = env.step(actionss)
#     cost = np.array(cost)
#     reward=np.array(reward)
#     rewards_avg = np.mean(reward)
#     cost_avg = np.sum(cost)
#     # print(cost_avg)
#     episode_reward+=rewards_avg
#     episode_cost+=cost_avg
# print('总回报：',episode_reward)
# print('总不安全：',episode_cost)
#
#
#
