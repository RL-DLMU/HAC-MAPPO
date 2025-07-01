import json
import random


with open("D:\BaiduSyncdisk\LCPPO改油耗数据实验室工作日1\workday_consumption.json", 'r')as file:
#with open("D:\BaiduSyncdisk\LCPPO改油耗数据实验室工作日1\holiday_consumption.json", 'r')as file:

# with open("D:/BaiduSyncdisk/验证实验/LCPPO改油耗数据实验室2/workday_hourly_consumption.json", 'r')as file:
   data=json.load(file)
def cal_remain_oil(station_id,oil_class,current_oil,step_num,step_minute):
    if step_num == 0:
        return current_oil,0
    random_factor = random.uniform(-1, 1)

    # 应用随机比例到 consumption
    consumption = data[station_id][oil_class]
    # consumption = data[station_id][oil_class]

    i=int((step_num-1)*step_minute/60)%24

    step_per_hour=int(60/step_minute)
    a = consumption[i] / step_per_hour
    if current_oil<=0:
        remain_oil=0
    elif current_oil-consumption[i]/step_per_hour<=0:
        remain_oil=0
    else:
        remain_oil=current_oil-consumption[i] * (1 + random_factor)/step_per_hour
        #remain_oil = current_oil - consumption[i]  / step_per_hour
    return remain_oil,a

def estimate_time_until_empty(station_id,oil_class,remain_oil,step_num,step_minute):
    # 确定油品消耗数据
    if oil_class == '92':
        consumption = data[station_id]["92"]
    elif oil_class == '95':
        consumption = data[station_id]["95"]
    else:
        consumption = data[station_id]["diesel"]

    # 初始化时间变量
    time = step_num * step_minute  # 当前总时间（分钟）
    step_per_hour = int(60 / step_minute)  # 每小时的步数
    current_hour = int(time / 60) % 24  # 当前小时
    remaining_minutes_in_hour = 60 - time % 60  # 当前小时剩余的分钟数

    # 消耗掉当前小时剩余时间内的油量
    remain_oil -= consumption[current_hour] / step_per_hour * (remaining_minutes_in_hour / step_minute)
    diff_step = (remaining_minutes_in_hour / step_minute)

    # 检查是否已经耗尽油量
    if remain_oil <= 0:
        return 0

    # 逐小时扣减油量，找到油耗尽的时间点
    for hour_step in range(1, 1000000):
        next_hour = (current_hour + hour_step) % 24  # 下一小时的索引
        remain_oil -= consumption[next_hour]  # 扣减每小时消耗的油量

        if remain_oil <= 0:
            # 计算油耗尽的小时数，并跳出循环
            break

    # 计算油耗尽后还需多少步
    hour_steps_taken = hour_step * step_per_hour  # 每小时步数乘以耗时小时数
    for extra_step in range(step_per_hour):
        next_hour = (current_hour + hour_step) % 24
        remain_oil -= consumption[next_hour] / step_per_hour  # 每步扣除油量
        if remain_oil <= 0:
            hour_steps_taken += extra_step  # 加上多余步数
            break

    remain_step = int(diff_step + hour_steps_taken)  # 总步数

    return remain_step

# test

