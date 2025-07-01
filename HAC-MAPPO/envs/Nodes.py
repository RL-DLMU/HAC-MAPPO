import queue
class depot():
    def __init__(self,id,location,tube):

        self.id = id # 油库名
        self.location = location  # 油库位置
        self.Single_gasoline = queue.Queue()
        self.Single_diesel = queue.Queue()
        self.Double_gasoline = queue.Queue()
        self.Double_diesel = queue.Queue()
        self.tube = tube  # 鹤管数量
        self.queue_92 = queue.Queue()
        self.queue_95 = queue.Queue()
        self.queue_derv = queue.Queue()
        self.total_orders=[]
        self.diesel_orders = []
        self.gasoline_orders = []
        self.combined_orders = []
        self.flag=False


class station():
    def __init__(self, station_id, location , capacity,oil_92,oil_95,oil_derv):
        self.station_id = station_id  # 加油站id
        self.location = location      # 加油站位置
        self.capacity = capacity        # 三种油品最大容量
        self.initial_oil_92 = oil_92  # 保存92号汽油的初始值
        self.initial_oil_95 = oil_95  # 保存95号汽油的初始值
        self.initial_oil_derv = oil_derv  # 保存柴油的初始值
        self.oil_92 = oil_92
        self.oil_95 = oil_95
        self.oil_derv = oil_derv
        self.cosm_92 = 0
        self.cosm_95 = 0
        self.cosm_derv = 0
        self.emp_cost=0
        self.full_cost = 0
        self.queue_92 = queue.Queue()
        self.queue_95 = queue.Queue()
        self.queue_derv = queue.Queue()
        self.single_92_reward=0
        self.single_95_reward = 0
        self.single_derv_reward = 0
        self.total_reward = 0
        self.time_deliver=0
        self.empty_stock_count=0
class TankerTruck:
    def __init__(self, capacity, truck_id,speed,cabin,cost):
        self.capacity = capacity          # 车厢容量
        self.truck_id = truck_id          # 车类id
        self.speed = speed
        self.cabin = cabin  # 油罐的数量（1或2）
        self.cost=cost
        self.target=[] # 路径
        self.order={}
        self.order_type=None
        self.oil_class_1 = None
        self.oil_class_2 = None
        self.current_refuel_cabin1 = None
        self.current_refuel_cabin2 = None


