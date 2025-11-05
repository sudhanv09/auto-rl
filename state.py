
class Company:
    def __init__(self, total_drivers: int, seed_money: int):
        self.total_drivers = total_drivers
        self.seed_money = seed_money
        
        self.profit: int = 0
        self.driver_churn: float = 0.0

        self.avg_driver_earnings: float = None
        self.avg_order_price: float = None
        self.avg_eta: float = None
        self.rejected_rate: float = None


class Driver:
    def __init__(self):
        self.profit: int = 0
        self.satisfaction_score: float = 1.0

        self.num_order_completed: int = 0
        self.avg_idle_time: float = 0.0
        self.avg_delivery_distance: float = 0.0