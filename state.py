
class Company:
    def __init__(self, drivers: int, seed_money: int):
        self.total_drivers = drivers
        self.seed_money = seed_money
        
        # Balance sheet
        self.current_balance: float = seed_money
        self.last_epoch_profit: float = 0.0
        self.consecutive_loss_epochs: int = 0

        # Driver metrics
        self.active_drivers = drivers
        self.driver_churn_rate: float = 0.0
        self.new_signups: int = 0

        # Operation Metrics
        self.avg_driver_profit: float = 0.0
        self.avg_order_price: float = 0.0
        self.avg_completion_rate: float = 1.0


class Driver:
    def __init__(self, driver_id: int):
        self.id = driver_id

        # Performance
        self.monthly_profit: float = 0

        self.num_orders_completed: int = 0
        self.total_kms_driven: float = 0.0
        self.cost_per_km: float = 0.3
        
        self.profit_threshold = 35_000
        self.satisfaction_score = 0.8
        self.is_active: bool = True

    