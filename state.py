import random
import math


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

        # Attributes
        self.cost_per_km: float = 30
        self.profit_threshold = 35_000
        self.minimum_profit_per_km: float = random.uniform(25, 40)

        # sentiment
        self.satisfaction_score = 0.8
        self.is_active: bool = True

    def reset(self):
        self.monthly_profit = 0
        self.num_orders_completed = 0
        self.total_kms_driven = 0.0

    def accept_order(self, order: dict, policy: dict) -> bool:
        earnings = order["distance"] * policy["rate_per_km"]
        cost = order["distance"] * self.cost_per_km

        profit = earnings - cost
        profit_per_km = profit / order["distance"] if order["distance"] > 0 else 0

        if profit_per_km >= self.minimum_profit_per_km:
            self.complete_order(order, earnings, cost)
            return True
        else:
            return False

    def complete_order(self, order, earnings, cost):
        self.num_orders_completed += 1
        self.total_kms_driven += order["distance"]
        self.monthly_profit += earnings - cost

    def update_sentiment(self, alpha=0.25):
        profit_ratio = self.monthly_profit / self.profit_threshold
        current_feeling = 1 / (1 + math.exp(-10 * (profit_ratio - 1)))

        self.satisfaction_score = (alpha * current_feeling) + (
            (1 - alpha) * self.satisfaction_score
        )

        churn_probability = (1 - self.satisfaction_score) ** 2

        if random.random() < churn_probability:
            self.is_active = False
