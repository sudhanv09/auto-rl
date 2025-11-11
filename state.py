import numpy as np

class GlobalState:
    def __init__(self):
        self.current_epoch: int = 0
        self.total_orders_available: int = 0
        self.total_orders_complete: int = 0


class Company:
    def __init__(self, drivers: int, seed_money: int):
        self.total_drivers = drivers
        self.seed_money = seed_money

        # Balance sheet
        self.current_balance: float = seed_money
        self.last_epoch_profit: float = 0.0
        self.consecutive_loss_epochs: int = 0

        # Internal
        self.epoch_expense: float = 0.0
        self.epoch_revenue: float = 0.0

        # Driver metrics
        self.active_drivers = drivers
        self.driver_churn_rate: float = 0.0
        self.new_signups: int = 0

        # Operation Metrics
        self.avg_driver_earnings: float = 0.0
        self.avg_driver_profit: float = 0.0
        self.avg_order_price: float = 0.0
        self.avg_completion_rate: float = 1.0

    def reset(self):
        self.epoch_revenue = 0.0
        self.epoch_expenses = 0.0

    def process_payment_to_driver(self, payment_amount: float):
        self.epoch_expenses += payment_amount

    def record_revenue_from_order(self, order_fee: float):
        self.epoch_revenue += order_fee

    def finalize_epoch_accounting(self):
        self.last_epoch_profit = self.epoch_revenue - self.epoch_expenses
        self.current_balance += self.last_epoch_profit

        if self.last_epoch_profit < 0:
            self.consecutive_loss_epochs += 1
        else:
            self.consecutive_loss_epochs = 0
            
    def update_driver_fleet(self, all_drivers: list, base_signup_rate: int = 10):
        quit_count = sum(1 for driver in all_drivers if not driver.is_active)
        
        active_driver_fleet = [driver for driver in all_drivers if driver.is_active]

        # Simulate new driver signups based on last month's average earnings
        if self.avg_driver_earnings > 0:
            attraction_factor = self.avg_driver_earnings / 1000.0 
            num_new_signups = int(base_signup_rate * attraction_factor)
        else:
            num_new_signups = 0
        
        self.new_driver_signups = num_new_signups
        
        # Add the new drivers to the fleet
        last_driver_id = all_drivers[-1].id if all_drivers else 0
        for i in range(num_new_signups):
            new_driver_id = last_driver_id + i + 1
            active_driver_fleet.append(Driver(new_driver_id))
            
        self.driver_churn_rate = quit_count / len(all_drivers) if all_drivers else 0.0
        
        return active_driver_fleet
    
    def epoch_report(self, all_drivers: list, state):
        self.active_drivers = len(all_drivers)

        if self.active_drivers > 0:
            all_earnings = [d.monthly_profit + (d.total_kms_driven * d.cost_per_km) for d in all_drivers]
            all_profits = [d.monthly_profit for d in all_drivers]
            self.avg_driver_earnings = np.mean(all_earnings)
            self.avg_driver_profit = np.mean(all_profits)
        else:
            self.avg_driver_earnings = 0.0
            self.avg_driver_profit = 0.0
            
        # Calculate order completion rate
        if state.total_orders_available > 0:
            self.order_completion_rate = state.total_orders_completed / state.total_orders_available
        else:
            self.order_completion_rate = 1.0


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
        self.minimum_profit_per_km: float = np.random.uniform(25, 40)

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
        current_feeling = 1 / (1 + np.exp(-10 * (profit_ratio - 1)))

        self.satisfaction_score = (alpha * current_feeling) + (
            (1 - alpha) * self.satisfaction_score
        )

        churn_probability = (1 - self.satisfaction_score) ** 2

        if np.random.rand() < churn_probability:
            self.is_active = False
