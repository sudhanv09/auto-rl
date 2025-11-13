import numpy as np

class GlobalState:
    def __init__(self, base_demand=50):
        self.current_epoch: int = 0
        self.base_demand = base_demand
        self.market_demand_multiplier = 1.0 
        self.orders_queue = []


class Order:
    def __init__(self, order_id, distance_km, base_price=40, base_expected_wage=15):
        self.id = order_id
        self.distance = distance_km
        self.time_waiting = 0
        self.max_wait_time = 5
        
        self.customer_price = base_price + (base_expected_wage * distance_km)


class Company:
    def __init__(self, seed_money_twd=1_000_000):
        self.balance = seed_money_twd
        
        # Metrics for the Agent
        self.daily_profit = 0
        self.daily_revenue = 0
        self.daily_cost = 0
        self.daily_delivered = 0
        self.daily_expired = 0
    
    def reset_daily_stats(self):
        self.daily_profit = 0
        self.daily_revenue = 0
        self.daily_cost = 0
        self.daily_delivered = 0
        self.daily_expired = 0

    def process_transaction(self, revenue, driver_payout):
        self.balance += (revenue - driver_payout)
        self.daily_revenue += revenue
        self.daily_cost += driver_payout
        self.daily_profit += (revenue - driver_payout)
        self.daily_delivered += 1


class Driver:
    def __init__(self, driver_id):
        self.id = driver_id
        self.is_active = True

        self.cost_per_km = 2.0
        self.avg_speed_kmh = 20.0
        self.target_hourly_wage = np.random.uniform(160, 250)

        self.monthly_profit_threshold = 35_000
        self.current_month_profit = 0.0
        self.satisfaction = 100.0
        

    def evaluate_offer(self, order: Order, offered_payout: float) -> bool:
        """
        Returns True if the driver accepts the offer.
        """
        trip_time_hours = order.distance / self.avg_speed_kmh
        
        wage_cost = trip_time_hours * self.target_hourly_wage
        material_cost = order.distance * self.cost_per_km
        
        min_required_payout = wage_cost + material_cost
        
        if offered_payout >= min_required_payout:
            net_profit = offered_payout - material_cost
            self.current_month_profit += net_profit
            return True
        
        return False
    
    def end_of_month_review(self) -> bool:
        if not self.is_active: 
            return False

        performance_ratio = self.current_month_profit / self.monthly_profit_threshold
        
        if performance_ratio >= 1.0:
            self.satisfaction = min(100.0, self.satisfaction + 10.0)
        else:
            penalty = 30.0 * (1.0 - performance_ratio)
            self.satisfaction -= penalty

        self.current_month_profit = 0.0

        if self.satisfaction <= 0:
            self.is_active = False
            return True # Driver Quit
        
        return False # Driver Stayed
