from state import GlobalState, Company, Driver, Order
import numpy as np

class Env:
    def __init__(self, config: dict):
        self.config = config

        self.company = Company()
        self.drivers = [Driver(i) for i in range(config["driver_count"])]
        self.state = GlobalState(base_demand=100)

        self.day_counter = 0

        # Action Space: Payout Ratio (0.4 to 1.2)
        # 0.4 = Greedy Company (Driver gets 40%)
        # 1.0 = Breakeven (Driver gets 100%)
        # 1.2 = Burning Cash (Subsidizing driver)
        self.action_space = np.linspace(0.4, 1.1, 15)

    def get_observation(self):
        active_drivers = [d for d in self.drivers if d.is_active]
        if not active_drivers:
            avg_progress = 0.0
        else:
            total_progress = sum(d.current_month_profit / d.monthly_profit_threshold for d in active_drivers)
            avg_progress = total_progress / len(active_drivers)

        return np.array([
            self.state.market_demand_multiplier,
            len(self.state.orders_queue),
            self.company.daily_profit / 10000.0, 
            avg_progress
        ])

    def reset(self):
        self.company = Company()
        self.drivers = [Driver(i) for i in range(self.config["driver_count"])]
        self.state = GlobalState()

        return np.array([
            self.state.market_demand_multiplier,
            0, # Backlog
            0, # Profit
            0  # Progress
        ], dtype=np.float32)

    def step(self, action_idx):
        self.day_counter += 1
        payout_ratio = self.action_space[action_idx]

        self.company.reset_daily_stats()
        self.state.current_epoch += 1
        
        # 3. Generate 
        new_orders_count = int(self.state.base_demand * self.state.market_demand_multiplier)
        
        # Random fluctuation
        new_orders_count = int(np.random.normal(new_orders_count, 5))
        new_orders_count = max(0, new_orders_count)
        
        for _ in range(new_orders_count):
            dist = np.random.uniform(1.0, 10.0)
            self.state.orders_queue.append(Order(len(self.state.orders_queue), dist))

        available_drivers = [d for d in self.drivers] 
        np.random.shuffle(available_drivers)
        
        orders_to_process = list(self.state.orders_queue)
        for order in orders_to_process:
            if not available_drivers:
                break
            
            offer_amount = order.customer_price * payout_ratio
            
            for _ in range(10):
                if not available_drivers: 
                    break
                
                driver = available_drivers.pop(0)
                
                if driver.evaluate_offer(order, offer_amount):
                    self.company.process_transaction(order.customer_price, offer_amount)
                    self.state.orders_queue.remove(order)
                    break 

        expired_count = 0
        active_queue = []
        for order in self.state.orders_queue:
            order.time_waiting += 1
            if order.time_waiting >= order.max_wait_time:
                expired_count += 1
            else:
                active_queue.append(order)
        
        self.state.orders_queue = active_queue
        self.company.daily_expired = expired_count

        if expired_count > 0:
            self.state.market_demand_multiplier -= (0.005 * expired_count)
        else:
            self.state.market_demand_multiplier += 0.01
            
        self.state.market_demand_multiplier = max(0.1, min(2.0, self.state.market_demand_multiplier))

        active_drivers_list = [d for d in self.drivers if d.is_active]
        if active_drivers_list:
            raw_progress = sum(d.current_month_profit / d.monthly_profit_threshold for d in active_drivers_list)
            avg_progress_signal = raw_progress / len(active_drivers_list)
        else:
            avg_progress_signal = 0.0

        drivers_quit_today = 0
        if self.day_counter % 30 == 0:
            for driver in self.drivers:
                if driver.is_active:
                    did_quit = driver.end_of_month_review()
                    if did_quit:
                        drivers_quit_today += 1

        # 1. Profit
        reward = self.company.daily_profit
        
        # 2. Backlog Penalty
        reward -= (self.company.daily_expired * 500)
        
        # 3. Churn Penalty (Huge, but rare)
        # This only fires every 30th step.
        if drivers_quit_today > 0:
            reward -= (drivers_quit_today * 5000)
            
        # 4. DONE Condition
        active_count = len([d for d in self.drivers if d.is_active])
        done = (self.company.balance < 0) or (active_count == 0)

        next_obs = np.array([
            self.state.market_demand_multiplier,
            len(self.state.orders_queue),
            self.company.daily_profit / 10000.0, 
            avg_progress_signal
        ], dtype=np.float32)
        
        return next_obs, reward, done, {}
