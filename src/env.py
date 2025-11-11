from state import GlobalState, Company, Driver
import numpy as np

class Env:
    def __init__(self, config: dict):
        self.config = config

        self.company = Company(config['driver_count'], config['seed_money'])
        self.drivers = [Driver(i) for i in range(config['driver_count'])]
        self.state = GlobalState()

    def get_observation(self):
        state = np.array([
            self.company.current_balance,
            self.company.last_epoch_profit,
            self.company.consecutive_loss_epochs,
            self.company.active_drivers,
            self.company.driver_churn_rate,
            self.company.new_signups,
            self.company.avg_driver_earnings,
            self.company.avg_driver_profit,
            self.company.order_completion_rate
        ], dtype=np.float32)
        return state
    
    def reset(self):
        self.company = Company(self.config['initial_driver_count'], self.config['seed_money'])
        self.drivers = [Driver(i) for i in range(self.config['initial_driver_count'])]
        self.state = GlobalState()
        
        # Generate the initial state for the agent
        self.company.epoch_report(self.drivers, self.state)
        return self.get_observation()

    def step(self, policy):
        DAYS_PER_MONTH = 30
        self.state.current_epoch += 1
        self.state.total_orders_available = 0
        self.state.total_orders_complete = 0

        for driver in self.drivers:
            driver.reset()

        for day in range(DAYS_PER_MONTH):
            base_demand = self.config["base_demand"]
            multiplier = self.config["multiplier"]
            daily_orders_generated = int(
                (base_demand + (len(self.drivers) * multiplier)) / DAYS_PER_MONTH
            )

            if (
                self.company.consecutive_loss_epochs >= 2
                or self.company.order_completion_rate < 0.9
            ):
                daily_orders_generated = int(daily_orders_generated * 0.95)
            
            orders = [
                {"distance": np.random.uniform(0.2, 12.0)}
                for _ in range(daily_orders_generated)
            ]

            completed = 0
            for order in orders:
                if len(self.drivers) == 0:
                    # company has no active drivers — simulate "bankruptcy" or reset scenario
                    done = True
                    reward = -100  # massive penalty for losing all drivers
                    next_obs = self.get_observation()
                    return next_obs, reward, done, {}

                driver = np.random.choice(self.drivers)
                if driver.is_active and driver.accept_order(order, policy):
                    payout = order["distance"] * policy["rate_per_km"]
                    order_fee = order["distance"] * self.config["avg_order_fee_per_km"]

                    self.company.process_payment_to_driver(payout)
                    self.company.record_revenue_from_order(order_fee)
                    self.state.total_orders_complete += 1
                    completed += 1

            # print(f"Day {day}: Orders generated={len(orders)}, Completed={completed}")
            self.state.total_orders_available += daily_orders_generated

        self.company.finalize_epoch_accounting()
        self.company.epoch_report(self.drivers, self.state)

        for driver in self.drivers:
            driver.update_sentiment()

        self.drivers = self.company.update_driver_fleet(self.drivers)

        reward = self.calculate_reward()
        next_obs = self.get_observation()
        done = self.company.current_balance <= 0

        return next_obs, reward, done, {}
    
    def calculate_reward(self):
        company = self.company
        reward = 0

        # Profitability
        if company.last_epoch_profit > 0:
            reward += 10
        else:
            reward -= 10

        if company.consecutive_loss_epochs >= 3:
            reward -= 20

        # Driver sentiment
        expectation_threshold = self.config.get("profit_expectation_threshold", 35_000)
        if company.avg_driver_profit > expectation_threshold:
            reward += 15

        # Churn and reliability
        reward += 10 * (1 - company.driver_churn_rate)
        if company.driver_churn_rate > 0.25:
            reward -= 50
        if company.order_completion_rate < 0.9:
            reward -= 30

        return reward