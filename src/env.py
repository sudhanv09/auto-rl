import numpy as np
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

@dataclass
class Order:
    id: int
    distance_km: float
    customer_payment: float

class Driver:
    def __init__(self, driver_id: int, cost_per_km: float = 30.0):
        self.id = driver_id
        self.cost_per_km = cost_per_km
        self.kms_driven = 0.0
        self.monthly_expense = 0.0
        self.current_month_earnings = 0.0
        self.satisfaction = 1.0
        self.active = True

    def evaluate_offer(self, order: "Order", payout: float) -> bool:
        return self.active

    def complete_order(self, order: "Order", payout: float):
        self.kms_driven += order.distance_km
        self.current_month_earnings += payout

    def end_month_review(self) -> bool:
        """
        Updates satisfaction and decides whether to quit (churn).
        Returns True if driver quits.
        """
        self.monthly_expense = self.cost_per_km * self.kms_driven
        profit = self.current_month_earnings - self.monthly_expense


        if profit >= 0:
            self.satisfaction = min(1.0, self.satisfaction + 0.1)
        else:
            self.satisfaction = max(0.0, self.satisfaction - 0.2)

        self.kms_driven = 0.0
        self.monthly_expense = 0.0
        self.current_month_earnings = 0.0

        if self.satisfaction <= 0.0:
            self.active = False
            return True

        return False

class Company:
    def __init__(
        self,
        initial_funds: float = 50_000_000.0,
        order_markup_fee: float = 25.0,
    ):
        self.initial_funds = initial_funds
        self.order_markup_fee = order_markup_fee

        # State
        self.current_funds = initial_funds
        self.churned_drivers_history = 0
        self.successful_deliveries = 0

    def reset(self):
        self.current_funds = self.initial_funds
        self.churned_drivers_history = 0
        self.successful_deliveries = 0


    def update_funds(
        self,
        order: Order,
        payout: float,
    ) -> float:
        self.current_funds += order.customer_payment - payout

    def price_order(self, distance_km: float) -> float:
        _ = distance_km  # for future use
        return self.order_markup_fee

    def process_month_end(self, drivers: List[Driver]) -> int:
        churned_this_month = 0
        for driver in drivers:
            if driver.active:
                quitted = driver.end_month_review()
                if quitted:
                    churned_this_month += 1

        self.churned_drivers_history += churned_this_month
        return churned_this_month

    def is_bankrupt(self) -> bool:
        return self.current_funds <= 0.0

class Env:
    def __init__(self):
        # Config
        self.orders_per_day = 300
        self.days_per_month = 30

        # Company
        self.company = Company(
            initial_funds=50_000_000.0,
            order_markup_fee=25.0,
        )
        self.initial_num_drivers = 50
        self._init_drivers(num_drivers=self.initial_num_drivers)

        # State
        self.day = 1
        self.orders_processed_today = 0
        self.drivers: List[Driver] = []
        self.order_queue: List[Order] = []

        # Metrics for last simulated day
        self.last_profit = 0.0
        self.last_orders_completed = 0
        self.last_avg_satisfaction = 1.0
        self.last_new_signups = 0.0


    def _init_drivers(self, num_drivers):
        self.drivers = [Driver(i) for i in range(num_drivers)]

    def _generate_orders(self):
        self.order_queue = []
        for i in range(self.orders_per_day):
            dist = np.random.lognormal(mean=2.0, sigma=0.5) 
            price = self.company.price_order(dist)
            self.order_queue.append(
                Order(
                    id=i,
                    distance_km=dist,
                    customer_payment=price,
                )
            )

    def reset(self):
        self.company.reset()
        self.day = 1
        self._init_drivers(num_drivers=self.initial_num_drivers)
        self.last_profit = 0.0
        self.last_orders_completed = 0
        self.last_avg_satisfaction = 1.0
        self.last_new_signups = 0.0
        return self._get_observation()

    def _get_observation(self):
        if self.is_done():
            return np.zeros(7, dtype=np.float32)

        active_drivers = [d for d in self.drivers if d.active]
        active_count = len(active_drivers)

        funds_norm = self.company.current_funds / max(1.0, self.company.initial_funds)
        active_driver_ratio = active_count / max(1, self.initial_num_drivers)
        day_of_month_norm = (self.day % self.days_per_month) / self.days_per_month

        completed_frac = self.last_orders_completed / max(1, self.orders_per_day)
        last_profit_norm = self.last_profit / 1000.0  # scale so it's not huge
        last_new_signups_norm = self.last_new_signups / 10.0  # arbitrary scaling

        return np.array(
            [
                funds_norm,
                active_driver_ratio,
                day_of_month_norm,
                completed_frac,
                last_profit_norm,
                self.last_avg_satisfaction,
                last_new_signups_norm,
            ],
            dtype=np.float32,
        )

    def is_done(self):
        active_drivers = len([d for d in self.drivers if d.active])
        return self.company.is_bankrupt() or active_drivers == 0 or self.day > 365

    def step(self, action_payout: float):
        if self.is_done():
            return self._get_observation(), 0.0, True, {}

        self._generate_orders()

        prev_funds = self.company.current_funds
        orders_completed = 0

        for order in self.order_queue:
            active_drivers = [d for d in self.drivers if d.active]
            if not active_drivers:
                break  # no supply left

            driver = np.random.choice(active_drivers)

            accepted = driver.evaluate_offer(order, action_payout)
            if accepted:
                self.company.update_funds(order=order, payout=action_payout)
                driver.complete_order(order, action_payout)
                orders_completed += 1

        self.order_queue = []

        profit_today = self.company.current_funds - prev_funds

        active_drivers = [d for d in self.drivers if d.active]
        if active_drivers:
            avg_satisfaction = float(
                sum(d.satisfaction for d in active_drivers) / len(active_drivers)
            )
        else:
            avg_satisfaction = 0.0

        completed_frac = orders_completed / max(1, self.orders_per_day)

        new_signups = completed_frac * avg_satisfaction
        self.last_new_signups = new_signups

        w_profit = 1.0
        w_orders = 10.0
        w_satisfaction = 5.0
        w_signups = 2.0

        reward = (
            w_profit * profit_today
            + w_orders * orders_completed
            + w_satisfaction * avg_satisfaction
            + w_signups * new_signups
        )

        self.last_profit = profit_today
        self.last_orders_completed = orders_completed
        self.last_avg_satisfaction = avg_satisfaction

        self.day += 1
        if self.day % self.days_per_month == 0:
            _ = self.company.process_month_end(self.drivers)

        done = self.is_done()
        info = {
            "profit_today": profit_today,
            "orders_completed": orders_completed,
            "avg_satisfaction": avg_satisfaction,
            "new_signups": new_signups,
            "funds": self.company.current_funds,
            "day": self.day,
        }

        return self._get_observation(), reward, done, info