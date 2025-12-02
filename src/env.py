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
