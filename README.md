# Auto-RL

## 1. Objective
The goal of this project is to train a Reinforcement Learning (RL) agent to act as the pricing algorithm for a food delivery platform. The agent must optimize the **Payout** offered to drivers for each order to achieve a sustainable balance between:
1.  **Company Health:** Maximizing profits and extending the runway of the initial seed funding.
2.  **Driver Health:** Maximizing driver satisfaction to minimize churn (drivers quitting).

The ultimate success metric is the company's survival duration and long-term profitability while maintaining a healthy fleet of drivers.

---

## 2. Environment Definition (`src/env.py`)

The environment simulates the daily operations of the delivery company.

### State Space (Observations)
The agent observes the current state of the ecosystem before making a decision.
- **Global Context:**
  - `current_funds`: Remaining money from seed funding.
  - `active_drivers`: Number of drivers currently in the pool.
  - `active_customers`: (Optional) Driver of order volume.
  - `date/day_of_month`: To track monthly review cycles.
- **Per-Order Context:**
  - `delivery_distance`: Randomly generated distance for the specific order.
  - `order_value`: (Optional) The price the customer pays (revenue source).

### Action Space
- **Agent Action:** `driver_payout`
  - A continuous value (or discretized set of values) representing the monetary amount offered to a driver for a specific order.

### Simulation Dynamics

#### A. Order Generation (Daily)
- Each day, $N$ orders are generated.
- Each order has a `delivery_distance` sampled from a distribution (e.g., log-normal).
- Revenue collected from the customer is fixed or distance-based (e.g., $Base + \alpha \times Distance$).

#### B. Driver Behavior (The "Environment" Logic)
Drivers are modeled agents (non-RL, rule-based or probabilistic) inside the environment.
1.  **Acceptance Decision:**
    - When presented with an order (Distance $D$, Payout $P$), a driver calculates a utility score (e.g., $Utility = P - Cost(D)$).
    - Driver accepts if $Utility > Threshold$.
2.  **Satisfaction & Churn (Monthly):**
    - Drivers track their monthly earnings.
    - At the end of the month, if `total_earnings` < `target_earnings`, the driver's satisfaction drops.
    - If satisfaction drops below a critical level, the driver **quits** (Churn).

#### C. Company Dynamics
- **Financials:**
  - `Funds(t+1) = Funds(t) + CustomerRevenue - DriverPayout - FixedCosts`.
  - **Churn Penalty:** When a driver quits, the company incurs a "Recruitment Cost" to replace them (or suffers from reduced capacity).
- **Growth:**
  - New customers/drivers may sign up based on platform reputation (e.g., % of fulfilled orders).

### Terminal State
The episode ends if:
1.  `current_funds` <= 0 (Bankruptcy).
2.  `active_drivers` <= 0 (Service collapse).
3.  Max time steps reached (Successful survival).

---

## 3. Agent Goals & Reward Function

The reward function must guide the agent away from greedy policies (paying 0 to maximize immediate profit) and over-generous policies (bankrupting the company).

**Proposed Reward ($R_t$):**

$$ R_t = w_1 \cdot \text{Profit}_t + w_2 \cdot \text{OrderFulfilled}_t - w_3 \cdot \text{DriverChurn}_t $$

- **Profit:** (Customer Revenue - Payout). Can be negative if subsidized.
- **OrderFulfilled:** Binary bonus if a driver accepts the order (incentivizes finding a market-clearing price).
- **DriverChurn:** Large penalty applied at the end of the month for every driver lost.

---

## 4. Implementation Roadmap

1.  **`src/env.py`**: Implement the `FoodDeliveryEnv` class inheriting from `gym.Env` (or similar interface).
    - `step(action)`: returns `next_state`, `reward`, `done`, `info`.
    - `reset()`: initializes seed funding and driver pool.
2.  **`src/driver.py`**: Implement `Driver` class with logic for accepting orders and tracking monthly satisfaction.
3.  **`src/main.py`**: Training loop using the RL agent (e.g., PPO or DQN) interacting with the environment.
```