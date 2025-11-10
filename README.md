# AutoRL

This is a hobby project for experimenting with delivery optimization using RL. The goal is to simulate how an RL agent would work to increase profits for the company and the driver.

## Environment

### State space

    Company balance
    Cumulative Profit/Loss
    Consecutive loss making epoch
    Active drivers
    Driver churn rate
    Driver signups
    Avg driver monthly earning
    Avg driver montly profit
    Monthly Orders generated
    Monthly Orders completed
    Completion Rate
    Avg delivery distance

### Rules
These are the rules that govern the simulation from one epoch to the next.

 - Individual Driver Simulation:

        Each driver is an individual entity in the simulation with their own attributes:

            patience_level: A hidden variable representing their tolerance for low profit.

            monthly_profit: Calculated as (total_kms_driven * payment_rate) - (total_kms_driven * cost_per_km).

            cost_per_km: A randomized value for each driver to simulate different vehicle efficiencies (e.g., $0.25 - $0.40 per km).

 - Driver Churn (Quitting) Model:

        At the end of an epoch, each driver evaluates their monthly_profit. If monthly_profit is below a dynamic "expectation threshold" (e.g., $400/month), their probability of quitting increases significantly. This threshold can slowly increase over time, simulating rising expectations. The churn probability is also influenced by their patience_level.

 - Driver Acquisition Model:
        
        The number of new drivers joining each month is a function of the average_driver_monthly_earnings from the previous epoch. Higher potential earnings will attract more new drivers to the platform.

 - Customer Demand Model:
        
        This is a crucial dynamic. Demand is not fixed; it is influenced by the platform's reliability.
        monthly_orders_generated = base_demand + (active_drivers * driver_to_customer_multiplier)
        base_demand: A starting number of orders (e.g., 1000).
        driver_to_customer_multiplier: A factor representing the network effect (e.g., each active driver brings in the potential for 50 additional monthly orders). This models the idea that better driver availability leads to faster delivery times and attracts more customers.

        However, if the completion_rate of orders drops below 90% for two consecutive epochs, the base_demand will start to shrink by 5% each month, simulating customer loss due to poor service.

- Order Generation & Assignment:
       
       At the start of an epoch, a pool of monthly_orders_generated is created. Each order has a randomly assigned location and delivery distance. These orders are distributed among the active_drivers throughout the simulated month. If there are not enough drivers to complete all orders, some will fail.

### Agent

The agent's role is to learn the best payment strategy to navigate the complex trade-offs of the environment.

1. Action Space (What the Agent Decides)

At the start of each epoch, the agent must decide on the payment structure. A sophisticated action space could be:

    Action Type: A choice between different payment models:

        Fixed Rate: Pay a flat rate per kilometer (e.g., $1.20/km).

        Tiered Rate: Pay a higher rate for longer distances (e.g., $1.00/km for the first 5km, $1.50/km thereafter).

        Bonus Structure: Pay a fixed rate plus a bonus for completing a certain number of deliveries (e.g., $1.00/km + a $200 bonus for completing 100 deliveries).

    Action Value(s): The specific monetary values for the chosen model (e.g., for "Fixed Rate", the agent chooses the value "$1.20"). Reinforcement learning is well-suited for such dynamic pricing problems.[6][7][8][9][10]

2. Reward Function (How the Agent is Guided)

This function is designed to directly reflect your stated goals. At the end of each epoch, the agent receives a reward calculated as follows:

Reward = (Company_Profit_Score) + (Driver_Satisfaction_Score) - (Penalty_Score)

    Company Profit Score:

        +10 if the company is profitable this epoch.

        -10 if the company is loss-making.

        An additional -20 if this is the 3rd consecutive loss-making epoch (to punish consistent losses).

    Driver Satisfaction Score:

        +15 if the average_driver_monthly_profit is above the "expectation threshold".

        Calculated proportionally to (1 - driver_churn_rate). A 0% churn rate gives a full +10 points, while a 20% churn rate would give 10 * (1 - 0.2) = +8 points. This provides a dense reward signal.

    Penalty Score:

        A massive -50 if driver_churn_rate is above 25% (a "mass exodus" event).

        -30 if the completion_rate of deliveries falls below 90%.

This composite reward structure forces the agent to learn a balanced policy. It cannot simply maximize company profit by underpaying drivers, as that would trigger massive churn penalties. Conversely, it cannot overpay drivers to the point of bankrupting the company.
