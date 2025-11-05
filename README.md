# Zuber-beats

This is a hobby project for experimenting with delivery optimization using RL. The goal is to simulate how an RL agent would work to increase profits for the company and the driver.

## Rules
These are some of the assumptions of the simulation.

### Riders
    - Riders quit if they have no profit or very less profit.

### Company
    - Company can't be loss making for more than 24 epochs
    - Company needs to grow or maintain driver fleet

## Environment
- We start the simulation with some seed money and seed drivers. 
- Each epoch is 1 month.
- Drivers get paid fixed amount per km
- 