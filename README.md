# Eco-Loop Building Agents

## Overview

Buildings waste a lot of energy because their heating and cooling systems usually run on fixed schedules. They don't know if the building is empty, if the weather has changed, or if electricity is cleaner or cheaper at a particular time.

I built an AI agent that can monitor a building and make smart decisions in real time.

I use **EnergyPlus** to simulate a building and a local **Qwen2.5 LLM** as the decision-making engine. The AI continuously watches the building's conditions, decides the best energy-saving strategy, and updates the building controls automatically without stopping the simulation.

---

## Project Status

| Phase | Description | Status |
|------|-------------|------------------|
| 1 | EnergyPlus runtime integration | **done, verified** |
| 2 | Closed-loop controller | **done, verified** |
| 3 | MCP server exposing tools | **done, verified** |
| 4 | Async LLM policy generation | **done, verified** |
| 5 | Dashboard | **done, verified** |

---

## How It Works

The system follows a continuous feedback loop.

1. EnergyPlus simulates the building and provides live data such as:
   - Zone temperatures
   - Energy consumption
   - Thermal comfort
   - Indoor conditions

2. Every 15 minutes, the AI collects the latest building information.

3. Every 2 hours, the collected data is sent to the local **Qwen2.5** model through the MCP server.

4. The LLM analyzes:
   - Building comfort
   - Current energy usage
   - Weather conditions
   - Grid carbon intensity

5. Based on this information, it decides the best control strategy, such as:
   - Increase or decrease AC setpoints
   - Pre-cool the building before peak hours
   - Reduce unnecessary cooling when occupancy is low

6. These new control values are directly injected into the running EnergyPlus simulation using the **Python Runtime API**, allowing the simulation to continue without restarting.

This creates a complete **closed-loop control system** where the AI continuously observes, thinks, acts, and repeats.

---

## Safety

Energy savings should never come at the cost of occupant comfort.

I added a safety layer that validates every decision made by the LLM. Any temperature or control value outside the allowed range is automatically corrected before being applied.

This guarantees that the building always remains within acceptable comfort limits.

---

## Features

- Real-time building monitoring
- Autonomous AI decision making
- Local Qwen2.5 LLM
- MCP Server for tool calling
- Asynchronous policy generation
- EnergyPlus Runtime API integration
- Closed-loop control
- Safety constraints for occupant comfort
- Automatic control updates without restarting the simulation
- Live dashboard for monitoring and analytics

---

## Tech Stack

- Python
- EnergyPlus
- Qwen2.5
- MCP Server
- Python Runtime API
- AsyncIO
- Pandas
- NumPy
- Matplotlib
- Streamlit

---

## Results

After running the system on a **15-zone office building** for **14 simulated days**, I achieved:

- **1.1% reduction in total energy consumption**
- **1.7% reduction in carbon emissions**
- **Occupant comfort maintained throughout the simulation**

The AI successfully monitored the simulation, generated policies asynchronously through the MCP server, updated building controls in real time, and maintained a stable closed-loop control pipeline throughout the experiment.

---

## Conclusion

This project demonstrates how an AI agent can manage a building more efficiently than traditional rule-based systems. By combining **EnergyPlus**, **Qwen2.5**, **MCP**, and **real-time control**, I created a smart building that continuously optimizes energy usage while keeping occupants comfortable.