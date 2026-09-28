# What We're Solving

## The Problem in 5 Points

- **The client** maintains jet engines and services every one of them on a fixed calendar, whatever condition it's in.
- **The problem** is that some engines break down before their service date, which is very expensive, while others get serviced while still healthy, which wastes money.
- **The idea** is to use the engines' sensor readings to estimate how many flights each engine has left before it fails.
- **The rule** is to service an engine only when its estimate gets low. That catches weak engines early and leaves healthy ones alone.
- **The goal** is to prove to the client that this costs less than their current schedule, and to show how confident we are in that saving.

## What We're Building

Today, engine sensors don't tell you how many flights are left before a failure. We don't add anything to the engine. We build software on top of the data the engines already produce.

- **Today:** the sensors only give raw readings like temperature, pressure and fan speed. Nothing tells you "this engine has 40 flights left."
- **What we build:** software that reads the sensor data the engines already produce and works out the flights left. No new hardware is needed.
- **How it learns:** we show it past engines that ran until they failed. It learns patterns like "when readings drift like this, failure is about 30 flights away."
- **Then:** it applies those patterns to engines that are still flying and gives each one an estimate.

## Analogy

It works like a doctor reading blood test results. The numbers don't say "you have X years left," but an experienced reader can estimate it from how the numbers change. We're building that experienced reader.

## Technical Solution in 5 Points

- **Data:** NASA C-MAPSS FD001 has 100 engines run from healthy until they fail, with 21 sensor readings on every flight (cycle).
- **Label:** for each row, flights left = the flight it failed on − the current flight, capped at 125. This is the remaining useful life (RUL) the model predicts.
- **Features and models:** drop sensors that never change, add each sensor's trend over the last 5, 10 and 30 flights, then train a simple straight-line baseline and LightGBM. Split train and test by engine and log every run in MLflow.
- **Cost rule:** service an engine when predicted RUL < *k*. Try many values of *k* and keep the cheapest (an unexpected failure is assumed to cost 10× a planned service). Compare against the fixed schedule and the one-sensor rule, with a bootstrap range on the savings.
- **Production (later):** a nightly job scores every engine and posts work orders to a FastAPI endpoint. We also check the model on FD003 to see if it still works when conditions change.
