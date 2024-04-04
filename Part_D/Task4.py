import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from skfuzzy import control as ctrl

# Define the universe of the inputs
water_flow = ctrl.Consequent(np.arange(0, 101, 1), 'water_flow')
error_universe = np.arange(-5, 5.1, 0.1)
change_error_universe = np.arange(-1, 1.1, 0.1)

# Define the membership functions for the error
error_cold = fuzz.trimf(error_universe, [-5, -3, 0])
error_perfect = fuzz.trimf(error_universe, [-2, 0, 2])
error_hot = fuzz.trimf(error_universe, [0, 3, 5])

# Define the membership functions for the change in error
change_error_cold = fuzz.trimf(change_error_universe, [-1, -0.5, 0])
change_error_nochange = fuzz.trimf(change_error_universe, [-0.25, 0, 0.25])
change_error_hot = fuzz.trimf(change_error_universe, [0, 0.5, 1])

# Define the fuzzy controller
controller = ctrl.ControlSystem([
    # Define the input variables
    ('error', ctrl.Antecedent(error_universe, 'error')),
    ('change_error', ctrl.Antecedent(change_error_universe, 'change_error')),

    # Define the output variable
    ('water_flow', ctrl.Consequent(np.arange(0, 101, 1), 'water_flow')),

    # Define the rules
    [ctrl.Rule(error_cold, change_error_cold, water_flow['low']),
     ctrl.Rule(error_cold, change_error_nochange, water_flow['low']),
     ctrl.Rule(error_cold, change_error_hot, water_flow['low']),

     ctrl.Rule(error_perfect, change_error_cold, water_flow['low']),
     ctrl.Rule(error_perfect, change_error_nochange, water_flow['medium']),
     ctrl.Rule(error_perfect, change_error_hot, water_flow['medium']),

     ctrl.Rule(error_hot, change_error_cold, water_flow['medium']),
     ctrl.Rule(error_hot, change_error_nochange, water_flow['medium']),
     ctrl.Rule(error_hot, change_error_hot, water_flow['high'])],

    # Define the defuzzification method
    ('water_flow', ctrl.Defuzzifier.center_of_mass),
])

# Simulate the fuzzy controller
simulation = ctrl.ControlSystemSimulation(controller)

# Set the input values
simulation.input['error'] = 2
simulation.input['change_error'] = 0.5

# Run the simulation
simulation.compute()

# Get the output value
output = simulation.output['water_flow']

# Print the output value
print(f"Water flow: {output}%")