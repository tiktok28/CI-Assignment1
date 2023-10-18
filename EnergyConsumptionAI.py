print("I gae")

import numpy as np
from skfuzzy import control as ctrl
from skfuzzy import membership as mf

indoor_light_level = ctrl.Antecedent(np.arange(0, 10, 0.1), 'indoor_light_level')
time_of_day = ctrl.Antecedent(np.arange(0, 10, 0.1), 'time_of_day')
people_nearby = ctrl.Antecedent(np.arange(0, 10, 0.1), 'people_nearby')
light_output = ctrl.Consequent(np.arange(0, 30, 0.1), 'light_output')

indoor_light_level['very low'] = mf.trimf(indoor_light_level.universe, [0, 0, 5])
indoor_light_level['low'] = mf.trimf(indoor_light_level.universe, [0, 5, 10])
indoor_light_level['medium'] = mf.trimf(indoor_light_level.universe, [5, 10, 10])
indoor_light_level['high'] = mf.trimf(indoor_light_level.universe, [0, 5, 10])
indoor_light_level['super high'] = mf.trimf(indoor_light_level.universe, [5, 10, 10])

time_of_day['poor'] = mf.trimf(time_of_day.universe, [0, 0, 5])
time_of_day['average'] = mf.trimf(time_of_day.universe, [0, 5, 10])
time_of_day['good'] = mf.trimf(time_of_day.universe, [5, 10, 10])
time_of_day['average'] = mf.trimf(time_of_day.universe, [0, 5, 10])
time_of_day['good'] = mf.trimf(time_of_day.universe, [5, 10, 10])

tips['low'] = mf.trimf(tips.universe, [0, 0, 15])
tips['medium'] = mf.trimf(tips.universe, [0, 15, 30])
tips['high'] = mf.trimf(tips.universe, [15, 30, 30])

service.view()
food.view()
tips.view()

rule1 = ctrl.Rule(service['poor']    & food['poor']   , tips['low'])
rule2 = ctrl.Rule(service['average'] & food['poor']   , tips['low'])
rule3 = ctrl.Rule(service['good']    & food['poor']   , tips['medium'])
rule4 = ctrl.Rule(service['poor']    & food['average'], tips['low'])
rule5 = ctrl.Rule(service['average'] & food['average'], tips['medium'])
rule6 = ctrl.Rule(service['good']    & food['average'], tips['high'])
rule7 = ctrl.Rule(service['poor']    & food['good']   , tips['medium'])
rule8 = ctrl.Rule(service['average'] & food['good']   , tips['high'])
rule9 = ctrl.Rule(service['good']    & food['good']   , tips['high'])

rules = [rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9]

tipping_ctrl = ctrl.ControlSystem(rules=rules)
tipping = ctrl.ControlSystemSimulation(control_system=tipping_ctrl)

x, y = np.meshgrid(np.linspace(service.universe.min(), service.universe.max(), 100),
                       np.linspace(food.universe.min(), food.universe.max(), 100))
z = np.zeros_like(x, dtype=float)

for i,r in enumerate(x):
  for j,c in enumerate(r):
    tipping.input['service'] = x[i,j]
    tipping.input['food'] = y[i,j]
    try:
      tipping.compute()
    except:
      z[i,j] = float('inf')
    z[i,j] = tipping.output['tips']

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot3d(x,y,z):
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')

  ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis', linewidth=0.4, antialiased=True)

  ax.contourf(x, y, z, zdir='z', offset=-2.5, cmap='viridis', alpha=0.5)
  ax.contourf(x, y, z, zdir='x', offset=x.max()*1.5, cmap='viridis', alpha=0.5)
  ax.contourf(x, y, z, zdir='y', offset=y.max()*1.5, cmap='viridis', alpha=0.5)

  ax.view_init(30, 200)
  
  return ax

ax = plot3d(x, y, z)

plt.show()
