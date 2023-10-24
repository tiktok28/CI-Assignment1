
import numpy as np
from skfuzzy import control as ctrl
from skfuzzy import membership as mf
import matplotlib.pyplot as plt

timeOfDay = ctrl.Antecedent(np.arange(0, 24, 0.1), 'timeOfDay')
peopleNearby = ctrl.Antecedent(np.arange(0, 100, 0.1), 'peopleNearby')
lightLevel = ctrl.Consequent(np.arange(0, 100, 0.1), 'lightLevel')

#Fit Vector for timeOfDay
timeOfDay['Daybreak'] = mf.trimf(timeOfDay.universe, [0, 0, 6.5])
timeOfDay['Morning'] = mf.trimf(timeOfDay.universe, [5, 8.5, 12])
timeOfDay['Noon'] = mf.trimf(timeOfDay.universe, [11, 14, 18.5])
timeOfDay['Evening'] = mf.trimf(timeOfDay.universe, [16, 19, 21])
timeOfDay['Night'] = mf.trapmf(timeOfDay.universe, [20, 22, 24, 24])

#Fit Vector for peopleNearby
peopleNearby['No'] = mf.trimf(peopleNearby.universe, [0, 0, 1])
peopleNearby['Few'] = mf.trimf(peopleNearby.universe, [1, 35, 45])
peopleNearby['Medium'] = mf.trimf(peopleNearby.universe, [30, 60, 75])
peopleNearby['A lot'] = mf.trimf(peopleNearby.universe, [70, 100, 100])

#Fit Vector for lightLevel
lightLevel['Off'] = mf.trimf(lightLevel.universe, [0, 0, 25])
lightLevel['Dim'] = mf.trimf(lightLevel.universe, [20, 50, 70])
lightLevel['Bright'] = mf.trimf(lightLevel.universe, [60, 100, 100])

#Rules
rule1 = ctrl.Rule(peopleNearby['No'] & timeOfDay['Daybreak'], (lightLevel['Off']))
rule2 = ctrl.Rule(peopleNearby['No'] & timeOfDay['Morning'], (lightLevel['Off']))
rule3 = ctrl.Rule(peopleNearby['No'] & timeOfDay['Noon'], (lightLevel['Off']))
rule4 = ctrl.Rule(peopleNearby['No'] & timeOfDay['Evening'], (lightLevel['Off']))
rule5 = ctrl.Rule(peopleNearby['No'] & timeOfDay['Night'], (lightLevel['Off']))

rule6 = ctrl.Rule(peopleNearby['Few'] & timeOfDay['Daybreak'], (lightLevel['Bright']))
rule7 = ctrl.Rule(peopleNearby['Few'] & timeOfDay['Morning'], (lightLevel['Dim']))
rule8 = ctrl.Rule(peopleNearby['Few'] & timeOfDay['Noon'], (lightLevel['Off']))
rule9 = ctrl.Rule(peopleNearby['Few'] & timeOfDay['Evening'], (lightLevel['Dim']))
rule10 = ctrl.Rule(peopleNearby['Few'] & timeOfDay['Night'], (lightLevel['Bright']))

rule11 = ctrl.Rule(peopleNearby['Medium'] & timeOfDay['Daybreak'], (lightLevel['Bright']))
rule12 = ctrl.Rule(peopleNearby['Medium'] & timeOfDay['Morning'], (lightLevel['Dim']))
rule13 = ctrl.Rule(peopleNearby['Medium'] & timeOfDay['Noon'], (lightLevel['Dim']))
rule14 = ctrl.Rule(peopleNearby['Medium'] & timeOfDay['Evening'], (lightLevel['Dim']))
rule15 = ctrl.Rule(peopleNearby['Medium'] & timeOfDay['Night'], (lightLevel['Bright']))

rule16 = ctrl.Rule(peopleNearby['A lot'] & timeOfDay['Daybreak'], (lightLevel['Bright']))
rule17 = ctrl.Rule(peopleNearby['A lot'] & timeOfDay['Morning'], (lightLevel['Bright']))
rule18 = ctrl.Rule(peopleNearby['A lot'] & timeOfDay['Noon'], (lightLevel['Dim']))
rule19 = ctrl.Rule(peopleNearby['A lot'] & timeOfDay['Evening'], (lightLevel['Bright']))
rule20 = ctrl.Rule(peopleNearby['A lot'] & timeOfDay['Night'], (lightLevel['Bright']))

rules = [rule1, rule2, rule3, rule4, rule5,
         rule6, rule7, rule8, rule9, rule10,
         rule11, rule12, rule13, rule14, rule15,
         rule16, rule17, rule18, rule19, rule20]

train_ctrl = ctrl.ControlSystem(rules=rules)
train = ctrl.ControlSystemSimulation(control_system=train_ctrl)

# define the values for the inputs
train.input['timeOfDay'] = 12
train.input['peopleNearby'] = 100

# compute the outputs
train.compute()

# print the output values
print(train.output)

# to extract one of the outputs
print(train.output['lightLevel'])


lightLevel.view(sim=train)

#Control/Output space
x, y = np.meshgrid(np.linspace(timeOfDay.universe.min(), timeOfDay.universe.max(), 100),
                   np.linspace(peopleNearby.universe.min(), peopleNearby.universe.max(), 100))
z_lightLevel = np.zeros_like(x, dtype=float)

for i,r in enumerate(x):
  for j,c in enumerate(r):
    train.input['timeOfDay'] = x[i,j]
    train.input['peopleNearby'] = y[i,j]
    try:
      train.compute()
    except:
      z_lightLevel[i,j] = float('inf')
    z_lightLevel[i,j] = train.output['lightLevel']

def plot3d(x,y,z):
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')

  ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis', linewidth=0.4, antialiased=True)

  ax.contourf(x, y, z, zdir='z', offset=-2.5, cmap='viridis', alpha=0.5)
  ax.contourf(x, y, z, zdir='x', offset=x.max()*1.5, cmap='viridis', alpha=0.5)
  ax.contourf(x, y, z, zdir='y', offset=y.max()*1.5, cmap='viridis', alpha=0.5)

  ax.view_init(30, 200)

plot3d(x, y, z_lightLevel)

plt.show()
