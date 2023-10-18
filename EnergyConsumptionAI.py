print("I gae")

import numpy as np
from skfuzzy import control as ctrl
from skfuzzy import membership as mf

indoor_light_level = ctrl.Antecedent(np.arange(0, 10, 0.1), 'indoor_light_level')
time_of_day = ctrl.Antecedent(np.arange(0, 10, 0.1), 'time_of_day')
people_nearby = ctrl.Antecedent(np.arange(0, 10, 0.1), 'people_nearby')
light_output = ctrl.Consequent(np.arange(0, 30, 0.1), 'light_output')
