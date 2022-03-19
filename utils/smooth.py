import numpy as np
import pandas as pd
import pmagpy.ipmag as ipmag

def running_mean_APWP(data,window_length,spacing,max_age,min_age,fig_name=None):
    """
    Code from Nick Swanson-Hysell
    """
    mean_pole_ages = np.arange(min_age,max_age+spacing,spacing)
    running_means = pd.DataFrame(columns=['Age','N','a95','RLon','RLat'])
    for age in mean_pole_ages:
        window_min = age - (window_length/2)
        window_max = age + (window_length/2)
        poles = data.loc[(data['Age'] >= window_min) & (data['Age'] <= window_max)]
        mean = ipmag.fisher_mean(dec=poles['RLon'].tolist(),inc=poles['RLat'].tolist())
        running_means.loc[age] = [age,mean['n'],mean['alpha95'],mean['dec'],mean['inc']]
    return running_means