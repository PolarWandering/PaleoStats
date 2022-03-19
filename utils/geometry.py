import numpy as np
from scipy.spatial.transform import Rotation as rotation

def spherical2cartesian(v):
	"""
	v[0] = theta - Latitude
	v[1] = phi - Longitude
	"""
    
	x = np.cos(v[0]) * np.cos(v[1])  
	y = np.cos(v[0]) * np.sin(v[1])  
	z = np.sin(v[0])  
    
	return [x,y,z]

def cartesian2spherical(v):  
	"""
	Take an array of lenght 3 correspoingt to a 3-dimensional vector and returns a array of lenght 2
	with latitade (inclination) and longitude (declination)
	"""
	theta = np.arcsin(v[2]) 
	phi = np.arctan2(v[1], v[0])
        
	return [theta, phi]