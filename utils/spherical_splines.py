"""
Implementation of the method introduced by Jupp & Kent in Fitting Smooth
Paths to Spherical Data (1987). 
"""

import pandas as pd
import numpy as np
import os
import cartopy.crs as ccrs
import pmagpy.ipmag as ipmag
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.spatial.transform import Rotation as rotation
from scipy.optimize import minimize
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import CubicSpline

from csaps import csaps

#######################################################################
########################### Utils functions ###########################
#######################################################################


def cart2sph(X, radians=True):
    '''
    X : (N,3) numpy array
    '''
    Y = np.zeros((X.shape[0], 2))
    Y[:,0] = np.arctan2(X[:,1], X[:,0])   # Longitude
    Y[:,1] = np.arcsin(X[:,2])            # Latitude

    # We transforn longitude to span from 0 to 2*pi
    Y[:,0] = np.where(Y[:,0] < 0.0, Y[:,0] + 2*np.pi, Y[:,0])
    
    if not radians:
        Y *= 180. / np.pi
    
    return Y


def sph2cart(v):
    """
    v[0] = theta - Latitude
    v[1] = phi - Longitude
    """
    
    x = np.cos(v[0]) * np.cos(v[1])  
    y = np.cos(v[0]) * np.sin(v[1])  
    z = np.sin(v[0])  
    
    return [x,y,z] / np.linalg.norm([x,y,z])


def slerp_interpolation(p1, p2, t, order, tmin=None, tmax=None):
    '''
    Returns the Slerp interpolation between two unitary vectors on the 3D sphere following a great circle
    '''
    angle = np.arccos(np.dot(p1,p2))

    if order == 0:
        return ( np.sin((1-t)*angle) * p1 + np.sin(t*angle) * p2 ) / np.sin(angle)
    elif order == 1:
        return angle * ( np.cos(t*angle) * p2 - np.cos((1-t)*angle) * p1 ) / (np.sin(angle) * (tmax-tmin)) 
    else:
        raise ValueError('Order derivative not implemented.')
        

def skew_matrix(v):
    skv = np.roll(np.roll(np.diag(v.flatten()), 1, 1), -1, 0)
    return skv - skv.T


def equal_distance_projection(X):
    
    Y = cart2sph(X, radians=True)
    Z = np.zeros(X.shape)
            
    Z[:,0] = (np.pi/2-Y[:,1]) * np.cos(Y[:,0])
    Z[:,1] = (np.pi/2-Y[:,1]) * np.sin(Y[:,0])
    
    return Z

        
#######################################################################
###################### Curve object definition ########################
#######################################################################

class S2Curve:
    '''
    Curve clase we are going to fit in the sphere.
    This will be consist in a series of timesteps where we can exactly evaluate the curve. For all the intermediate steps, we use the 
    great circle between adjacent knots to evalaute the function 
    '''
    
    def __init__(self, time_values, knot_values):
        
        self.time_values = time_values   # (N,)
        self.knot_values = knot_values   # (N,3)
        self.planar_values = None        # (N,3) last component is zero
        self.Rotations = None            # (N,3,3)
    
    
    def evaluate(self, t, order=0):
        """
        Return evaluation of the curve
        
        Parameters:
        - order: order correspondint to the evaluation of the derivative. If equals to 0 this is just the evaluation of the function. 
        
        """
        assert (np.min(self.time_values) <= t) & (t <= np.max(self.time_values)), 'Age value is not inside the evaluation window.'
        idx_inf = np.max(np.where(self.time_values - t <= 0.0))
        idx_sup = np.min(np.where(self.time_values - t >= 0.0))

        if idx_inf == idx_sup and order == 0:
            #print('Evaluating self node.')
            return self.knot_values[idx_inf, :]
        elif idx_inf == idx_sup and order > 0:
            idx_sup += 1
        assert idx_inf + 1 == idx_sup
        
        time_inf = self.time_values[idx_inf]
        time_sup = self.time_values[idx_sup]
        incremental = (t - time_inf) / (time_sup - time_inf)
        assert 0 <= incremental <= 1
        
        knot_inf = self.knot_values[idx_inf, :]
        knot_sup = self.knot_values[idx_sup, :]

        return slerp_interpolation(knot_inf, knot_sup, incremental, order=order, tmin=time_inf, tmax=time_sup)

    def update_rotation(self, R):
        self.Rotations = R
        
    def update_unroll(self, unrolled):
        self.planar_values = unrolled
    
    def plot(self):
        plt.figure(figsize=(15,8))
        ax1 = plt.subplot2grid((2,2), (0,0), rowspan=2, projection=ccrs.Orthographic(central_latitude=90))
        ax2 = plt.subplot2grid((2,2), (0,1))
        ax3 = plt.subplot2grid((2,2), (1,1))
        
        ax1.gridlines()
        
        knot_values_sph = cart2sph(self.knot_values, radians=False) 
        
        sns.scatterplot(ax=ax1, x = knot_values_sph[:,0], y = knot_values_sph[:,1], 
                        hue = self.time_values,
                        transform = ccrs.PlateCarree())
        
        ax2.plot(self.time_values,  knot_values_sph[:,0], 'o-', ms=1, c='red')
        ax3.plot(self.time_values,  knot_values_sph[:,1], 'o-', ms=1)
        
        
#######################################################################
######################  Rolling and Unrolling  ########################
#######################################################################


def unroll_curve(curve, delta_t=0.1):
    '''
    Unroll one single curve f to f* following the Appendix in Jupp et. al. (1987)
    
    Given a Curve() object, we return the 
    '''    
    N = curve.time_values.shape[0]
    
    f_star = np.zeros((N,3))    # We leave the third coordinate always equal to zero for simplicity
    Rotation = np.zeros((N,3,3))
    
    # Initial Conditions
    # We need to find a rotation R0 such that it satisfies the conditions R0 * f(0) = (0,0,1) and R0 * f'(0) = (c,0,0)
    # First we find a rotation R1 such that R1 * f(0) = (0,0,1)
    
    t0 = curve.time_values[0]
    f0 = curve.evaluate(t0, order=0)
    
    if np.linalg.norm(f0 - [0,0,1]) < 0.0001:
        R1 = rotation.from_rotvec([0,0,0])
    else:
        angle = np.arccos(np.dot(f0, [0,0,1]))
        rotation_axis = np.cross(f0, [0,0,1])
        rotation_axis /= np.linalg.norm(rotation_axis)
        R1 = rotation.from_rotvec(angle * rotation_axis)
    
    # and now we define a second rotation with axis (0,0,1) that rotates f'(0) to the x-axis  
    f_derivative = curve.evaluate(t0, order=1)
    f_derivative_rotated = R1.apply(f_derivative)
    f_derivative_rotated_normalized = f_derivative_rotated / np.linalg.norm(f_derivative_rotated)
    
    assert np.abs(f_derivative_rotated_normalized[2]) < 0.0001, 'After rotation, the gradient is not paralel to (0,0,1). Instead it gives {}'.format(f_derivative_rotated_normalized)
    
    
    angle = np.arctan2(f_derivative_rotated_normalized[1], f_derivative_rotated_normalized[0])
    R2 = rotation.from_rotvec(-angle * np.array([0,0,1]))
    
    # Composition of rotations
    R0 = R2 * R1
    
    assert np.linalg.norm(R0.apply(f0) - [0,0,1]) < 0.001
    f_derivative_normalized = f_derivative / np.linalg.norm(f_derivative)
    assert np.linalg.norm(R0.apply(f_derivative_normalized) - [1,0,0]) < 0.001, 'The obtained initial rotation R(0) does not satisfy R(0)df(0)=[c,0,0]. Instead, the rotation gives {}'.format(R.apply(f_derivative_rotated_normalized))
    
    # This doesnt work
    # f_star[0,:] = [0,0,0]
    # Rotation[0,:,:] = R0.as_matrix()
    # f_ = f_star[0,:]
    # R = Rotation[0,:,:]
    
    f_ = [0,0,0]
    R = R0.as_matrix()
    f_star[0,:] = f_
    Rotation[0,:,:] = R
    
    # Iterative steps
    #print('f_star: ', f_star[0:2,:])
    
    for idx in range(len(curve.time_values)-1):
    
        t = curve.time_values[idx]
        next_t = curve.time_values[idx+1]
        
        while t < next_t:
            
            dtt = min(delta_t, next_t - t)
            
            f = curve.evaluate(t, order=0)
            df = curve.evaluate(t, order=1)
            f_ += dtt * np.dot(R, df)
            f_[-1] = 0  # we force the last component of f* to be zero 
            R += dtt * np.dot(R, skew_matrix(np.cross(df,f)))
            t += dtt

        f_star[idx+1,:] = f_ 
        assert np.abs(f_star[idx+1,2]) < 0.01, 'Reduce the solver time step to ensure the solution is correct. The z-component of f^* gives {} instead of 0.0'.format(f_star[idx+1,2])
        Rotation[idx+1,:,:] = R
        
    #print('f_star: ', f_star[0:2,:])
        
    return curve.time_values, f_star, Rotation


def roll_curve(times, X, delta_t=0.1):
    '''
    Arguments:
        - time: (N,) 
        - X: (N, 3) numpy array
    '''
    
    N = X.shape[0]
    
    F = np.zeros((N,3))   
    Rotation = np.zeros((N,3,3))

    # Derivative of the projected path
    Df_star = np.diff(X, axis=0) / np.diff(times)[:, np.newaxis]

    # initial conditions
    f = [0,0,1]
    df = [1,0,0] 
    
    # Initial condition for the rotation needs to be pick such that f*'(0) = R(0) f'(0) 
    #R = np.eye(3)
    df_star = Df_star[0,:]
    df_star /= np.linalg.norm(df_star)
    angle = np.arctan2(df_star[1], df_star[0])
    R0 = rotation.from_rotvec(angle * np.array([0,0,1])) 
    R = R0.as_matrix()
    assert np.linalg.norm(df_star - np.dot(R, df)) < 0.001, "After rotation we obtain {} instead of {}".format(np.dot(R, df), df_star)
    
    F[0,:] = f   # The initial potition of the rolling can be chosen arbitratrarialy
    Rotation[0,:,:] = R
    
    # We compute the derivative of f*
    
    for idx in range(len(times)-1):
        
        t = times[idx]
        next_t = times[idx+1]
        
        f_star = X[idx,:,]
        df_star = Df_star[idx,:]
        
        while t < next_t:
            
            dtt = min(delta_t, next_t - t)
            
            f += dtt * np.dot(R.T, df_star)
            # Renormalize for numerical error
            f /= np.linalg.norm(f)
            R += dtt * np.dot(skew_matrix(np.cross(df_star, [0,0,1])), R)
            t += dtt

        F[idx+1,:] = f
        Rotation[idx+1,:,:] = R

    return  S2Curve(time_values=times,
                    knot_values=F)


def unroll_points(curve, point_times, D):
    '''
    We unroll the points D using a given curve.
    
    Arguments:
        - curve: Curve() object
        - D: (M,3)
        - point_times: (M,)
    '''
    
    D_star = np.zeros(D.shape)
    
    for idx, t in enumerate(point_times):
        # First we identify the right time 
        idx_inf = np.max(np.where(curve.time_values - t <= 0.0))
        # Evaluate rotation there 
        R = curve.Rotations[idx_inf,:,:]
        # Apply rotation to point
        d_rotated = np.dot(R, D[idx,:])
        # Since we have numerical errors, we need to renormalize the vectors on the sphere
        d_rotated /= np.linalg.norm(d_rotated)
        d_rotated = d_rotated[np.newaxis,:]
        # equal distance projection
        d_projected = equal_distance_projection(d_rotated)
        D_star[idx,:] = curve.planar_values[idx_inf,:] + d_projected

    return D_star


def roll_points(curve, point_times, D):
    '''
    Arguments
        - point_times: (M,)
        - D: (M,3) with last component equal to zero
    '''
    
    K = np.zeros((len(point_times), 3))
    
    for idx, t in enumerate(point_times):
        # First we identify the right time 
        idx_inf = np.max(np.where(curve.time_values - t <= 0.0))
        # Evaluate rotation there and unrolled path there
        R = curve.Rotations[idx_inf,:,:]
        f = curve.knot_values[idx_inf,:]
        f_star = curve.planar_values[idx_inf,:]
        # We append and extra zero in the z-component to the planar valies
        v = D[idx,:]
        angle = np.linalg.norm(v-f_star)
        if angle < 0.0001:
            K[idx,:] = f 
        tangent = np.dot(R.T, v-f_star)
        rotation_axis = np.cross(f, tangent)
        rotation_axis /= np.linalg.norm(rotation_axis)
        R0 = rotation.from_rotvec(angle * rotation_axis)
        K[idx,:] = R0.apply(f)
        #print(f, f_star, v)
    return K
        
        
#######################################################################
######################  Computation of splines  #######################
#######################################################################


def spherical_spline(times, knot_values, smoothing, precision=0.1, ode_stepsize=0.01, n_iter=2, tol=0.001):
    
    # We first define a curve with the values we have in the dataset
    curve_original = S2Curve(time_values=times,
                             knot_values=knot_values)
    
    # and then we use it to construct a more finer curve
    #time_steps = np.unique(np.sort(np.concatenate([np.arange(0, df.Time.max(), params.delta), df.Time.values])))
    time_steps = np.arange(np.min(times), np.max(times), precision)
    knot_steps = np.array([curve_original.evaluate(t,0) for t in time_steps])

    curve = S2Curve(time_values=time_steps,
                    knot_values=knot_steps)
    all_curves = {}

    for idx_iter in range(n_iter):
        
        ### Unroll curve ###
        time, f_star, Rotation = unroll_curve(curve, delta_t=ode_stepsize)

        # Update rotation and f* from unrolling
        curve.update_rotation(Rotation)
        curve.update_unroll(f_star)

        ### Unroll points ###
        D_star = unroll_points(curve, times, knot_values)
        
        ### Fit Splines ###

        # Weight first node if we want to reinforce origin of coordinates to be fixed
        weights = np.ones(len(times))
        if times[0] < 0.001:
            weights[0] = 100
            
        X_star = csaps(times, D_star[:,0], curve.time_values, weights=weights, smooth=smoothing)
        Y_star = csaps(times, D_star[:,1], curve.time_values, weights=weights, smooth=smoothing)
        
        K = roll_points(curve, time_steps, np.array([X_star, Y_star, np.zeros(X_star.shape[0])]).T)

        curve = S2Curve(time_values=time_steps,
                        knot_values=K)
        all_curves[idx_iter] = curve
        
        if idx_iter > 1:
            old_curve = all_curves[idx_iter-1]
            err = np.max(np.linalg.norm(curve.knot_values - old_curve.knot_values, axis=1))
            if err < tol: 
                print("Maximul tolerance reached after a total of {} iterations.".format(n_iter))
                break
        
    return curve


#######################################################################
###########################  Other  ###################################
#######################################################################


def cv_UnivariateSpline(X_, Y_, K=5, smin=0.01, smax=10):
    
    if K == -1:
        n_folds = len(X_)
    else:
        n_folds = K
    
    s_all = np.logspace(np.log(smin), np.log(smax), 100)
    Val_error = np.zeros((len(s_all), n_folds))
    
    for i, s in enumerate(s_all):
        kf = KFold(n_splits=n_folds)
        for j, (train_index, val_index) in enumerate(kf.split(X_)):
            X_train = X_[train_index]
            Y_train = Y_[train_index]
            spl = UnivariateSpline(X_train, Y_train, k=3, s=s)
            X_val = X_[val_index]
            Y_val = Y_[val_index]
            Y_hat = spl(X_val)
            Val_error[i,j] = np.sum((Y_val - Y_hat)**2)
            
    Val = np.median(Val_error, axis=1)
    plt.xscale('log')
    plt.yscale('log')
    
    return s_all[np.argmin(Val)]