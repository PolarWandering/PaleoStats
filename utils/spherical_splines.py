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
    """
    Cartesian to spherical coordinate operator.
    
    Parameters
    ----------
    X : array_like, (N,3) 
        Cartesian coordinates. 
    radians : bool, optional
        Whether to use radians or degrees. 
        
    Returns
    -------
    Y : array_like, (N,2)
        Spherical coordinates
    
    """
    
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
    Spherical to cartesian coordinates. 
    
    Parameters
    ----------
    v : array_like, (2,)
        Two vector composed on (latitude, longitude)

    Returns 
    -------
    array_like, (3,)
        Vector with unit norm representing the point in the sphere.
    
    """
    
    x = np.cos(v[0]) * np.cos(v[1])  
    y = np.cos(v[0]) * np.sin(v[1])  
    z = np.sin(v[0])  
    
    return [x,y,z] / np.linalg.norm([x,y,z])


def slerp_interpolation(p1, p2, t, order, tmin=None, tmax=None):
    """
    Returns the Slerp interpolation between two unitary vectors on the 3D sphere following a great circle.
    
    Parameters
    ----------
    p1, p2 : array_like, (3,)
        Vectors used for the interpolation. 
    t : float, between 0 and 1. 
        Parameter for the interpolations. p=0 will return p1 while p=1 will give p2. 
    order : {0,1}
        Order of the interpolation. order=0 refers to the classical interpolation, while order=1 is for the derivative.
    tmin, tmax : float
        Times used for computing the time derivative. Just needed for order=1. 
        
    Returns
    -------
    array_like, (3,)
        Slerp interpolation
    
    """
    
    angle = np.arccos(np.dot(p1,p2))

    if order == 0:
        return ( np.sin((1-t)*angle) * p1 + np.sin(t*angle) * p2 ) / np.sin(angle)
    elif order == 1:
        return angle * ( np.cos(t*angle) * p2 - np.cos((1-t)*angle) * p1 ) / (np.sin(angle) * (tmax-tmin)) 
    else:
        raise ValueError('Order derivative not implemented.')
        

def skew_matrix(v):
    """
    Given a vector, defines a matrix that is equivalent to the cross product. That is, M(v)([.]) = v x [.]
    
    Parameters
    ----------
    v : array_like, (3,)
        Vector that defines the cross product operator.
        
    Returns 
    -------
    array_like, (3,3)
        Matrix that defines the linear operator.
    
    """

    skv = np.roll(np.roll(np.diag(v.flatten()), 1, 1), -1, 0)
    return skv - skv.T


def equal_distance_projection(X):
    """
    Equal distance projection around the north pole as defined in Jupp et. al. (1987)
    
    Parameters
    ----------
    X : array_like, (N,3)
        List of points in the sphere to project
        
    Returns
    -------
    Z : array_like, (N,3)
        Projected points in the plane, last component equals zero. 
    
    """
    
    Y = cart2sph(X, radians=True)
    Z = np.zeros(X.shape)
            
    Z[:,0] = (np.pi/2-Y[:,1]) * np.cos(Y[:,0])
    Z[:,1] = (np.pi/2-Y[:,1]) * np.sin(Y[:,0])
    
    return Z

        
#######################################################################
###################### Curve object definition ########################
#######################################################################

class S2Curve:
    """
    Curve clase we are going to fit in the sphere.
    This will be consist in a series of timesteps where we can exactly evaluate the curve. For all the intermediate steps, we use the 
    great circle between adjacent knots to evalaute the function 
    """
    
    def __init__(self, time_values, knot_values):
        """
        Constructor for S2Curve object
        
        Parameters
        ----------
        time_values : array_like, (N,)
            Time where the curve is evaluated. 
        knot_values : array_like, (N,3)
            Cartesian coordinates of the points used to define the curve. Middle points will be interpolated using great circle. 
        planar_values : array_like, (N,3), optional
            Planar projection of the curve. Last component is always zero. 
        Rotations : array_like, (N,3,3), optional
            Rotations needed to unrill/rolled the curve. 
        """
        
        self.time_values = time_values  
        self.knot_values = knot_values   
        self.planar_values = None        
        self.Rotations = None            
    
    
    def evaluate(self, t, order=0):
        """
        Return evaluation of the curve
        
        Parameters
        ----------
        t : float
            time we we evaluate the function or any of its derivatives.
        order : int, optional
            order correspondint to the evaluation of the derivative. 
            If equals to 0 this is just the evaluation of the function. 
        
        Returns 
        -------
        array_like
            Estimation of the function evaluations / gradients on the sphere based on the 
            infomation provided by the curve. 
        
        """
        assert (np.min(self.time_values) <= t) & (t <= np.max(self.time_values)), 'Age value is not inside the evaluation window.'
        idx_inf = np.max(np.where(self.time_values - t <= 0.0))
        idx_sup = np.min(np.where(self.time_values - t >= 0.0))

        if idx_inf == idx_sup and order == 0:
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
        """
        Plotting capacity for quick check on the trajectory of the curve. 
        """
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
    """
    Unrolling curve from the sphere to a series of points in the plane and the rotation requiered
    to make the unrolling. 
    
    Parameters 
    ----------
    curve : S2Curve object
        Base curve with respect we are going to create an unrolled version
    delta_t: float, optional
        Timestep for the differential equation solver
    
    Returns 
    -------
    curve.time_values : array_like
        Time evaluations of the curve
    f_star : array_like, (N,3)
        Unrolled points based on the curve
    Rotation : array_like, (N,3,3)
        Series of rotations needed to unroll poitns
    
    """
    
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
    
    f_ = [0,0,0]
    R = R0.as_matrix()
    f_star[0,:] = f_
    Rotation[0,:,:] = R
    
    for idx in range(len(curve.time_values)-1):
    
        t = curve.time_values[idx]
        next_t = curve.time_values[idx+1]
        
        while t < next_t:
            
            dtt = min(delta_t, next_t - t)
            
            f = curve.evaluate(t, order=0)
            df = curve.evaluate(t, order=1)
            assert not np.any(np.isnan(f)), 'The evaluation of f in time {} includes nan values: {}'.format(t, f)
            assert not np.any(np.isnan(df)), 'The evaluation of df in time {} includes nan values: {}'.format(t, df)
            f_ += dtt * np.dot(R, df)
            f_[-1] = 0  # we force the last component of f* to be zero 
            R += dtt * np.dot(R, skew_matrix(np.cross(df,f)))
            t += dtt

        f_star[idx+1,:] = f_ 
        assert np.abs(f_star[idx+1,2]) < 0.01, 'Reduce the solver time step to ensure the solution is correct. The z-component of f^* gives {} instead of 0.0'.format(f_star[idx+1,2])
        Rotation[idx+1,:,:] = R
        
    return curve.time_values, f_star, Rotation


def roll_curve(times, X, delta_t=0.1):
    """
    We roll the points that define a curve 
    
    Parameters
    ----------
    times : array_like, (M,)
        Times associated to each point in the curve.
    X : array_like, (M,3)
        Points in the plane to roll
    
    Returns 
    -------
    S2Curve object with the curve rolled in the sphere. 
        
    """
    
    N = X.shape[0]
    
    F = np.zeros((N,3))   
    Rotation = np.zeros((N,3,3))

    # Derivative of the projected path
    Df_star = np.diff(X, axis=0) / np.diff(times)[:, np.newaxis]

    # initial conditions
    f = [0,0,1]
    df = [1,0,0] 
    
    # Initial condition for the rotation needs to be pick such that f*'(0) = R(0) f'(0) 
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
    """
    We unroll the points D using a given curve.
    
    Parameters
    ----------
    curve : S2curve object
        Curve with respect the rolling is going to be performed. 
    point_times : array_like, (M,)
        Times associated to each point.
    D : array_like, (M,3)
        Matrix of points to be rolled to the sphere with respect to a given curve.
    
    Returns 
    -------
    D_star : array_like (M,)
        
    """
    
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
    """
    Rolling of points in the plane to the sphere with respect to a reference curve. 
    
    Parameters
    ----------
    curve : S2curve object
        Curve with respect the unrolling is going to be performed. 
    point_times : array_like, (M,)
        Times associated to each point.
    D : array_like (M,3) 
        Matrix of points to be rolled to the sphere with respect to a given curve.
        The last component must be equal to zero.
        
    Returns
    -------
    K : array_like, (M,3)
        Array of rolled points.
    
    """
    
    K = np.zeros((len(point_times), 3))
    
    for idx, t in enumerate(point_times):
        # First we identify the right time 
        idx_inf = np.max(np.where(curve.time_values-t <= 0.0))
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
    return K
        
        
#######################################################################
######################  Computation of splines  #######################
#######################################################################


def spherical_spline(times, 
                     knot_values, 
                     smoothing, 
                     precision=0.1, 
                     ode_stepsize=0.01, 
                     n_iter=5, 
                     tol=0.001, 
                     weights=None, 
                     fix_origin=True, 
                     reference_hemisphere='North'):
    
    """
    Constructs a smooth fitiing on points in the sphere based on the methods introduced 
    in Jupp et. al. (1987). This function executes the rolling and unrolling of points from 
    the sphere to the 2D plane and fits spline in the plane. 
    
    Parameters
    ----------
    times : array_like, (N,)
        Time series associated with the points in the sphere.
    knot_values: 2D array_like, (N,3)
        Cartesian cordinates of the points to fit. These points must be restricted to 
        lie in the sphere
    smoothing: float between 0 and 1. 
        Smoothing parameter used in the 2D splines method. Lower values correspond to more 
        smoothing, while larger value will weight more the data over the smoothing. For more 
        information, see https://csaps.readthedocs.io/en/latest/index.html\
    precision: float, positive, optional
        Precision in time units of the curve. 
    ode_stepsize: float, positive, optional
        Stepsize of the numerical solver for the differential equations involved in the 
        rolling/unrolling of the curve from the sphere to the plane and viceversa. 
    n_iter: int, optional
        Number of iterations of unrolling-splines fit-rolling. If the curve doens't 
        significatively change after certain iterations, the method stops. 
    tol: float, optional
        Total tolerance for stopping the iteration procedure. 
    weights: array_like, (N,), optional
        Weights use for the splines. By defaul, the weights are uniform across points. 
        However, we can manually set the importance we want to give to each pole by 
        increasing its weight. 
    fix_origin: bool, optional
        Boolean variable for setting the position of the first point in the path (age=0)
        in the North or South pole, depending the value of `reference)_hemisphere`.
    reference_hemispher: {'South', 'North'}
        Reference pole using to construct the fit. This will mostly depend of the location of 
        the points, but both options are equivalent and can be used. 
        
    Returns
    -------
    curve.time_values: array_like, (M,)
        Time series associated to the final curve fit
    curve.knot_values: array_like, (M,3)
        Cartesian coordiantes of the final splines. 

    Notice that the internals of this funciton include objects of the class `CurveS2`. This
    has been implemented in order to simplify the manipulations of the code, but the function 
    just resturns the time series and the cartesian coordinates of the spline. 

    """
    
    # We define internal variables for times, knot_values and weights 
    times_ = np.array(times)
    if reference_hemisphere=='North':
        knot_values_ = np.array(knot_values)
    elif reference_hemisphere=='South':
        knot_values_ = -np.array(knot_values)
    if weights is None:
        weights_ = np.ones(len(times_))
    else:
        weights_ = np.array(weights)
    
    idx_ord = np.argsort(times_)
    
    times_ = times_[idx_ord]
    knot_values_ = knot_values_[idx_ord, :]
    weights_ = weights_[idx_ord]
    
    # We inclide the origin of the path 
    if fix_origin:
        if times_[0] > 0.001:
            times_ = np.insert(times_, 0, 0.0)
            knot_values_ = np.insert(knot_values_, 0, [0., 0., 1.], axis=0)
            weights_ = np.insert(weights_, 0, 100*np.sum(weights_))
        else:
            weights_[0] = 100*np.sum(weights_)
    
    # We first define a curve with the values we have in the dataset
    curve_original = S2Curve(time_values=times_[[0,-1]],
                             knot_values=knot_values_[[0,-1]])

    time_steps = np.arange(np.min(times_), np.max(times_)-0.0001, precision)
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
        D_star = unroll_points(curve, times_, knot_values_)
        
        ### Fit Splines ###
        X_star = csaps(times_, D_star[:,0], curve.time_values, weights=weights_, smooth=smoothing)
        Y_star = csaps(times_, D_star[:,1], curve.time_values, weights=weights_, smooth=smoothing)
        
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
    
    if reference_hemisphere=='South':
        curve = S2Curve(time_values=time_steps,
                        knot_values=-K)
        
    return curve.time_values, curve.knot_values


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