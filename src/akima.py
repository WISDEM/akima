#!/usr/bin/env python
# encoding: utf-8
"""
akima.py

Created by Andrew Ning on 2013-12-17.
"""

import _akima
import numpy as np


def akima_interp(xpt, ypt, x):
    """convenience method for those who don't want derivatives
    and don't want to evaluate the same spline multiple times"""

    a = Akima(xpt, ypt, delta_x=0.0)
    y, _ = a.interp(x)
    return y



def akima_interp_with_derivs(xpt, ypt, x, delta_x=0.1):
    a = Akima(xpt, ypt, delta_x)
    return a.interp(x)




class Akima(object):
    """class for evaluating Akima spline and its derivatives"""


    def __init__(self, xpt, ypt, delta_x=0.1):
        """setup akima spline

        Parameters
        ----------
        xpt : array_like
            x discrete data points
        ypt : array_like
            x discrete data points
        delta_x : float, optional
            half-width of the smoothing interval added in the valley of absolute-value function
            this allows the derivatives with respect to the data points (dydxpt, dydypt)
            to also be C1 continuous.
            set to parameter to 0 to get the original Akima function (but only if
            you don't need dydxpt, dydypt)

        """
        xpt = np.array(xpt)
        ypt = np.array(ypt)
        self.nctrl = len(xpt)
        self.delta_x = delta_x
        self.akimaObj = _akima.Akima(xpt, ypt, float(delta_x))





    def interp(self, x):
        """interpolate at new values

        Parameters
        ----------
        x : array_like
            x values to sample spline at

        Returns
        -------
        y : nd_array
            interpolated values y = fspline(x)
        dydx : nd_array
            the derivative of y w.r.t. x at each point
        dydxpt : 2D nd_array (only returned if delta_x != 0.0)
            dydxpt[i, j] the derivative of y[i] w.r.t. xpt[j]
        dydypt : 2D nd_array (only returned if delta_x != 0.0)
            dydypt[i, j] the derivative of y[i] w.r.t. ypt[j]

        """

        x = np.asarray(x)
        npt = len(x)
        try:
            len(x)
            isFloat = False
        except TypeError:  # if x is just a float
            x = np.array([x])
            isFloat = True

        if x.size == 0:  # error check for empty array
            y = np.array([])
            dydx = np.array([])
            dydxpt = np.array([])
            dydypt = np.array([])
        else:
            y, dydx, dydxpt, dydypt = self.akimaObj.interp(x)
            dydxpt = dydxpt.reshape((npt, self.nctrl))
            dydypt = dydypt.reshape((npt, self.nctrl))
            
        if isFloat:
            y = y[0]
            dydx = dydx[0]
            dydxpt = dydxpt[0, :]
            dydypt = dydypt[0, :]

        if self.delta_x == 0.0:
            return y, dydx
        else:
            return y, dydx, dydxpt, dydypt
