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




def abs_smooth_dv(x, x_deriv, delta_x):
    """
    Compute the absolute value in a smooth differentiable manner.
    The valley is rounded off using a quadratic function.
    Parameters
    ----------
    x : float
        Quantity value
    x_deriv : float
        Derivative value
    delta_x : float
        Half width of the rounded section.
    Returns
    -------
    float
        Smooth absolute value of the quantity.
    float
        Smooth absolute value of the derivative.
    """
    if x >= delta_x:
        y_deriv = x_deriv
        y = x

    elif x <= -delta_x:
        y_deriv = -x_deriv
        y = -x

    else:
        y_deriv = 2.0 * x * x_deriv / (2.0 * delta_x)
        y = x**2 / (2.0 * delta_x) + delta_x / 2.0

    return y, y_deriv


class AkimaPY(object):

    def __init__(self, xpt, ypt, delta_x=0.1, eps=1e-30):
        """
        Train the akima spline and save the derivatives.
        Conversion of fortran function AKIMA_DV.
        Parameters
        ----------
        xpt : ndarray
            Values at which the akima spline was trained.
        ypt : ndarray
            Training values for the akima spline.
        """
        xpt    = np.array(xpt)
        ncp    = np.size(xpt)
        nbdirs = 2 * ncp

        ypt = np.array(ypt)
        self.flatFlag = (ypt.ndim == 1)
        
        if self.flatFlag:
            ypt = ypt.reshape((1,ncp))
        if ypt.shape[0] == ncp:
            ypt = ypt.T
        vec_size = ypt.shape[0]

        # Poly points and derivs
        p1 = np.empty((vec_size, ncp - 1), dtype=ypt.dtype)
        p2 = np.empty((vec_size, ncp - 1), dtype=ypt.dtype)
        p3 = np.empty((vec_size, ncp - 1), dtype=ypt.dtype)
        p0d = np.empty((vec_size, nbdirs, ncp - 1), dtype=ypt.dtype)
        p1d = np.empty((vec_size, nbdirs, ncp - 1), dtype=ypt.dtype)
        p2d = np.empty((vec_size, nbdirs, ncp - 1), dtype=ypt.dtype)
        p3d = np.empty((vec_size, nbdirs, ncp - 1), dtype=ypt.dtype)

        md = np.zeros((nbdirs, ncp + 3), dtype=ypt.dtype)
        m = np.zeros((ncp + 3, ), dtype=ypt.dtype)
        td = np.zeros((nbdirs, ncp), dtype=ypt.dtype)
        t = np.zeros((ncp, ), dtype=ypt.dtype)

        xptd = np.vstack([np.eye(ncp, dtype=ypt.dtype),
                          np.zeros((ncp, ncp), dtype=ypt.dtype)])
        yptd = np.vstack([np.zeros((ncp, ncp), dtype=ypt.dtype),
                          np.eye(ncp, dtype=ypt.dtype)])

        dx = xpt[1:] - xpt[:-1]
        dx2 = dx**2
        dxd = xptd[:, 1:] - xptd[:, :-1]

        p0 = ypt[:, :-1]

        for jj in range(vec_size):

            ypt_jj = ypt[jj, :]

            # Compute segment slopes
            md[:, 2:ncp + 1] = ((yptd[:, 1:] - yptd[:, :-1]) * (xpt[1:] - xpt[:-1]) -
                                (ypt_jj[1:] - ypt_jj[:-1]) * (xptd[:, 1:] - xptd[:, :-1])) / \
                (xpt[1:] - xpt[:-1]) ** 2

            m[2:ncp + 1] = (ypt_jj[1:] - ypt_jj[:-1]) / (xpt[1:] - xpt[:-1])

            # Estimation for end points.
            md[:, 1] = 2.0 * md[:, 2] - md[:, 3]
            md[:, 0] = 2.0 * md[:, 1] - md[:, 2]
            md[:, ncp + 1] = 2.0 * md[:, ncp] - md[:, ncp - 1]
            md[:, ncp + 2] = 2.0 * md[:, ncp + 1] - md[:, ncp]

            m[1] = 2.0 * m[2] - m[3]
            m[0] = 2.0 * m[1] - m[2]
            m[ncp + 1] = 2.0 * m[ncp] - m[ncp - 1]
            m[ncp + 2] = 2.0 * m[ncp + 1] - m[ncp]

            # Slope at points.
            for i in range(2, ncp + 1):
                m1d = md[:, i - 2]
                m2d = md[:, i - 1]
                m3d = md[:, i]
                m4d = md[:, i + 1]
                arg1d = m4d - m3d

                m1 = m[i - 2]
                m2 = m[i - 1]
                m3 = m[i]
                m4 = m[i + 1]
                arg1 = m4 - m3

                w1, w1d = abs_smooth_dv(arg1, arg1d, delta_x)

                arg1d = m2d - m1d
                arg1 = m2 - m1

                w2, w2d = abs_smooth_dv(arg1, arg1d, delta_x)

                if w1 < eps and w2 < eps:
                    # Special case to avoid divide by zero.
                    td[:, i - 2] = 0.5 * (m2d + m3d)
                    t[i - 2] = 0.5 * (m2 + m3)

                else:
                    td[:, i - 2] = ((w1d * m2 + w1 * m2d + w2d * m3 + w2 * m3d) *
                                    (w1 + w2) - (w1 * m2 + w2 * m3) * (w1d + w2d)) \
                        / (w1 + w2) ** 2

                    t[i - 2] = (w1 * m2 + w2 * m3) / (w1 + w2)

            # Polynomial Coefficients
            t1 = t[:-1]
            t2 = t[1:]

            p1[jj, :] = t1
            p2[jj, :] = (3.0 * m[2:ncp + 1] - 2.0 * t1 - t2) / dx
            p3[jj, :] = (t1 + t2 - 2.0 * m[2:ncp + 1]) / dx2

            p0d[jj, ...] = yptd[:, :-1]
            p1d[jj, ...] = td[:, :-1]
            p2d[jj, ...] = ((3.0 * md[:, 2:ncp + 1] - 2.0 * td[:, :-1] - td[:, 1:]) * dx -
                            (3.0 * m[2:ncp + 1] - 2.0 * t1 - t2) * dxd) / dx2
            p3d[jj, ...] = ((td[:, :-1] + td[:, 1:] - 2.0 * md[:, 2:ncp + 1]) * dx2 -
                            (t1 + t2 - 2.0 * m[2:ncp + 1]) * 2 * dx * dxd) / (dx2)**2

        self.xpt = xpt
        
        self.p0 = p0
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3

        self.dp0_dxcp = p0d[:, :ncp, :].transpose((0, 2, 1))
        self.dp0_dycp = p0d[:, ncp:, :].transpose((0, 2, 1))
        self.dp1_dxcp = p1d[:, :ncp, :].transpose((0, 2, 1))
        self.dp1_dycp = p1d[:, ncp:, :].transpose((0, 2, 1))
        self.dp2_dxcp = p2d[:, :ncp, :].transpose((0, 2, 1))
        self.dp2_dycp = p2d[:, ncp:, :].transpose((0, 2, 1))
        self.dp3_dxcp = p3d[:, :ncp, :].transpose((0, 2, 1))
        self.dp3_dycp = p3d[:, ncp:, :].transpose((0, 2, 1))

    def interp(self, x):

        xcp = self.xpt
        ncp = np.size(xcp)
        n   = np.size(x)
        vec_size = self.p0.shape[0]
        
        p0 = self.p0
        p1 = self.p1
        p2 = self.p2
        p3 = self.p3

        # All vectorized points uses same grid, so find these once.
        j_idx = np.zeros(n, dtype=np.int)
        for i in range(n):

            # Find location in array (use end segments if out of bounds)
            if x[i] < xcp[0]:
                j = 0

            else:
                # Linear search for now
                for j in range(ncp - 2, -1, -1):
                    if x[i] >= xcp[j]:
                        break

            j_idx[i] = j

        dx = x - xcp[j_idx]
        dx2 = dx * dx
        dx3 = dx2 * dx

        # Evaluate polynomial (and derivative)
        y = p0[:, j_idx] + p1[:, j_idx] * dx + p2[:, j_idx] * dx2 + p3[:, j_idx] * dx3

        dydx = p1[:, j_idx] + 2.0 * p2[:, j_idx] * dx + 3.0 * p3[:, j_idx] * dx2

        dydxcp = self.dp0_dxcp[:, j_idx, :] + \
            np.einsum('kij,i->kij', self.dp1_dxcp[:, j_idx, :], dx) + \
            np.einsum('kij,i->kij', self.dp2_dxcp[:, j_idx, :], dx2) + \
            np.einsum('kij,i->kij', self.dp3_dxcp[:, j_idx, :], dx3)

        for jj in range(vec_size):
            for i in range(n):
                j = j_idx[i]
                dydxcp[jj, i, j] -= dydx[jj, i]

        dydycp = self.dp0_dycp[:, j_idx, :] + \
            np.einsum('kij,i->kij', self.dp1_dycp[:, j_idx, :], dx) + \
            np.einsum('kij,i->kij', self.dp2_dycp[:, j_idx, :], dx2) + \
            np.einsum('kij,i->kij', self.dp3_dycp[:, j_idx, :], dx3)

        if self.flatFlag:
            y = np.squeeze(y)
            dydx = np.squeeze(dydx)
            dydxcp = np.squeeze(dydxcp)
            dydycp = np.squeeze(dydycp)
        return (y, dydx, dydxcp, dydycp)        
