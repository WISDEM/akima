#!/usr/bin/env python
# encoding: utf-8
"""
example.py

Created by Andrew Ning on 2013-12-17.
"""

import numpy as np
from akima import Akima, AkimaPY, akima_interp
from scipy.interpolate import Akima1DInterpolator

# setup spline based on fixed points
xpt = np.array([1.0, 2.0, 4.0, 6.0, 10.0, 12.0])
ypt = np.array([5.0, 12.0, 14.0, 16.0, 21.0, 29.0])
spline = Akima(xpt, ypt)
spline2 = Akima1DInterpolator(xpt, ypt)

# interpolate  (extrapolation will work, but beware the results may be silly)
n = 50
x = np.linspace(1.001, 11.999, n)
y, dydx, dydxpt, dydypt = spline.interp(x)
y2 = spline2(x)

# an alternative way to call akima if you don't care about derivatives
# (and also don't care about evaluating the spline multiple times)
# a slight amount of smoothing is used for the one with derivatives so
# y will not exactly match y2 unless you set delta_x=0.0 in the Akima constructor
spline3 = AkimaPY(xpt, ypt)
y3,dydx3,dydxpt3,dydypt3 = spline3.interp(x)

# compare derivatives w.r.t. x to finite differencing
h = 1e-6
xstep = x + h  # can do all steps at same time b.c. they are independent
ystep, _, _, _ = spline.interp(xstep)
fd = (ystep - y)/h

# compare derivatives of xpt and ypt for one point
idx = 3
xptstep = np.copy(xpt)
xptstep[idx] += h
spline = Akima(xptstep, ypt)
ystep, _, _, _ = spline.interp(x)
fd2 = (ystep - y)/h

yptstep = np.copy(ypt)
yptstep[idx] += h
spline = Akima(xpt, yptstep)
ystep, _, _, _ = spline.interp(x)
fd3 = (ystep - y)/h

import matplotlib.pyplot as plt
plt.plot(xpt, ypt, 'o')
plt.plot(x, y, x, y2, '--', x, y3, '-.')
plt.legend(('Control Points','In-House','SciPy','dx=0'))

plt.figure()
plt.plot(x, dydx)
plt.plot(x, fd, '--')

plt.figure()
plt.plot(x, dydxpt[:, idx])
plt.plot(x, fd2, '--')

plt.figure()
plt.plot(x, dydypt[:, idx])
plt.plot(x, fd3, '--')

plt.show()
