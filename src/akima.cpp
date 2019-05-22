
#include <cmath>
#include <vector>
#include <tuple>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

using namespace std;


tuple<double, vector<double>> abs_smooth(const double x, const vector<double> &dx, const double delta_x) {
  // absolute value function with small quadratic in the valley
  // so that it is C1 continuous
  double y;
  vector<double> dy(dx.size(), 0.0);
  if (x >= delta_x) {
    y  = x;
    for (size_t k=0; k<dy.size(); k++) dy[k] = dx[k];
  } else if (x <= -delta_x) {
    y  = -x;
    for (size_t k=0; k<dy.size(); k++) dy[k] = -dx[k];
  } else {
    y  = pow(x, 2.0) / (2.0*delta_x) + delta_x/2.0;
    for (size_t k=0; k<dy.size(); k++) dy[k] = 2.0 * x * dx[k] / (2.0*delta_x);
  }
  
  return make_tuple(y, dy);
}


class Akima {
public:
  vector<double> xpt, y, dydx, p0, p1, p2, p3;
  vector<double> dydxpt, dydypt;
  vector<double> dp0dxpt, dp1dxpt, dp2dxpt, dp3dxpt;
  vector<double> dp0dypt, dp1dypt, dp2dypt, dp3dypt;

  Akima(const vector<double> &xpt, const vector<double> &ypt, const double delta_x) {
    // setup the Akima spline function
    double eps = 1e-20;

    // Number of control points
    int n = xpt.size();

    // Copying vector by assign function
    this->xpt.resize(n);
    this->xpt.assign(xpt.begin(), xpt.end());
    
    // Number of differentiation directions
    int nbdirs = 2*n; // xptd.size();
    vector<double> xptd(nbdirs*n, 0.0);
    vector<double> yptd(nbdirs*n, 0.0);
    for (size_t i=0; i<n; i++) {
      xptd[i*n+i] = 1.0;
      yptd[(n+i)*n+i] = 1.0;
    }
    
    // Segment slopes
    vector<double> m(n + 3, 0.0);
    
    vector<double> md(nbdirs*(n+3), 0.0);
    vector<double> m1d(nbdirs, 0.0);
    vector<double> m2d(nbdirs, 0.0);
    vector<double> m3d(nbdirs, 0.0);
    vector<double> m4d(nbdirs, 0.0);

    vector<double> w1d(nbdirs, 0.0);
    vector<double> w2d(nbdirs, 0.0);
    
    // Middle points
    for (size_t i=0; i<n-1; i++) {
      m[i+2] = (ypt[i+1] - ypt[i]) / (xpt[i+1] - xpt[i]);
      for (size_t nd=0; nd<nbdirs; nd++) {
	md[nd*(n+3)+i+2] = ( (yptd[nd*n+i+1] - yptd[nd*n+i]) - m[i+2]*(xptd[nd*n+i+1] - xptd[nd*n+i]) ) / (xpt[i+1] - xpt[i]);
      }
    }
    // End points
    for (size_t nd=0; nd<nbdirs; nd++) {
      md[nd*(n+3)+  1] = 2.0*md[nd*(n+3)+2]   - md[nd*(n+3)+3];
      md[nd*(n+3)+  0] = 2.0*md[nd*(n+3)+1]   - md[nd*(n+3)+2];
      md[nd*(n+3)+n+1] = 2.0*md[nd*(n+3)+n]   - md[nd*(n+3)+n-1];
      md[nd*(n+3)+n+2] = 2.0*md[nd*(n+3)+n+1] - md[nd*(n+3)+n];
    }
    m[1]   = 2.0*m[2]   - m[3];
    m[0]   = 2.0*m[1]   - m[2];
    m[n+1] = 2.0*m[n]   - m[n-1];
    m[n+2] = 2.0*m[n+1] - m[n];

    
    // Slope at points
    vector<double> t(n, 0.0);
    
    vector<double> td(nbdirs*n, 0.0);
    vector<double> t1d(nbdirs, 0.0);
    vector<double> t2d(nbdirs, 0.0);
    vector<double> dxd(nbdirs, 0.0);
    vector<double> arg1d(nbdirs, 0.0);
    vector<double> arg2d(nbdirs, 0.0);
    
    for (size_t i=2; i<n+1; i++) {
      for (size_t nd=0; nd<nbdirs; nd++) {
	m1d[nd]   = md[nd*(n+3)+i-2];
	m2d[nd]   = md[nd*(n+3)+i-1];
	m3d[nd]   = md[nd*(n+3)+i];
	m4d[nd]   = md[nd*(n+3)+i+1];
	arg1d[nd] = m4d[nd] - m3d[nd];
	arg2d[nd] = m2d[nd] - m1d[nd];
      }
      double m1   = m[i - 2];
      double m2   = m[i - 1];
      double m3   = m[i];
      double m4   = m[i + 1];
      double arg1 = m4 - m3;
      double arg2 = m2 - m1;

      auto result1 = abs_smooth(arg1, arg1d, delta_x);
      double w1   = get<0>(result1);
      vector<double> w1d  = get<1>(result1);
      auto result2 = abs_smooth(arg2, arg2d, delta_x);
      double w2   = get<0>(result2);
      vector<double> w2d  = get<1>(result2);
	
      if ( w1 < eps && w2 < eps ) {
	t[i-2] = 0.5*(m2 + m3); // special case to avoid divide by zero
	for (size_t nd=0; nd<nbdirs; nd++)
	  td[nd*n+i-2] = 0.5*(m2d[nd] + m3d[nd]);
      } else {
	t[i-2] = (w1*m2 + w2*m3) / (w1 + w2);
	for (size_t nd=0; nd<nbdirs; nd++)
	  td[nd*n+i-2] = ( (w1d[nd]*m2 + w1*m2d[nd] + w2d[nd]*m3 + w2*m3d[nd]) -
			  t[i-2]*(w1d[nd] + w2d[nd]) ) / (w1+w2);
      }
    }

    // Polynomial coefficients
    p0.resize(n-1);
    p1.resize(n-1);
    p2.resize(n-1);
    p3.resize(n-1);

    vector<double> p0d(nbdirs*(n-1), 0.0);
    vector<double> p1d(nbdirs*(n-1), 0.0);
    vector<double> p2d(nbdirs*(n-1), 0.0);
    vector<double> p3d(nbdirs*(n-1), 0.0);
    
    for (size_t i=0; i<n-1; i++) {
      double dx = xpt[i+1] - xpt[i];
      double t1 = t[i];
      double t2 = t[i+1];
      
      p0[i] = ypt[i];
      p1[i] = t1;
      p2[i] = (3.0*m[i+2] - 2.0*t1 - t2) / dx;
      p3[i] = (t1 + t2 - 2.0*m[i+2]) / pow(dx, 2.0);
      for (size_t nd=0; nd<nbdirs; nd++) {
	dxd[nd] = xptd[nd*n+i+1] - xptd[nd*n+i];
	t1d[nd] = td[nd*n+i];
	t2d[nd] = td[nd*n+i+1];
	
	p0d[nd*(n-1)+i] = yptd[nd*n+i];
	p1d[nd*(n-1)+i] = t1d[nd];
	p2d[nd*(n-1)+i] = ((3.0*md[nd*(n+3)+2+i] - 2.0*t1d[nd] - t2d[nd]) - p2[i]*dxd[nd]) / dx;
	p3d[nd*(n-1)+i] = ((t1d[nd] + t2d[nd] - 2.0*md[nd*(n+3)+2+i])/dx - p3[i]*2*dx*dxd[nd]) / dx;
      }
    }

    dp0dxpt.resize(n*(n-1));
    dp1dxpt.resize(n*(n-1));
    dp2dxpt.resize(n*(n-1));
    dp3dxpt.resize(n*(n-1));

    dp0dypt.resize(n*(n-1));
    dp1dypt.resize(n*(n-1));
    dp2dypt.resize(n*(n-1));
    dp3dypt.resize(n*(n-1));
    for (size_t i=0; i<n; i++) {
      for (size_t j=0; j<n-1; j++) {
	dp0dxpt[j*n+i] = p0d[i*(n-1)+j];
	dp1dxpt[j*n+i] = p1d[i*(n-1)+j];
	dp2dxpt[j*n+i] = p2d[i*(n-1)+j];
	dp3dxpt[j*n+i] = p3d[i*(n-1)+j];

	dp0dypt[j*n+i] = p0d[(n+i)*(n-1)+j];
	dp1dypt[j*n+i] = p1d[(n+i)*(n-1)+j];
	dp2dypt[j*n+i] = p2d[(n+i)*(n-1)+j];
	dp3dypt[j*n+i] = p3d[(n+i)*(n-1)+j];
      }
    }
  }


  void interp(const vector<double> x) {

    // Number of control and interpolation points
    int nctrl = xpt.size();
    int n   = x.size();

    // Initialize outputs
    y.resize(n);      // interpolate y values
    dydx.resize(n);   // derivative of y w.r.t. x
    dydxpt.resize(n*nctrl); // derivative of y w.r.t. xpt
    dydypt.resize(n*nctrl); // derivative of y w.r.t. ypt

    // interpolate at each point
    for (size_t i=0; i<n; i++) {
      int j = 0;

      // find location in array (use end segments if out of bounds)
      if (x[i] < xpt[0]) {
	j = 0;
      } else {
	// linear search for now
	for (j=nctrl-1; j>0; j--)
	  if (x[i] >= xpt[j])
	    break;
      }

      // evaluate polynomial (and derivative)
      double dx = x[i] - xpt[j];
      y[i]      = p0[j] + p1[j]*dx + p2[j]*pow(dx, 2.0) + p3[j]*pow(dx, 3.0);
      dydx[i]   = p1[j] + 2.0*p2[j]*dx + 3.0*p3[j]*pow(dx, 2.0);

      for (size_t k=0; k<nctrl; k++) {
	dydxpt[i*nctrl+k] = dp0dxpt[j*nctrl+k] + dp1dxpt[j*nctrl+k]*dx + dp2dxpt[j*nctrl+k]*pow(dx, 2.0) + dp3dxpt[j*nctrl+k]*pow(dx, 3.0);
	if (k == j) dydxpt[i*nctrl+k] = dydxpt[i*nctrl+k] - dydx[i];
	dydypt[i*nctrl+k] = dp0dypt[j*nctrl+k] + dp1dypt[j*nctrl+k]*dx + dp2dypt[j*nctrl+k]*pow(dx, 2.0) + dp3dypt[j*nctrl+k]*pow(dx, 3.0);
      }
    }
    
  }
  
};



//MARK: ---------- PYTHON WRAPPER FOR AKIMA ---------------------
namespace py = pybind11;

class pyAkima {
  Akima *akima;

public:
  pyAkima(const py::array_t<double, py::array::c_style> xpt, const py::array_t<double, py::array::c_style> ypt, const double delta_x) {
    // allocate std::vector (to pass to the C++ function)
    vector<double> xpt_vec(xpt.size());
    vector<double> ypt_vec(ypt.size());

    // copy py::array -> std::vector
    memcpy(xpt_vec.data(), xpt.data(), xpt.size()*sizeof(double));
    memcpy(ypt_vec.data(), ypt.data(), ypt.size()*sizeof(double));
    
    akima = new Akima(xpt_vec, ypt_vec, delta_x);
  }

  ~pyAkima(){
    delete akima;
  }

  py::tuple interp(const py::array_t<double, py::array::c_style> x) {
    // allocate std::vector (to pass to the C++ function)
    vector<double> x_vec(x.size());

    // copy py::array -> std::vector
    memcpy(x_vec.data(), x.data(), x.size()*sizeof(double));

    // Call function
    akima->interp(x_vec);

    auto y_resultB = py::buffer_info(
                // Pointer to buffer
                akima->y.data(),
                // Size of one scalar
                sizeof(double),
                // Python struct-style format descriptor
                py::format_descriptor<double>::format(),
                // Number of dimensions
                1,
                // Buffer dimensions
                { akima->y.size() },
                // Strides (in bytes) for each index
                { sizeof(double) }
            );
    auto y_result = py::array_t<double>(y_resultB);

    auto dy_resultB = py::buffer_info(
                // Pointer to buffer
                akima->dydx.data(),
                // Size of one scalar
                sizeof(double),
                // Python struct-style format descriptor
                py::format_descriptor<double>::format(),
                // Number of dimensions
                1,
                // Buffer dimensions
                { akima->dydx.size() },
                // Strides (in bytes) for each index
                { sizeof(double) }
            );
    auto dy_result = py::array_t<double>(dy_resultB);

    auto dydxpt_resultB = py::buffer_info(
                // Pointer to buffer
                akima->dydxpt.data(),
                // Size of one scalar
                sizeof(double),
                // Python struct-style format descriptor
                py::format_descriptor<double>::format(),
                // Number of dimensions
                1,//2,
                // Buffer dimensions
                { akima->dydxpt.size() }, //, akima->dydxpt[0].size() },
                // Strides (in bytes) for each index
                { sizeof(double) } //*akima->dydxpt[0].size(), sizeof(double) }
            );
    auto dydxpt_result = py::array_t<double>(dydxpt_resultB);

    auto dydypt_resultB = py::buffer_info(
                // Pointer to buffer
                akima->dydypt.data(),
                // Size of one scalar
                sizeof(double),
                // Python struct-style format descriptor
                py::format_descriptor<double>::format(),
                // Number of dimensions
                1,//2,
                // Buffer dimensions
                { akima->dydypt.size() }, //, akima->dydypt[0].size() },
                // Strides (in bytes) for each index
                { sizeof(double) } //*akima->dydypt[0].size(), sizeof(double) }
            );
    auto dydypt_result = py::array_t<double>(dydypt_resultB);

    return py::make_tuple(y_result, dy_result, dydxpt_result, dydypt_result);
  }
};


// MARK: --------- PYTHON MODULE ---------------

PYBIND11_MODULE(_akima, m)
{
  m.doc() = "akima python plugin module";
  
  py::class_<pyAkima>(m, "Akima")
    .def(py::init<py::array_t<double, py::array::c_style>, py::array_t<double, py::array::c_style>, double>())
    .def("interp", &pyAkima::interp)
    ;
}
