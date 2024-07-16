import numpy as np
import matplotlib.pyplot as plt
from pyomo.opt.plugins import sol

from scipy.interpolate import RectBivariateSpline
from scipy.integrate import solve_ivp


def get_path(U, V, x0, y0, t_max=10, dt=1):
    def flow_field(t, z):
        """Interpolate flow field between grid-points"""
        x, y = z
        u_val = u_spline(y, x, grid=False)
        v_val = v_spline(y, x, grid=False)
        return [u_val, v_val]

    h, w = U.shape
    y = np.arange(h)
    x = np.arange(w)

    u_spline = RectBivariateSpline(y, x, U)
    v_spline = RectBivariateSpline(y, x, V)

    # Time span for integration
    t_span = (0, t_max)  # From t=0 to t=10
    t_eval = np.linspace(0, t_max, t_max * dt)  # Time points where the solution is saved

    # Solve the differential equations
    sol = solve_ivp(flow_field, t_span, [x0, y0], method='RK45', t_eval=t_eval)

    # Extract the trajectory
    trajectory_x = sol.y[0]
    trajectory_y = sol.y[1]

    return trajectory_x, trajectory_y


def test_field():
    w, h = 20, 15

    x = np.arange(w)
    y = np.arange(h)
    X, Y = np.meshgrid(x, y)

    U = (Y / (Y.max() / 2)) - 1
    V = -((X / (X.max() / 2)) - 1)

    return x, y, U, V


def plot_path(x, y, U, V, trajectory_x, trajectory_y):
    # Plot the trajectory
    X, Y = np.meshgrid(x, y)
    plt.quiver(X, Y, U, V)
    plt.plot(trajectory_x, trajectory_y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Particle Trajectory in Flow Field')
    plt.show()


def test_integration():
    def flow_field(t, z):
        """Interpolate flow field between grid-points"""
        x, y = z
        u_val = u_spline(y, x, grid=False)
        v_val = v_spline(y, x, grid=False)
        return [u_val, v_val]

    x, y, U, V = test_field()
    
    u_spline = RectBivariateSpline(y, x, U)
    v_spline = RectBivariateSpline(y, x, V)
    
    x0, y0 = 5, 2
    
    # Time span for integration
    t_span = (0, 10)  # From t=0 to t=10
    t_eval = np.linspace(0, 10, 500)  # Time points where the solution is evaluated
    
    # Solve the differential equations
    sol = solve_ivp(flow_field, t_span, [x0, y0], method='RK45', t_eval=t_eval)
    
    # Extract the trajectory 
    trajectory_x = sol.y[0]
    trajectory_y = sol.y[1]

    plot_path(x, y, U, V, trajectory_x, trajectory_y)


if __name__ == '__main__':
    x, y, U, V = test_field()
    X, Y = np.meshgrid(x, y)
    trajectory_x, trajectory_y = get_path(U, V, 10, 13, t_max=50)
    plot_path(x, y, U, V, trajectory_x, trajectory_y)
    
    # test_integration()


