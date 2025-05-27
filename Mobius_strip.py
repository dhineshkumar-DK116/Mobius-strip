import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class MobiusStrip:
    """
    A class to model a Mobius strip using parametric equations and compute
    its key geometric properties.
    """

    def __init__(self, R: float, w: float, n: int):
        """
        Initializes the MobiusStrip object.

        Args:
            R (float): Radius R (distance from the center to the strip).
            w (float): Width w (strip width).
            n (int): Resolution n (number of points in the mesh along each parameter).
        """
        if not all(isinstance(arg, (int, float)) and arg > 0 for arg in [R, w, n]):
            raise ValueError("R, w, and n must be positive numerical values.")
        if not isinstance(n, int):
            raise TypeError("n must be an integer.")

        self.R = float(R)
        self.w = float(w)
        self.n = int(n)

        self.u_vals = np.linspace(0, 2 * np.pi, self.n)
        self.v_vals = np.linspace(-self.w / 2, self.w / 2, self.n)
        self.U, self.V = np.meshgrid(self.u_vals, self.v_vals)

        self._x = None
        self._y = None
        self._z = None
        self._surface_area = None
        self._edge_length = None

    def _compute_coordinates(self):
        """
        Computes the (x, y, z) coordinates of the Mobius strip surface.
        """
        if self._x is None:
            self._x = (self.R + self.V * np.cos(self.U / 2)) * np.cos(self.U)
            self._y = (self.R + self.V * np.cos(self.U / 2)) * np.sin(self.U)
            self._z = self.V * np.sin(self.U / 2)

    @property
    def mesh_points(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns the 3D mesh/grid of (x, y, z) points on the surface.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: x, y, and z coordinate arrays.
        """
        self._compute_coordinates()
        return self._x, self._y, self._z

    def _compute_partial_derivatives(self):
        """
        Computes the partial derivatives of the parametric equations with respect to u and v.
        Returns:
            tuple: du_x, du_y, du_z, dv_x, dv_y, dv_z
        """
        # Partial derivatives with respect to u
        du_x = -(self.R + self.V * np.cos(self.U / 2)) * np.sin(self.U) - \
               (self.V / 2) * np.sin(self.U / 2) * np.cos(self.U)
        du_y = (self.R + self.V * np.cos(self.U / 2)) * np.cos(self.U) - \
               (self.V / 2) * np.sin(self.U / 2) * np.sin(self.U)
        du_z = (self.V / 2) * np.cos(self.U / 2)

        # Partial derivatives with respect to v
        dv_x = np.cos(self.U / 2) * np.cos(self.U)
        dv_y = np.cos(self.U / 2) * np.sin(self.U)
        dv_z = np.sin(self.U / 2)
        return du_x, du_y, du_z, dv_x, dv_y, dv_z

    @property
    def surface_area(self) -> float:
        """
        Computes the surface area of the Mobius strip numerically using integration
        (approximation based on the magnitude of the cross product of partial derivatives).

        Returns:
            float: The approximated surface area.
        """
        if self._surface_area is None:
            du_x, du_y, du_z, dv_x, dv_y, dv_z = self._compute_partial_derivatives()

            # Compute the cross product of the partial derivative vectors
            # Normal vector components: N_x, N_y, N_z
            N_x = du_y * dv_z - du_z * dv_y
            N_y = du_z * dv_x - du_x * dv_z
            N_z = du_x * dv_y - du_y * dv_x

            # Magnitude of the normal vector
            magnitude_N = np.sqrt(N_x**2 + N_y**2 + N_z**2)

            # Area element dA = ||r_u x r_v|| du dv
            # Using numerical integration (sum of magnitudes multiplied by differential areas)
            delta_u = self.u_vals[1] - self.u_vals[0]
            delta_v = self.v_vals[1] - self.v_vals[0]
            self._surface_area = np.sum(magnitude_N * delta_u * delta_v)
        return self._surface_area

    @property
    def edge_length(self) -> float:
        """
        Computes the total length of the single edge of the Mobius strip numerically.
        The edge of the Mobius strip is where v = +/- w/2. Due to the twist, these
        two edges meet to form a single continuous loop. We can compute the length
        along one of these parameter lines (e.g., v = w/2) and the other will be the same.

        Returns:
            float: The approximated edge length.
        """
        if self._edge_length is None:
            # We will trace the edge where v = w/2
            v_edge = self.w / 2
            u_vals_dense = np.linspace(0, 2 * np.pi, 2 * self.n) # Higher resolution for edge
            
            x_edge = (self.R + v_edge * np.cos(u_vals_dense / 2)) * np.cos(u_vals_dense)
            y_edge = (self.R + v_edge * np.cos(u_vals_dense / 2)) * np.sin(u_vals_dense)
            z_edge = v_edge * np.sin(u_vals_dense / 2)

            # Calculate infinitesimal arc length ds = sqrt((dx/du)^2 + (dy/du)^2 + (dz/du)^2) du
            dx_du = np.gradient(x_edge, u_vals_dense)
            dy_du = np.gradient(y_edge, u_vals_dense)
            dz_du = np.gradient(z_edge, u_vals_dense)

            ds = np.sqrt(dx_du**2 + dy_du**2 + dz_du**2)
            self._edge_length = np.trapz(ds, u_vals_dense) # Use trapezoidal rule for integration
        return self._edge_length

    def plot_strip(self):
        """
        Generates a 3D plot of the Mobius strip.
        """
        x, y, z = self.mesh_points

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        ax.plot_surface(x, y, z, cmap='viridis', edgecolor='none', alpha=0.8)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Mobius Strip (R={self.R}, w={self.w})')
        plt.show()

# Example Usage
if __name__ == "__main__":
    mobius = MobiusStrip(R=1.0, w=0.4, n=300)
    area = mobius.surface_area
    print(f"Approximated Surface Area: {area:.4f}")
    length = mobius.edge_length
    print(f"Approximated Edge Length: {length:.4f}")
    mobius.plot_strip()
