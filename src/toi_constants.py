'''
toi_constants.py
Author: Timothy Kodikara <Timothy.Kodikara  [@]  dlr.de>
LICENSE: CC BY-NC-SA 4.0 https://creativecommons.org/licenses/by-nc-sa/4.0/

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
'''

##
import numpy as np
##

'''
This module defines some constants and relevant parameters used by
tiegcm_orbit_interpolator.py
'''
# **********
semi_a = 6378137.0     # [m] Semi-major axis (a): 6378.1370 km
semi_b = 6356752.3142  # [m] Semi-minor axis (b): 6356.7523142 km
# flattening = (a - b) / a
flatt = (semi_a - semi_b) / semi_a  # 0.003352811
flatt_2 = 2.0*flatt
# Linear Eccentricity
E_linear = np.sqrt(semi_a**2 - semi_b**2)
# Eccentricity = sqrt(a^2 - b^2)/a = 0.081819
E_eccn = np.sqrt(semi_a**2 - semi_b**2) / semi_a
polar_grav = 9.8321849378  # [m s^-2]  # Polar gravity
equtr_grav = 9.7803253359  # [m s^-2]  # Equatorial gravity
# Somigliana's Constant ks = (b/a)*(polar_g/equtr_g) - 1 = 1.931853 x 10-3
Som_k = (semi_b/semi_a) * (polar_grav/equtr_grav) - 1
# Angular velocity of Earth (w): 7292115.0 x 10-11 rad/s
E_omega = 7292115.0e-11  # [rad/s]
# Earth's Gravitational Constant (Atmosphere incl.)
# (GM): 3986004.418 x 10^8 m3/s 2
E_GM = 3986004.418e8  # [m^3/s^2]
# to find effective radius of Earth relative to WGS-84 reference ellipsoid
# Gravity ratio = (w^2 * a^2 * b)/GM = 0.003449787
effR_c1 = ((E_omega**2)*(semi_a**2)*semi_b) / E_GM
effR_c1 = 1.0 + flatt + effR_c1
# tiegcm constant gravity
grav_tgcm = 8.7   # [m s^-2]
##
