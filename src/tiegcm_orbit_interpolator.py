'''
tiegcm_orbit_interpolator.py
Author: Timothy Kodikara <Timothy.Kodikara@dlr.de>
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
import netCDF4 as nc
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
# from scipy.interpolate import interp1d
import calendar
import toi_constants as tc
from numba import jit
##


'''
This python code provides a class to interpolate TIE-GCM netcdf
model outputs to given time, lat, lon, altitude.
The function "nc_wrt_wgs84" provides an accurate method to interpolate
to input altitude given as geometric altitude wrt WGS-84 reference ellipsoid,
(e.g., satellite orbit coordinates).
Extrapolation on the vertical is True by default with an upper limit of 20 km.
The extrapolation limit (tolerance) is set in function interpalt (alt_tole).
Note: vertical extrapolation not recommended for horizontal winds UN, VN.
Requirements:
toi_constants.py in the current directory.

The code is experimental. If you find any bugs, problems or
have any other comments to improve the code, please send them to me.

No installation required. To use in your work,
copy the tiegcm_orbit_interpolator.py and the toi_constants.py file (from src/)
to your working directory. That's all.

TIE-GCM: Thermosphere-Ionosphere-Electrodynamics General Circulation Model
(https://www.hao.ucar.edu/modeling/tgcm/tie.php)
'''
##


def toTimestamp(d):
    return calendar.timegm(d.timetuple())
##


def get_txrow(mdf, ix0, ix1):
    txrow_ix = mdf.index[ix0:ix1].to_pydatetime()
    txlist = []
    for t in range(len(txrow_ix)):
        txlist.append(toTimestamp(txrow_ix[t]))
    ##
    txrow_ix = np.array(txlist)
    return txrow_ix
##


def extract_ncvar(ncf, fldstr):
    ''' ncf is netcdf file name
        fldstr is the variable name
        Returns a numpy array of the variable fldstr
    '''
    ncDset = nc.Dataset(ncf)
    Qout = np.array(ncDset.variables.get(fldstr)[:])
    # close ncfile
    ncDset.close()
    ncDset = None
    return Qout
##


def inq_ncvar_dim_names(ncf, fldstr):
    ''' ncf is netcdf file name
        fldstr MUST be variable with 4 dims
    '''
    ncDset = nc.Dataset(ncf)
    (nd1, nd2, nd3, nd4) = getattr(ncDset.variables[fldstr], 'dimensions')
    # close ncfile
    ncDset.close()
    ncDset = None
    return nd1, nd2, nd3, nd4
##


def get_tgtime_df(ncf):
    ''' Get TIEGCM time data
        ncf is a tiegcm netcdf history file.
        Returns a Pandas df with index as the time (in UTC)
    '''
    tg_ncdata = nc.Dataset(ncf)
    # construct tiegcm time array
    # tgst_time = '2014-01-01 00:00:00'
    tgst_time = getattr(tg_ncdata.variables['time'], 'units')
    # close ncfile
    tg_ncdata.close()
    tg_ncdata = None
    ##
    tgst_time = datetime.strptime(tgst_time[14::], '%Y-%m-%d %H:%M:%S')
    tgtime = extract_ncvar(ncf, 'time')
    tgtime = [tgst_time + timedelta(minutes=mx) for mx in tgtime]
    tgtime_df = pd.DataFrame(index=pd.to_datetime(tgtime))
    # clean
    tgtime = None
    tgst_time = None
    ##
    return tgtime_df
##


@jit('float64[:,:](float64[:,:,:,:],float64,float64[:],float64,float64[:])',
     nopython=True, nogil=True)
def bilinear_resample(mvar, inlat, latX, inlon, lonY):
    '''
        bi-linear interpolation
        mvar: 4D numpy array in the shape,
        time, level, lat, lon ==> a1, b1, c1, d1
        latX: -88.75: 2.5: 88.75
        lonY: -180.0: 2.5: 177.5
        inlat, inlon: input lat and lon at which mvar needs resampled.
        xystep: difference between two grid points
        in TIE-GCM, xystep is same for both lat and lon

        Guide: https://archive.org/details/numericalrecipes0865unse/page/123
        Press et al.,
        Numerical Recipes in C : The Art of Scientific Computing, 1992
    '''
    ##
    # make sure inlon agrees with our longitude format: -180:180 range
    inlon = np.mod(inlon - 180.0, 360.0) - 180.0
    xystep = 2.5
    # find latitude corners
    latlen = len(latX)
    lx = (np.abs(latX - inlat)).argmin()
    c1 = lx
    if (latX[lx] >= inlat):
        c2 = lx - 1
        if (c2 < 0):
            c2 = c1
            w_x1 = 0.5
        else:
            w_x1 = np.true_divide(np.abs(inlat - latX[c2]), xystep)
    elif (latX[lx] < inlat):
        c2 = lx + 1
        if (c2 > latlen - 1):
            c2 = c1
            w_x1 = 0.5
        else:
            w_x1 = 1 - np.true_divide(np.abs(inlat - latX[c1]), xystep)
    ##
    w_x2 = 1 - w_x1
    ##
    # find longitude corners
    lonlen = len(lonY)
    ly = (np.abs(lonY - inlon)).argmin()
    d1 = ly
    # handle global wrap around case: -180 == 180
    if (lonY[ly] >= inlon):
        d2 = ly - 1
        if (d2 < 0):
            d2 = lonlen - 1
            w_y1 = np.true_divide(np.abs(lonY[d1] - inlon), xystep)
        else:
            w_y1 = np.true_divide(np.abs(inlon - lonY[d1]), xystep)
    elif (lonY[ly] < inlon):
        d2 = ly + 1
        if (d2 > lonlen - 1):
            d2 = 0
        ##
        w_y1 = np.true_divide(np.abs(inlon - lonY[d1]), xystep)
    ##
    w_y2 = 1 - w_y1
    ##
    blvar = (w_x2 * w_y2 * mvar[:, :, c1, d1]) + (w_x1 * w_y2 * mvar[:, :, c1, d2])
    blvar = blvar + (w_x1 * w_y1 * mvar[:, :, c2, d2]) + (w_x2 * w_y1 * mvar[:, :, c2, d1])
    ##
    return blvar.astype(np.float64)
##


@jit('float64(float64, float64, float64, float64)', nopython=True, cache=True)
def find_normgrav(geoh, phir, lamr, sinlat2):
    '''
        Calculate normal gravity at input coordinates
        An Exact Expression for Normal Gravity, MJ Mahoney, 2001
        geoh = geometric height in [m];
        phir = latitude in radians
        lamr = longitude in radians
        sinlat2  = np.power(np.sin(phir), 2)
    '''
    ##
    c2 = 1 - np.multiply(np.power(tc.E_eccn, 2), sinlat2)
    c2 = np.sqrt(c2)
    Nphi = np.true_divide(tc.semi_a, c2)
    # The Cartesian rectangular coordinates are given
    # in terms of the geodetic coordinates by:
    x = (Nphi + geoh) * np.cos(phir) * np.cos(lamr)
    y = (Nphi + geoh) * np.cos(phir) * np.sin(lamr)
    c2 = (1 - np.power(tc.E_eccn, 2))
    c2 = np.multiply(c2, Nphi) + geoh
    z = np.multiply(c2, np.sin(phir))
    ##
    chiX = x**2 + y**2 + z**2 - tc.E_linear
    # The coordinate u is the semi-major axis of an ellipsoid of revolution,
    # which is confocal with the reference ellipsoid and whose surface
    # passes through the altitude Z of interest.
    c1 = np.multiply(0.5, chiX)
    #
    c2 = 2.0 * tc.E_linear * z
    c2 = np.true_divide(c2, chiX)
    c2 = 1.0 + np.power(c2, 2)
    c2 = 1.0 + np.sqrt(c2)
    #
    c1 = np.multiply(c1, c2)
    uellip = np.sqrt(c1)
    ##
    capU = np.power(uellip, 2) + np.power(tc.E_linear, 2)
    # The coordinate Beta is the reduced altitude
    c1 = np.sqrt(capU)
    c1 = np.multiply(z, c1)
    #
    c2 = x**2 + y**2
    c2 = np.sqrt(c2)
    c2 = np.multiply(uellip, c2)
    #
    c1 = np.true_divide(c1, c2)
    Beta = np.arctan(c1)
    #  capW
    c1 = np.power(np.sin(Beta), 2)
    c1 = np.multiply(np.power(tc.E_linear, 2), c1)
    c1 = np.power(uellip, 2) + c1
    #
    c1 = np.true_divide(c1, capU)
    capW = np.sqrt(c1)
    #  qval
    c1 = np.true_divide(uellip, tc.E_linear)
    c1 = np.power(c1, 2)
    c1 = 1.0 + np.multiply(3.0, c1)
    #
    c2 = np.true_divide(tc.E_linear, uellip)
    c2 = np.arctan(c2)
    #
    c1 = np.multiply(c1, c2)
    #
    c2 = np.true_divide(uellip, tc.E_linear)
    c2 = np.multiply(3.0, c2)
    #
    c1 = c1 - c2
    qval = np.multiply(0.5, c1)
    # q0
    c1 = np.true_divide(tc.semi_b, tc.E_linear)
    c1 = np.power(c1, 2)
    c1 = 1.0 + np.multiply(3.0, c1)
    #
    c2 = np.true_divide(tc.E_linear, tc.semi_b)
    c2 = np.arctan(c2)
    #
    c1 = np.multiply(c1, c2)
    #
    c2 = np.true_divide(tc.semi_b, tc.E_linear)
    c2 = np.multiply(3.0, c2)
    #
    c1 = c1 - c2
    q0 = np.multiply(0.5, c1)
    # q1
    c1 = np.true_divide(uellip, tc.E_linear)
    c1 = 1.0 + np.power(c1, 2)
    c1 = np.multiply(3.0, c1)
    #
    c2 = np.true_divide(tc.E_linear, uellip)
    c2 = np.arctan(c2)
    c2 = 1.0 - np.multiply(np.true_divide(uellip, tc.E_linear), c2)
    #
    c1 = np.multiply(c1, c2)
    q1 = c1 - 1.0
    # *****
    # grav_u
    c1 = np.power(np.sin(Beta), 2)
    c1 = np.multiply(0.5, c1) - np.true_divide(1, 6)
    #
    c2 = np.power(np.multiply(tc.E_omega, tc.semi_a), 2)
    c2 = np.multiply(c2, tc.E_linear)
    c2 = np.true_divide(c2, capU)
    c2 = np.multiply(c2, np.true_divide(q1, q0))
    #
    c1 = np.multiply(c2, c1)
    c1 = np.true_divide(tc.E_GM, capU) + c1
    #
    c2 = np.true_divide(-1.0, capW)
    #
    c1 = np.multiply(c2, c1)
    #
    c2 = np.power(np.cos(Beta), 2)
    c2 = np.multiply(uellip, c2)
    c2 = np.multiply(np.power(tc.E_omega, 2), c2)
    c2 = np.true_divide(c2, capW)
    grav_u = c1 + c2
    # *****
    # grav_beta
    c1 = np.multiply(np.sin(Beta), np.cos(Beta))
    grav_beta = np.true_divide(c1, capW)
    #
    c1 = np.power(np.multiply(tc.E_omega, tc.semi_a), 2)
    c1 = np.true_divide(c1, np.sqrt(capU))
    c1 = np.multiply(c1, np.true_divide(qval, q0))
    #
    c2 = np.power(tc.E_omega, 2)
    c2 = np.multiply(c2, np.sqrt(capU))
    #
    c1 = c1 - c2
    grav_beta = np.multiply(grav_beta, c1)
    # *****
    Ng = np.power(grav_u, 2) + np.power(grav_beta, 2)
    Ng = np.sqrt(Ng)
    return Ng
##


@jit('float64(float64)', nopython=True, cache=True)
def effRE_at_lat(sinlat2):
    '''
        effective radius of Earth at surface relative to
        WGS-84 reference ellipsoid
        effR = tc.semi_a / (1.0 + flatt + grav_ratio - 2.0*flatt*sinlat2)
    '''
    ##
    c1 = tc.effR_c1 - np.multiply(tc.flatt_2, sinlat2)
    effR = np.true_divide(tc.semi_a, c1)  # [m]
    return effR
##


@jit('float64(float64)', nopython=True, cache=True)
def grav_at_lat(sinlat2):
    '''
        Calculate gravity - normal gravity on surface of ellipsoid
        g = equtr_grav*(1.0 + Som_k*sinlat2) / SQRT(1.0 - (E_eccn**2)*sinlat2)
    '''
    ##
    c1 = 1.0 + np.multiply(tc.Som_k, sinlat2)
    c2 = 1 - np.multiply(np.power(tc.E_eccn, 2), sinlat2)
    c2 = np.sqrt(c2)
    gs = np.true_divide(c1, c2)
    gs = np.multiply(tc.equtr_grav, gs)
    return gs
##


@jit('float64(float64, float64, float64, float64)', nopython=True, cache=True)
def potH_at_hgt(geoh, effR, Ng, gs):
    ''' geoh = geometric height in [m]
        effR = effRE_at_lat(sinlat2) [m]
        Ng = find_normgrav() [m/s^2]
        gs = grav_at_lat() [m/s^2]
        Return c1: geopotential height
    '''
    ##
    c1 = np.multiply(effR, geoh)
    c2 = effR + geoh
    #
    c1 = np.true_divide(c1, c2)
    #
    c2 = np.true_divide(Ng, tc.grav_tgcm)
    c1 = np.multiply(c2, c1)
    # print(Ng, c2, c1)
    # empirical correction for gravity offset due to tiegcm's constant gravity
    kc = np.true_divide(tc.grav_tgcm, gs)
    c1 = np.true_divide(c1, kc)
    # print("new c1, kc, gs: ", c1, kc, gs)
    return c1
##


@jit('float64[:](float64[:,:], float64[:,:], float64, int32, int32, boolean)',
     nopython=True, cache=True)
def interpalt(tgZ, mvar, tohgt, lev_id, Qln, extrap):
    ''' tgZ: tiegcm height array [cm]
        mvar: Q1 array
        tohgt: height to interpolate to [m]
        lev_id: whether Q1 is on lev or ilev
        lev_id=100 (ilev)
        lev_id=111 (lev)
        extrap=True
        vertical extrapolation not recommended for horizontal winds UN, VN
    '''
    ##
    # Convert tgZ to meters
    tgZ = np.true_divide(tgZ, 100)
    # Interpolate to tohgt -- linear
    (a2, b2) = mvar.shape
    blvar = np.zeros(a2)
    # extrapolate tolerance: 20 km
    alt_tole = 20*1000.0
    # Determine whether mvar needs to be interpolated in log-space
    # for now this is handled via a simple pre-determined Qln
    # Qln = 900 means convert to log (e.g., "DEN")
    if (Qln == 900):
        mvar = np.log(mvar)
    ##
    for i in range(a2):
        zg_x = np.copy(tgZ[i, :])
        # Interpolate Z to midpoints for "lev" variables
        # e.g., TI,TE,TN,UN,VN,O1,O2,HE
        if (lev_id == 111):
            zg_x = 0.5 * (zg_x[0:-1] + zg_x[1:None])
            va_y = np.copy(mvar[i, 0:-1])
            # print(zg_x.shape, va_y.shape)
        else:
            va_y = np.copy(mvar[i, :])
        # print(zg_x.shape, va_y.shape)
        ##
        # extrapolate only above model's top level
        # no extrapolation below the bottom level
        if (tohgt < zg_x.min()):
            blvar[i] = np.nan
            continue
        # Scipy interp1d is not supported by numba-jit
        # f=interp1d(zg_x,np.log(va_y),kind='linear',fill_value='extrapolate')
        # f = interp1d(zg_x, va_y, kind='linear', fill_value='extrapolate')
        alt_diff = tohgt - zg_x.max()
        if extrap:
            if (alt_diff > alt_tole):
                # No extrapolation above zg_x.max() + alt_tole
                blvar[i] = np.nan
            elif (alt_diff > 0) and (alt_diff <= alt_tole):
                # Linearly extrapolate to tohgt
                # Ne vs altitude profile is complicated. Instead of the full
                # profile use only the last few values for the linear fit.
                Z_x = zg_x[-4:]
                A_x = np.vstack((Z_x, np.ones(4)))
                A_x = A_x.T
                V_y = va_y[-4:]
                ##
                M = np.linalg.lstsq(A_x, V_y, rcond=-1)
                m1 = M[0][0]
                c1 = M[0][1]
                blvar[i] = m1*tohgt + c1  # y = mx + c
            else:
                # tohgt is within zg_x range. linearly interpolate.
                blvar[i] = np.interp(tohgt, zg_x, va_y)
        if not extrap:
            if (alt_diff > 0):
                # extrapolation not requested
                blvar[i] = np.nan
            else:
                # tohgt is within zg_x range. linearly interpolate.
                blvar[i] = np.interp(tohgt, zg_x, va_y)
        ##
    ##
    if (Qln == 900):
        blvar = np.exp(blvar)
    ##
    return blvar.astype(np.float64)
##


def nc_process_loop(epochs_df, Mod_df, cur_ncf, Q1, extrap):
    # a separate function would be useful here to determine
    # quantities that need to be interpolated in log-space.
    # For now add the desired Qs to the list logQs
    logQs = ["DEN"]
    if (Q1 in logQs):
        Qln = np.int32(900)
    else:
        Qln = np.int32(899)
    ##
    # Load the model quantities
    tgQ1 = extract_ncvar(cur_ncf, Q1)
    tgQ2 = extract_ncvar(cur_ncf, "Z")  # Z is GeoPOTENTIAL height
    tglat = np.float64(extract_ncvar(cur_ncf, 'lat'))
    tglon = np.float64(extract_ncvar(cur_ncf, 'lon'))
    ##
    (_, Q1lev, _, _) = inq_ncvar_dim_names(cur_ncf, Q1)
    if (Q1lev == "ilev"):
        Q1lev = np.int32(100)
    elif (Q1lev == "lev"):
        Q1lev = np.int32(111)
    else:
        print(Q1lev)
        print("Q1 level is not as expected.")
        raise SystemExit
    ##
    # store dimensions locally
    (aa, bb, cc, dd) = tgQ1.shape  # time, ilev, lat, lon
    ##
    # Sanity check dimensions
    if (tgQ1.shape != tgQ2.shape):
        print("shape mismatch tgQ1 tgQ2: ", tgQ1.shape, " vs ", tgQ2.shape)
        raise SystemExit
    # time
    # print("model time length: ", len(Mod_df.index), tgQ1.shape)
    if (len(Mod_df.index) != aa):
        print("time length mismatch: ", len(Mod_df.index), " vs ", aa)
        raise SystemExit
    # lat
    if (len(tglat) != cc):
        print("Lat length mismatch: ", len(tglat), " vs ", cc)
        raise SystemExit
    # lon
    if (len(tglon) != dd):
        print("Lon length mismatch: ", len(tglon), " vs ", dd)
        raise SystemExit
    ##
    # now we should have everything ready to start the loop
    postlist = []
    nr = len(epochs_df.index)
    print(f'num of epochs: {nr}, current file: {cur_ncf[-6:]}')
    ##
    for i in range(nr):
        # coordinates
        inlat = epochs_df['lat'][i]
        inlon = epochs_df['lon'][i]
        inalt = epochs_df['height'][i]
        # lat and lon in radians
        latrad = np.deg2rad(inlat)
        lonrad = np.deg2rad(inlon)
        sphi2 = np.power(np.sin(latrad), 2)
        # A dict for this epoch
        psdict = {'Date': epochs_df.index[i], 'Lat': inlat,
                  'Lon': inlon, 'Height': inalt}
        ##
        # check time to continue
        pivot = epochs_df.index[i].to_pydatetime()
        tx = Mod_df.index.get_indexer([pivot], method='nearest')
        tx = int(tx[0])
        tx1 = Mod_df.index[tx].to_pydatetime()
        tdel = (abs(tx1 - pivot)).total_seconds()
        if (tdel > 600):
            # No interpolation on the time axis for tdel greater
            # than 10 mins (arbitrary decision)
            print('time diff above 10 min')
            print(f'setting Tg{Q1} to nan for epoch {epochs_df.index[i]}')
            # set to nan.
            psdict["Tg"+Q1] = np.nan
            postlist.append(psdict)
            psdict = None
            continue
        ##
        # Time Index to trim model
        if (tx == aa-1):
            tm0 = tx - 1
            tm1 = None
        elif (tx == 0):
            tm0 = tx
            tm1 = tm0 + 2
        else:
            tm0 = tx - 1
            tm1 = tm0 + 2
        ##
        # txrow for the time-interpolation
        pivot = toTimestamp(pivot)
        txrow = get_txrow(Mod_df, tm0, tm1)
        ##
        # limit model time row (4D: time, lev, lat, lon)
        iq2_ps = np.float64(np.copy(tgQ2[tm0:tm1, :, :, :]))
        iq1_ps = np.float64(np.copy(tgQ1[tm0:tm1, :, :, :]))
        # Trim and Interpolate on lat-lon
        # bilinear_resample(mvar, inlat, latX, inlon, lonY)
        iq2_ps = bilinear_resample(iq2_ps, inlat, tglat,
                                   inlon, tglon)
        iq1_ps = bilinear_resample(iq1_ps, inlat, tglat,
                                   inlon, tglon)
        ##
        # Find the geopotential height at inalt
        ##
        # First, calculate gravity normal at input coordinates
        grav_norm = find_normgrav(inalt, latrad, lonrad, sphi2)
        # Effective radius at inlat at the surface
        Rphi = effRE_at_lat(sphi2)
        # Gravity on the surface at inlat
        gsphi = grav_at_lat(sphi2)
        # Calculate geopotential height at inalt
        potHi = potH_at_hgt(inalt, Rphi, grav_norm, gsphi)
        # Interpolate on the vertical
        # interpalt(Z, mvar, tohgt, lbl, Qln, extrap)
        # Here, use potHi as the inalt
        iq1_ps = interpalt(iq2_ps, iq1_ps, potHi,
                           Q1lev, Qln, extrap)
        ##
        # Interpolate to current time
        # ynew = np.interp(xnew, x, y)
        iq1_ps = np.interp(pivot, txrow, iq1_ps)
        # Append to Dictionary
        # iq1_ps is in model units
        psdict["Tg"+Q1] = iq1_ps
        # Append to df list
        postlist.append(psdict)
        psdict = None
        #
        if i % 500 == 0:
            print(i, nr, Q1)
    # prepare tmp dfs
    odf = pd.DataFrame(postlist)
    postlist = None
    ##
    return odf
##


def out_emptydf(epochs_df, fldstr):
    odf = epochs_df.copy(deep=True)
    odf['Date'] = epochs_df.index
    odf.reset_index(drop=True, inplace=True)
    new_cols = {'lat': 'Lat', 'lon': 'Lon', 'height': 'Height'}
    odf.rename(columns=new_cols, inplace=True)
    ##
    modqty = "Tg"+fldstr
    odf[modqty] = np.nan
    return odf
##


class interp_epochs:
    def nc_wrt_wgs84(epochs_df, ncf, fldstr, extrapolate=True):
        ''' This code interpolates the fldstr variable found in the tiegcm
            ncf file to the locations given in epochs_df.
            Input Conditions:
            epochs_df is a Pandas dataframe with datetime as the index
            epochs_df has 3 columns with exactly these names and units:
            epochs_df['lat'] = geographic latitude (-south, +north) [degrees]
            epochs_df['lon'] = geographic longitude (-west, +east) [degrees]
            epochs_df['height'] = geometric altitude
            with WGS-84 reference ellipsoid [m]

            ncf is a TIEGCM NetCDF history file and no more than one day's
            data is stored in a single file.
            fldstr MUST be variable with 4 dims (time, ilev, lat, lon)

            Returns a Pandas dataframe with columns:
            Date, Lat, Lon, Height, Tg+"fldstr"

            USE this with a single nc file
            e.g., parallel call Long_runs/GPI* individual files
        '''
        ##
        tgtime_df = get_tgtime_df(ncf)
        # screen
        refday = tgtime_df.index[0].strftime('%Y-%m-%d')
        try:
            epochs_df = epochs_df.loc[refday].copy(deep=True)
            # Call the MAIN Interpolation loop
            outdf = nc_process_loop(epochs_df,
                                    tgtime_df,
                                    ncf, fldstr,
                                    extrap=extrapolate)
        except KeyError:
            # if epochs_df has no matching data for this day,
            # then we leave with a nan df
            outdf = out_emptydf(epochs_df, fldstr)
        ##
        return outdf
