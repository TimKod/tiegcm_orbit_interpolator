# TIE-GCM orbit interpolator
This python code provides a class to interpolate TIE-GCM netcdf
model outputs to given time, lat, lon, altitude.
The function "wrt_wgs84" provides an accurate method to interpolate
to input altitude given as geometric altitude wrt WGS-84 reference ellipsoid,
(e.g., satellite orbit coordinates).
Extrapolation on the vertical is True by default with an upper limit of 20 km.
The extrapolation limit (tolerance) is set in function interpalt (alt_tole).
Note: vertical extrapolation not recommended for horizontal winds UN, VN.
Requirements:
toi_constants.py in the current directory.

No installation required. To use in your work,
copy the tiegcm_orbit_interpolator.py and the toi_constants.py file (from src/)
to your working directory. That's all.

TIE-GCM: Thermosphere-Ionosphere-Electrodynamics General Circulation Model
(https://www.hao.ucar.edu/modeling/tgcm/tie.php)

## Example
The function wrt_wgs84 interpolates the fldstr variable found in the TIE-GCM
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
fldstr is string type name of the TIE-GCM variable.
fldstr MUST be variable with 4 dims (time, ilev, lat, lon)

Returns a Pandas dataframe with columns:
Date, Lat, Lon, Height, Tg+"fldstr"

```python
import tiegcm_orbit_interpolator as toi

df = toi.interp_epochs.wrt_wgs84(epochs_df,
                                 ncf, fldstr,
                                 extrapolate=True)
```

## Contact
The code is experimental. If you find any bugs, problems or
have any other comments to improve the code, please send them to me.
Timothy.Kodikara [@] dlr.de
