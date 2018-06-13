import numpy as np
import xarray as xr
import pandas as pd

class Gridder(object):
    def __init__(self, step):
        self.step = step
        self.grid_bins()

    def grid_bins(self):
        self.lon_bins = np.arange(-180, (180 + self.step / 2.0), self.step)
        self.lat_bins = np.arange(-90, (90 + self.step / 2.0), self.step)
 
    def binning(self, lon, lat):
        """
        Get indices of the global grid bins for the longitudes and latitudes
        of observations stored in frpFrame pandas DataFrame. Must have 'lon' and 'lat'
        columns.

        Arguments:
            lon : np.array, representing unprojected longitude coordinates.
            lat : np.array, representing unprojected longitude coordinates.

        Returns:
            Raises TypeError if frpFrame is not a pandas DataFrame
            frpFrame : pandas DataFrame
                Same DataFrame with added columns storing positional indices
                in the global grid defined in grid_bins method
        """
        lonind = np.digitize(lon, self.lon_bins) - 1
        latind = np.digitize(lat, self.lat_bins) - 1
        return lonind, latind

    def to_grid(self, dfr):
        lonind, latind = self.binning(dfr['lon'].values, dfr['lat'].values)
        dfr.loc[:, 'lonind'] = lonind
        dfr.loc[:, 'latind'] = latind
        ignitions = np.zeros((self.lat_bins.shape[0] - 1,
                         self.lon_bins.shape[0] - 1))
        grouped = pd.DataFrame({'ign_count' : dfr.groupby(['lonind', 'latind']).size()}).reset_index()
        latinds = grouped['latind'].values.astype(int)
        loninds = grouped['lonind'].values.astype(int)
        ignitions[latinds, loninds] = grouped['ign_count']
        ignitions = np.flipud(ignitions)
        return ignitions

    def to_netcdf(self, data):
        lats = np.arange((-90 + self.step / 2.), 90., self.step)[::-1]
        lons = np.arange((-180 + self.step / 2.), 180., self.step)
        dataset = xr.Dataset({'ignitions': (['latitude', 'longitude'], data)},
                              coords={'latitude': lats,
                                     'longitude': lons})
        return dataset
 
