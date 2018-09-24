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

    def to_xarray(self, data, var_name, timestamps):
        lats = np.arange((-90 + self.step / 2.), 90., self.step)[::-1]
        lons = np.arange((-180 + self.step / 2.), 180., self.step)
        dataset = xr.Dataset({var_name: (['latitude', 'longitude', 'date'], data)},
                              coords={'latitude': lats,
                                     'longitude': lons,
                                     'date': timestamps})
        return dataset

    def grid_centroids_all(self, dfr_list):
        dsl = []
        dates = pd.date_range('2002-01-01', periods=dfr_list[0].day_since.max(), freq='d')
        for nr, dur in enumerate(['2', '4', '8', '16']):
            print(dur)
            dfr = dfr_list[nr]
            dfr.loc[:, 'date'] = dates[dfr.day_since-1]
            grouped_days = dfr.groupby('date')
            grids = [self.to_grid(x[1]) for x in grouped_days]
            dataset = xr.Dataset({'ignitions{0}'.format(dur): (['latitude', 'longitude', 'date'], np.dstack(grids))},
                                  coords={'latitude': lats,
                                         'longitude': lons,
                                         'date': dates})
            dsl.append(dataset)
            return dsl


    def grid_centroids(self, years, dfr_list):
        for year in years:
            dsy = []
            for nr, dur in enumerate(['2', '4', '8', '16']):
                dfr = dfr_list[nr]
                gr_year = dfr[dfr.year == year]
                grouped_days = gr_year.groupby('date')
                grids = [self.to_grid(x[1]) for x in grouped_days]
                timestamps = [x[0] for x in grouped_days]
                netcdf_store = self.to_xarray(np.dstack(grids), 'ign_agg_{0}'.format(dur), timestamps)
                dsy.append(netcdf_store)
            dsa = xr.merge(dsy)
            dsa.to_netcdf('/mnt/data/frp/ignitions_tropics_{0}_frp.nc'.format(year),
                          encoding={'ign_agg_2': {'dtype': 'int16', 'zlib': True},
                                    'ign_agg_4': {'dtype': 'int16', 'zlib': True},
                                    'ign_agg_8': {'dtype': 'int16', 'zlib': True},
                                    'ign_agg_16': {'dtype': 'int16', 'zlib': True}})
