import os
import glob
import datetime
import numpy as np
import xarray as xr
import pandas as pd
from sklearn.cluster import DBSCAN
from pyhdf.SD import SD
import matplotlib.pyplot as plt

def cluster_haversine(dfr):
    db = DBSCAN(eps=0.8/6371., min_samples=2, algorithm='ball_tree', 
            metric='haversine', n_jobs=-1).fit(np.radians(dfr[['lat', 'lon']]))
    return db.labels_

def cluster_euc(XYZ, dates, space_eps):
    XYZT = np.column_stack((XYZ, dates * (space_eps / 2.5))
    db = DBSCAN(eps=space_eps, min_samples=2, algorithm='ball_tree', 
            metric='euclidean', n_jobs=-1).fit(XYZT)
    return db.labels_


def get_tile_ref(fname):
    hv = os.path.basename(fname).split('.')[-4]
    tile_h = int(hv[1:3])
    tile_v = int(hv[4:6])
    return tile_v, tile_h

def spher_to_cartes(lon_rad, lat_rad):
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    return np.column_stack((x,y,z))



class FireObs(object):

    def __init__(self, data_path, store_name, bbox=None, hour=None):
        self.data_path = data_path
        self.store_name =  store_name
        self.bbox = bbox
        self.hour = hour
        self.tile_size = 1111950 # height and width of MODIS tile in the projection plane (m)
        self.x_min = -20015109 # the western linit ot the projection plane (m)
        self.y_max = 10007555 # the northern linit ot the projection plane (m)
        self.w_size = 463.31271653 # the actual size of a "500-m" MODIS sinusoidal grid cell
        self.earth_r = 6371007.181 # the radius of the idealized sphere representing the Earth

        self.regions_bounds = {'Am_tr': [-113, 31.5, -3.5, -55],
                               'Af_tr': [-18, 22.5, 52, -35],
                               'As_tr': [59.5, 40, 155.5, -30.5]}

    def pixel_lon_lat(self, tile_v, tile_h, idi, idj):
        """
        A method to calculate pixel lon lat, based on the MCD64A1 ATBD formulas (Giglio)
        """
        # positions of centres of the grid cells on the global sinusoidal grid
        x_pos = ((idj + 0.5) * self.w_size) + (tile_h * self.tile_size) + self.x_min
        y_pos = self.y_max - ((idi + 0.5) * self.w_size) - (tile_v * self.tile_size)
        # and then lon lat
        lat = y_pos / self.earth_r
        lon = x_pos / (self.earth_r * np.cos(lat))
        return np.rad2deg(lon), np.rad2deg(lat)

    def read_hdf4(self, file_name, dataset=None):
        """
        Reads Scientific Data Set(s) stored in a HDF-EOS (HDF4) file
        defined by the file_name argument. Returns SDS(s) given
        name string provided by dataset argument. If
        no dataset is given, the function returns pyhdf
        SD instance of the HDF-EOS file open in read mode.
        """
        dataset_path = os.path.join(self.data_path, file_name)
        try:
            product = SD(dataset_path)
            if dataset == 'all':
                dataset = list(product.datasets().keys())
            if isinstance(dataset, list):
                datasetList = []
                for sds in dataset:
                    selection = product.select(sds).get()
                    datasetList.append(selection)
                return datasetList
            elif dataset:
                selection = product.select(dataset).get()
                return selection
            return product
        except IOError as exc:
            print('Could not read dataset {0}'.format(file_name))
            raise

    def spatial_subset(self, dataset, bbox):
        """
        Selects data within spatial bbox. bbox coords must be given as
        positive values for the Northern hemisphere, and negative for
        Southern. West and East both positive - Note - the method is
        naive and will only work for bboxes fully fitting in the Eastern hemisphere!!!
        Args:
            dataset - xarray dataset
            bbox - (list) [North, South, West, East]
        Returns:
            xarray dataset
        """
        lat_name = [x for x in list(dataset.coords) if 'lat' in x]
        lon_name = [x for x in list(dataset.coords) if 'lon' in x]
        dataset = dataset.where((dataset[lat_name[0]] < bbox[0]) &
                                (dataset[lat_name[0]] > bbox[1]), drop=True)
        dataset = dataset.where((dataset[lon_name[0]] > bbox[2]) &
                                (dataset[lon_name[0]] < bbox[3]), drop=True)
        return dataset

    def time_subset(self, dataset, hour=None, start_date=None, end_date=None):
        """
        Selects data within spatial bbox.
        Args:
            dataset - xarray dataset
            hour - (int) hour
        Returns:
            xarray dataset
        """
        if hour:
            dataset = dataset.sel(time=datetime.time(hour))
        return dataset

    def get_file_names(self, fname_dir, extension=None):
        if extension:
            pattern = '*.{0}'.format(extension)
        else:
            pattern = '*.*'
        fnames = glob.glob(os.path.join(fname_dir, pattern))
        return fnames

    def check_if_ba(self, ds):
        """
        Check if dataset contains burned pixels
        """
        burned_cells = ds.attributes()['BurnedCells']
        return burned_cells

    def ba_to_dataframe(self, ds, tile_v, tile_h):
        ba_date = ds.select('Burn Date').get()
        ba_indy, ba_indx = np.where(ba_date > 0)
        ba_date = ba_date[ba_indy, ba_indx]
        ba_unc = ds.select('Burn Date Uncertainty').get()[ba_indy, ba_indx]
        ba_qa = ds.select('QA').get()[ba_indy, ba_indx]
        ba_first = ds.select('First Day').get()[ba_indy, ba_indx]
        ba_last = ds.select('Last Day').get()[ba_indy, ba_indx]
        lons, lats = self.pixel_lon_lat(tile_v, tile_h, ba_indy, ba_indx)
        dfr = pd.DataFrame({'date': ba_date,
                            'unc': ba_unc,
                            'qa': ba_qa,
                            #'first_date': ba_first,
                            #'last_date': ba_last,
                            'lon': lons,
                            'lat': lats})
        #selecting only qa == 3 (8 bit field 11000000, 
        #indicating land pixels (first bit) and valid data (second bit) refer to ATBD)
        dfr = dfr[dfr['qa'] == 3]
        dfr.drop('qa', axis=1, inplace=True)
        return dfr

    def read_ba_year(self, year):
        fnames = self.get_file_names(os.path.join(self.data_path, str(year)), 'hdf')
        fr_list = []
        for nr, fname in enumerate(fnames):
            ds = self.read_hdf4(fname)
            burned_cells = ds.attributes()['BurnedCells']
            if burned_cells:
                print(nr)
                tile_v, tile_h = get_tile_ref(fname)
                dfr = self.ba_to_dataframe(ds, tile_v, tile_h)
                fr_list.append(dfr)
        fr_all = pd.concat(fr_list)
        fr_all.reset_index(inplace=True)
        fr_all.loc[:, 'year'] = year
        return fr_all


 
    def populate_store(self):
        years = list(range(2002, 2016))
        for year in years:
            dfr = self.read_ba_year(year)
            dfr.to_hdf(self.store_name, key='ba', format='table', 
                       data_columns=['year', 'date', 'lon', 'lat'], append=True)


    def populate_store_tropics(self, tropics_store_name):
        years = list(range(2002, 2016))
        for year in years:
            dfr = pd.read_hdf(self.store_name, 'ba', where="year=={0}".format(str(year)))
            for reg in self.regions_bounds.keys():
                bbox = self.regions_bounds[reg]
                sel_dfr = dfr[(dfr.lon > bbox[0])&(dfr.lon < bbox[2])]
                sel_dfr = sel_dfr[(sel_dfr.lon < bbox[1])&(sel_dfr.lon > bbox[3])]
                sel_dfr.to_hdf(tropics_store_name, key=reg, format='table', 
                       data_columns=['year', 'date', 'lon', 'lat'], append=True)



    def select_ba(self, selection):
        pass



       
if __name__ == '__main__':
    data_path = '.'
    #data_path = '/mnt/data/area_burned_glob'
    #store_name = 'ba_store.h5'
    store_name = 'ba_tropics_store.h5'
    #tropics_store = 'ba_tropics_store.h5'
    ba = FireObs(data_path, os.path.join(data_path, store_name))
    #ba.populate_store_tropics(tropics_store)
    #ba.populate_store()



