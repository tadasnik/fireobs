import os
import time
import datetime
import numpy as np
import xarray as xr
import pandas as pd
from sklearn.cluster import DBSCAN
from pyhdf.SD import SD
from gridding import Gridder


def cluster_haversine(dfr):
    db = DBSCAN(eps=0.8/6371., min_samples=2, algorithm='ball_tree', 
            metric='haversine', n_jobs=-1).fit(np.radians(dfr[['lat', 'lon']]))
    return db.labels_

def cluster_euc(xyzt, eps, min_samples):
    #divide space eps by approximate Earth radius in km to get eps in radians.
    #xyzt = np.column_stack((xyz, dates * (eps / time_eps)))
    db = DBSCAN(eps=eps, min_samples=min_samples, algorithm='ball_tree', 
            metric='euclidean', n_jobs=-1).fit(xyzt)
    return db.labels_

def lon_lat_to_spherical(dfr):
    lon_rad, lat_rad = np.deg2rad(dfr.lon), np.deg2rad(dfr.lat)
    xyz = spher_to_cartes(lon_rad, lat_rad)
    return xyz

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

def get_days_since(dfr):
    basedate = pd.Timestamp('2002-01-01')
    dates = pd.to_datetime(dfr.year, format='%Y') + pd.to_timedelta(dfr.date, unit='d')
    dfr.loc[:, 'day_since'] = (dates - basedate).dt.days
    return dfr#(dates - basedate).dt.days

def find_ignitions_naive(dfr):
    df = dfr[dfr.day_since==dfr.day_since.min()]
    centroids = df.groupby(['labs1']).agg({'lon':'mean', 'lat':'mean'})
    return centroids

def find_ignitions(dfr):
    days = dfr.day_since.unique()
    days[::-1].sort()
    obj_cs = {}
    for nr, day in enumerate(days[1:]):
        print(day)
        dr = dfr[dfr.day_since <= day]
        labs = cluster_euc(dr[['x', 'y', 'z']], dr.day_since, 1, 0.7)# make this class method, self.eps!
        for lab in np.unique(labs):
            if len(labs[labs==lab] == 1):
                obj_cs[str(lab)] = [dr[labs==lab]['lon'].mean(), dr[labs==lab]['lat'].mean()]
                obj_cs[str(lab)] = [dr[labs==lab]['lon'].mean(), dr[labs==lab]['lat'].mean()]
        

def add_xyz(dfr):
    xyz = lon_lat_to_spherical(dfr)
    dfr.loc[:, 'x'] = xyz[:,0]
    dfr.loc[:, 'y'] = xyz[:,1]
    dfr.loc[:, 'z'] = xyz[:,2]
    return dfr

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
        self.years = list(range(2002, 2016))
        #DBSCAN eps in radians = 650 meters / earth radius
        self.eps = 750 / self.earth_r
        self.basedate = pd.Timestamp('2002-01-01')


        self.regions_bounds = {'Am_tr': [-113, 31.5, -3.5, -55],
                               'Af_tr': [-18, 22.5, 52, -35],
                               'As_tr': [59.5, 40, 155.5, -30.5]}

    def pixel_lon_lat(self, tile_v, tile_h, idi, idj):
        """
        A method to calculate pixel lon lat, using the formulas 
        given in the MCD64A1 product ATBD document (Giglio)
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

    def to_day_since(self, dtime_string):
        """
        Method returning day since the self base date. Takes string datetime in
        YYYY-MM-DD format.
        """
        dtime = pd.to_datetime(dtime_string, format='%Y-%m-%d')
        return (dtime - self.basedate).days


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
        for year in self.years:
            dfr = self.read_ba_year(year)
            dfr.to_hdf(self.store_name, key='ba', format='table', 
                       data_columns=['year', 'date', 'lon', 'lat'], append=True)


    def populate_store_tropics(self, store_name):
        for year in self.years:
            dfr = pd.read_hdf(self.store_name, 'ba', where="year=={0}".format(str(year)))
            dfr = self.get_days_since(dfr)
            for reg in self.regions_bounds.keys():
                bbox = self.regions_bounds[reg]
                sel_dfr = dfr[(dfr.lon > bbox[0])&(dfr.lon < bbox[2])]
                sel_dfr = sel_dfr[(sel_dfr.lat < bbox[1])&(sel_dfr.lat > bbox[3])]
                sel_dfr.to_hdf(tropics_store_name, key=reg, format='table', 
                       data_columns=['year', 'date', 'day_since', 'lon', 'lat'], append=True)

    def populate_store_af_blocks(self, store_name, tropics_store_name):
        for year in self.years[:-1]:
            block_strings = self.block_strings(year)
            dfr = [pd.read_hdf(store_name, 'ba', where=x) for x in block_strings]
            dfr = pd.concat(dfr)
            reg = 'Af_tr'
            bbox = self.regions_bounds[reg]
            dfr = dfr[(dfr.lon > bbox[0])&(dfr.lon < bbox[2])]
            dfr = dfr[(dfr.lat < bbox[1])&(dfr.lat > bbox[3])]
            dfr = get_days_since(dfr)
            dfr = add_xyz(dfr)
            dfr.sort_values(by='day_since', inplace=True)
            dfr.to_hdf(tropics_store_name, key=reg+'/block_{0}'.format(year), mode='r+', format='table', 
                       data_columns=['day_since'], append=True)



    def select_ba(self, selection):
        pass

    def lon_lat_to_spherical(self, dfr):
        lon_rad, lat_rad = np.deg2rad(dfr.lon), np.deg2rad(dfr.lat)
        xyz = spher_to_cartes(lon_rad, lat_rad)
        return xyz

    def write_xyz_day_since(self, store_objects):
        for obj in store_objects:
            print(obj)
            dfr = pd.read_hdf(ba.store_name, obj)
            dfr = add_xyz(dfr)
            dfr[['x', 'y', 'z']].to_hdf(store_name, key='{0}/xyz'.format(obj), append=True)
            #dfr.drop(columns=['x', 'y', 'z'])
            dfr = get_days_since(dfr)
            dfr['day_since'].to_hdf(store_name, key='{0}/day_since'.format(obj), append=True)

    def block_strings(self, year):
        if year not in self.years[:-1]:
            print('year {0} out of range'.format(year))
            return None
        break_day = pd.Timestamp('{0}-04-21'.format(year)).dayofyear
        if year == 2002:
            block_strings = ['year=={0}'.format(year), 
                             'year=={0}&date<={1}'.format(year+1, break_day)]
        elif year == 2014:
            block_strings = ['year=={0}&date>{1}'.format(year, break_day), 
                             'year=={0}'.format(year+1)]
        else:
            break_day_next = pd.Timestamp('{0}-04-21'.format(year+1)).dayofyear
            block_strings = ['year=={0}&date>{1}'.format(year, break_day), 
                             'year=={0}&date<={1}'.format(year+1, break_day_next)]
        return block_strings

    def get_overlap_dur(self, block_string, dur):
        label_str = 'labels_{0}'.format(dur)
        labels_pr = pd.read_hdf(store_name, 
                                key=block_string+'/labels_{0}'.format(dur)).values
        dfr_pr = pd.read_hdf(store_name, key=block_string, columns=['lon', 'lat', 'day_since'])
        dfr_pr.loc[:,label_str] = labels_pr 
        last_day = dfr_pr['day_since'].max()
        ovarlap_labels = dfr_pr[dfr_pr['day_since'] >= last_day - dur][label_str]
        overlap_dfr = dfr_pr[dfr_pr[label_str].isin(overlap_labels[overlap_labels > -1])]
        overlap_labels = overlap_dfr[label_str]
        #TODO finish this
        #+ -1's from overlap duration and return!

 

    def cluster_blocks(self, store_name, dur):
        start_time = time.time()
        for dur in np.arange(2, 15, 2):
            print(dur)
            for nr, year in enumerate(self.years[:-1]):
                print(year)
                block_string = 'Af_tr/block_{0}'.format(year)
                dfr = pd.read_hdf(store_name, key=block_string, columns=['x', 'y', 'z', 'day_since'])
                dfr.loc[:, 'day_since_tmp'] = dfr['day_since'] * (self.eps / dur)
                if nr == 0:
                    labels = cluster_euc(dfr[['x', 'y', 'z', 
                                                    'day_since_tmp']].values,
                                                                    self.eps, 
                                                                    min_samples=2)
                    label_fr = pd.DataFrame({'labels_{0}'.format(dur): labels})
                    label_fr.to_hdf(store_name, key=block_string+'/labels_{0}'.format(dur),
                                    format='table', data_columns=['labels_{0}'.format(dur)],
                                    append=True)
                else:
                    labels = cluster_euc(dfr[['x', 'y', 'z', 'day_since']].values, self.eps, min_samples=2)

                    overlap = get_overlap_dur()
                    label_fr = pd.DataFrame({'labels_{0}'.format(labels): labels})
                    labels_pr = pd.read_hdf(store_name, 
                                   key='Af_tr/block_{0}/labels_{1}'.format(years[nr-1], dur))
                fr['labs_{0}'.format(str(dur))] = labels
                fr.to_hdf(store_name, key='{0}/labs_{1}'.format(obj, str(dur)), format='table',
                        columns=['labs_{0}'.format(str(dur))], append=True)
                print("--- %s seconds ---" % (time.time() - start_time))
                start_time = time.time()


    def cluster_store(self, store_name, obj):
        start_time = time.time()
        print(obj)
        dfr = pd.read_hdf(store_name, key=obj, columns=['x', 'y', 'z', 'day_since'])
        for dur in [1, 2, 4, 8, 16]:
            print(dur)
            dfr.loc[:, 'day_since_tmp'] = dfr['day_since'] * (self.eps / dur)
            labels = cluster_euc(dfr[['x', 'y', 'z', 'day_since_tmp']].values, self.eps, min_samples=1)
            label_fr = pd.DataFrame({'labels_{0}'.format(dur): labels})
            label_fr.to_hdf(store_name, key=obj+'/labels_{0}'.format(dur),
                            format='table', data_columns=['labels_{0}'.format(dur)],
                            append=True)

if __name__ == '__main__':
    data_path = '.'
    #data_path = '/mnt/data/area_burned_glob'
    #store_name = os.path.join(data_path, 'ba_store.h5')
    store_name = 'ba_tropics_store.h5'
    #tropics_store = 'ba_tropics_store.h5'
    ba = FireObs(data_path, os.path.join(data_path, store_name))
    #dur = 16
    #dfr.loc[:, 'day_since_tmp'] = dfr['day_since'] * (self.eps / dur)
    ##labs16 = cluster_euc(dfr[['x', 'y', 'z', 'day_since_tmp']].values, self.eps, min_samples=2)
    #dfr.loc[:, 'labs16'] = labs16
    #ba.populate_store_af_blocks(store_name, tropics_store, )
    #ba.cluster_store(store_name, ['Af_tr', 'Am_tr', 'As_tr'])
    #ba.populate_store_tropics(tropics_store)
    #ba.populate_store()
