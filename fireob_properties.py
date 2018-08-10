import os
import pandas as pd

def spatial_subset_dfr(dfr, bbox):
    """
    Selects data within spatial bbox. bbox coords must be given as
    positive values for the Northern hemisphere, and negative for
    Southern. West and East both positive - Note - the method is
    naive and will only work for bboxes fully fitting in the Eastern hemisphere!!!
    Args:
        dfr - pandas dataframe
        bbox - (list) [North, South, West, East]
    Returns:
        pandas dataframe
    """
    dfr = dfr[(dfr['lat'] < bbox[0]) & (dfr['lat'] > bbox[2])]
    dfr = dfr[(dfr['lon'] > bbox[1]) & (dfr['lon'] < bbox[3])]
    return dfr


class FireObjs(object):
    def __init__(self, data_path):
        self.data_path = data_path
        self.basedate = pd.Timestamp('2002-01-01')
        self.labels = ['labs1', 'labs2', 'labs4', 'labs8', 'labs16']
        self.regions_bounds = {'indonesia': [8.0, 93.0, -13.0, 143.0],
                               'riau': [3, -2, 99, 104]}

    def read_fires_labeled(self, bbox):
        dfr = self.read_dfr_from_parquet('As_tr_mod_lab')
        dfr = spatial_subset_dfr(dfr, bbox)
        return dfr

    def read_dfr_from_parquet(self, region_id):
        columns=['lon', 'lat', 'date', 'day_since'] + self.labels
        store_name = os.path.join(self.data_path, '{0}.parquet'.format(region_id))
        dfr = pd.read_parquet(store_name, columns=columns)
        return dfr

if __name__ == '__main__':
    data_path = 'data'
    fo = FireObjs(data_path)
