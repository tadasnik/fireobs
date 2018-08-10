import numpy as np
import holoviews as hv
import holoviews.plotting.mpl
print(hv.__version__)

r = hv.renderer('bokeh')

curve = hv.Curve(range(10))
img = hv.Image(np.random.rand(10,10))
_=r.save([curve, img, curve+img], plot.html)

