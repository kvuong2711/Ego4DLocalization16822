import json

import geojson
import math
from io import BytesIO
import numpy as np
import requests
from PIL import Image
import geog
from datetime import datetime
from bokeh.io import output_file, show, output_notebook
from bokeh.models import ColumnDataSource, GMapOptions
from bokeh.plotting import gmap
from bokeh.models import Range1d
from geopy import distance

# -----------------------------
# API KEYS AND URLs
# -----------------------------

# Official Google Street View API
API_KEY = 'AIzaSyCFXBzFQ3j0-n1shW0YMV6ellhd0zfXF3A'

# URL to get metadata from panoid. One is working, one is not (as of Sep 2022)
# URL_STR = 'https://maps.google.com/cbk?output=json&cb_client=maps_sv&v=4&dm=1&pm=1&ph=1&hl=en&panoid={}'
URL_STR = 'https://www.google.com/maps/photometa/v1?authuser=0&hl=en&gl=uk&pb=!1m4!1smaps_sv.tactile!11m2!2m1!1b1!2m2!1sen!2suk!3m3!1m2!1e2!2s{}!4m57!1e1!1e2!1e3!1e4!1e5!1e6!1e8!1e12!2m1!1e1!4m1!1i48!5m1!1e1!5m1!1e2!6m1!1e1!6m1!1e2!9m36!1m3!1e2!2b1!3e2!1m3!1e2!2b0!3e3!1m3!1e3!2b1!3e2!1m3!1e3!2b0!3e3!1m3!1e8!2b0!3e3!1m3!1e1!2b0!3e3!1m3!1e4!2b0!3e3!1m3!1e10!2b1!3e2!1m3!1e10!2b0!3e3'

# URL to download panorama image based on panoid
PANO_URL = 'https://maps.google.com/cbk?output=tile&panoid={panoid}&zoom={z}&x={x}&y={y}&' + str(datetime.now().microsecond)

# Parameters
GSV_TILEDIM = 512

# -----------------------------
# API KEYS AND URLs
# -----------------------------


def circular_grid(cntr_latlng, dim=50, min_cnt=50):
    lnglats = [(cntr_latlng[1], cntr_latlng[0])]
    d = dim
    n = 0
    while len(lnglats) < min_cnt:
        c = d * 2 * math.pi
        n_points = int(math.floor(c / dim))
        angles = np.linspace(0, 360, n_points)
        if n % 2 == 0:
            angles += 360.0 / n_points / 2.0
        lnglats.extend(geog.propagate(lnglats[0], angles, d))
        # print("{} \t {}".format(d,n_points))
        d += dim
        n += 1

    return lnglats


def panoid_to_depthinfo(panoid):
    # URL of the json file of a GSV depth map
    url_depthmap = URL_STR.format(panoid)

    r = requests.get(url_depthmap)
    resp = r.text[5:]
    json_data = json.loads(resp)

    size_img = (int(json_data[1][0][2][2][1]), int(json_data[1][0][2][2][0]))
    size_til = (0, 0)

    # # OLD ONE: no idea why this isn't working anymore... (as of Sep 2022)
    # r = requests.get(url_depthmap)  # getting the json file
    # json_data = r.json()
    # try:
    #     size_img = (int(json_data['Data']['image_width']), int(json_data['Data']['image_height']))
    #     size_til = (int(json_data['Data']['tile_width']), int(json_data['Data']['tile_height']))
    # except:
    #     print("The returned json could not be decoded")
    #     print(url_depthmap)
    #     print("status code: {}".format(r.status_code))
    #     return False, False, False

    return json_data, size_img, size_til


def panoid_to_img(panoid, api_key, zoom, size_img, flip=False):
    w, h = 2 ** zoom, 2 ** (zoom - 1)
    dim = False
    if size_img[0] == 13312: dim = 416
    if size_img[0] == 16384: dim = 512
    if not dim:
        print("!!!! THIS PANO IS A STRANGE DIMENSION {}".format(panoid))
        print("zoom:{}\t w,h: {}x{} \t image_size:{}x{}".format(zoom, w, h, size_img[0], size_img[1]))
        return False

    img = Image.new("RGB", (w * dim, h * dim), "red")
    try:
        for y in range(h):
            # if y % 5 == 0:
            #     print('{}/{}'.format(y, h))
            for x in range(w):
                # print(y, x)
                url_pano = PANO_URL.format(panoid=panoid, z=zoom, x=x, y=y)
                response = requests.get(url_pano)
                img_tile = Image.open(BytesIO(response.content))
                img.paste(img_tile, (GSV_TILEDIM * x, GSV_TILEDIM * y))
    except:
        print("!!!! FAILED TO DOWNLOAD PANO for {}".format(panoid))
        return False

    if flip:
        print('FLIPPING LEFT-RIGHT TO MATCH DEPTH!')
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    else:
        return img


# def visualize_gmap(map_lat, map_lng, lats, lons, map_name, colors):
#     map_options = GMapOptions(lat=map_lat, lng=map_lng, map_type="hybrid", zoom=20, tilt=0)
#
#     # For GMaps to function, Google requires you obtain and enable an API key:
#     #
#     #     https://developers.google.com/maps/documentation/javascript/get-api-key
#     #
#     # Replace the value below with your personal API key:
#     p = gmap(API_KEY,
#              map_options=map_options, title=map_name, width=960, height=540)
#
#     source = ColumnDataSource(
#         data=dict(lat=lats,
#                   lon=lons)
#     )
#
#     p.circle(x="lon", y="lat", size=15, fill_color="red", fill_alpha=0.8, source=source)
#
#     show(p)
#
#     # hacky shit
#     lats_blue = []
#     lons_blue = []
#     for i, color in enumerate(colors):
#         if color == 'blue':
#             lats_blue.append(lats[i])
#             lons_blue.append(lons[i])
#
#     source = ColumnDataSource(
#         data=dict(lat=lats_blue,
#                   lon=lons_blue)
#     )
#
#     p.circle(x="lon", y="lat", size=15, fill_color='blue', fill_alpha=0.8, source=source)
#
#     show(p)
