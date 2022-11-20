import json
import os
import re
import time
from geopy import distance
import cv2
import numpy as np
import Equirec2Perspec as E2P
from tqdm import tqdm
from utils import *


def visualize_gmap(pano_json, map_title='Default Title'):
    # read file
    with open(pano_json, 'r') as file:
        data = file.read()

    # parse file
    flat_panoids = json.loads(data)

    print('All panos:', len(flat_panoids))

    lats = []
    lons = []
    for pano_info in flat_panoids:
        lats.append(pano_info['lat'])
        lons.append(pano_info['lng'])

    map_lat = np.mean(np.array(lats))
    map_lng = np.mean(np.array(lons))

    map_options = GMapOptions(lat=map_lat, lng=map_lng, map_type="hybrid", zoom=18, tilt=0)

    # For GMaps to function, Google requires you obtain and enable an API key:
    #
    #     https://developers.google.com/maps/documentation/javascript/get-api-key
    #
    # Replace the value below with your personal API key:
    p = gmap(API_KEY, map_options=map_options, title=map_title, width=960, height=540)

    source = ColumnDataSource(
        data=dict(lat=lats,
                  lon=lons)
    )

    p.circle(x="lon", y="lat", size=15, fill_color="red", fill_alpha=0.8, source=source)

    output_file('./map_visualization.html')
    show(p)


def get_metadata(panoid, lat=None, lng=None):
    assert panoid is not None or (lat is not None and lng is not None)
    if panoid is not None:
        url = 'https://maps.googleapis.com/maps/api/streetview/metadata?pano={0}&key={1}'.format(panoid, API_KEY)
    else:
        url = 'https://maps.googleapis.com/maps/api/streetview/metadata?location={0}%2C{1}&key={2}'.format(lat, lng,
                                                                                                           API_KEY)
    return requests.get(url, proxies=None)


def get_closest_panos(lat, lng):
    """
    Get the closest panos around a lat/lng
    Note: unofficial API!
    """

    url = 'https://maps.googleapis.com/maps/api/js/GeoPhotoService.SingleImageSearch?pb=!1m5!1sapiv3!5sUS!11m2!1m1!1b0!2m4!1m2!3d{0:}!4d{1:}!2d50!3m18!2m2!1sen!2sUS!9m1!1e2!11m12!1m3!1e2!2b1!3e2!1m3!1e3!2b1!3e2!1m3!1e10!2b1!3e2!4m6!1e1!1e2!1e3!1e4!1e8!1e6&callback=_xdc_._vllbfd'

    resp = requests.get(url.format(lat, lng), proxies=None)

    pans = re.findall('\[[0-9]+,"(.+?)"\].+?\[\[null,null,(-?[0-9]+.[0-9]+),(-?[0-9]+.[0-9]+)', resp.text)
    # print(pans)
    pans = [{
        "panoid": p[0],
        "lat": float(p[1]),
        "lng": float(p[2])} for p in pans]  # Convert to floats

    # Remove duplicate panoramas
    pans = [p for i, p in enumerate(pans) if p not in pans[:i]]

    return pans


def query_all_panos(lat, lng, all_panos_json_path, pruned_panos_json_path, read_only=True):
    # -----------------------------------------------------------------------------
    # Parameters
    MIN_YEAR = 2019
    MAX_YEAR = 2022
    MAX_DIST = 50.0
    # -----------------------------------------------------------------------------
    orig_lat_lng = (lat, lng)

    if not read_only:
        # We will have two ways of sampling panoramas (we will merge them together):
        # 1) Circular grid
        # 2) Two recursive layers of get_closest_panos()
        panodict_list = []

        # Method 1) Obtain the initial panodict by sampling circular grid and get closest panos with rough id/lat/lng
        gjpts = circular_grid((lat, lng), dim=30, min_cnt=30)

        for pano_lnglat in gjpts:
            panodict_i = get_closest_panos(pano_lnglat[1], pano_lnglat[0])
            panodict_list.append(panodict_i)

        # Method 2) Two recursive layers of get_closest_panos()
        panodict_recursive = get_closest_panos(lat=lat, lng=lng)
        for pano_info in panodict_recursive:
            panodict_i = get_closest_panos(pano_info['lat'], pano_info['lng'])
            panodict_list.append(panodict_i)

        # Flatten it out
        flat_panoids = [item for sublist in panodict_list for item in sublist]

        # Write to file (all panos, no pruning performed yet)
        with open(all_panos_json_path, 'w') as file:
            file.write(json.dumps(flat_panoids, indent=4))

    # -----------------------------
    # Processing starts here!
    # -----------------------------
    # Read the file back in
    with open(all_panos_json_path, 'r') as file:
        data = file.read()

    # parse file
    flat_panoids = json.loads(data)
    print('All panos:', len(flat_panoids))

    # Remove duplicates
    flat_panoids = list({v['panoid']: v for v in flat_panoids}.values())

    print('After removing duplicates:', len(flat_panoids))

    # Prune based on year
    # Additionally: the metadata will be queried from official Google API (maybe better lat/lon?)
    final_panoids_timepruned = []
    for i, pano_info in enumerate(tqdm(flat_panoids)):
        meta = get_metadata(pano_info['panoid']).json()
        if not meta['status'] == "OK":
            print("NO PANORAMA FOUND FOR GIVEN LATLNG. status: {}".format(meta['status']))
            continue

        if not (meta['copyright'].split()[-1].lower() == "Google".lower()):
            print("Found a non-google copyright ({}). skipping {}.".format(meta['copyright'], meta['pano_id']))
            continue

        if 'date' not in meta:
            continue

        # Get all the metadata from the official API to ensure correctness when doing any alignment later
        year, month = int(meta['date'][:4]), int(meta['date'][-2:])
        if year < MIN_YEAR or year > MAX_YEAR:
            continue

        pano_info['year'] = year
        pano_info['month'] = month
        pano_info['lat'] = float(meta['location']['lat'])
        pano_info['lng'] = float(meta['location']['lng'])

        final_panoids_timepruned.append(pano_info)

    print('After pruning by year:', len(final_panoids_timepruned))

    # # TODO: extra hack
    # additional_panoids = ['wGTnx4TjvkrjL2qvocA45Q', 'o3laPnddonmCwEsHtA4Dgg', 'tUaYMJnVB1gNmWya5pa0mQ']
    #
    # for i, pano_id in enumerate(tqdm(additional_panoids)):
    #     meta = get_metadata(pano_id).json()
    #     if not meta['status'] == "OK":
    #         print("NO PANORAMA FOUND FOR GIVEN LATLNG. status: {}".format(meta['status']))
    #         continue
    #
    #     if not (meta['copyright'].split()[-1].lower() == "Google".lower()):
    #         print("Found a non-google copyright ({}). skipping {}.".format(meta['copyright'], meta['pano_id']))
    #         continue
    #
    #     if 'date' not in meta:
    #         continue
    #
    #     # Get all the metadata from the official API to ensure correctness when doing any alignment later
    #     year, month = int(meta['date'][:4]), int(meta['date'][-2:])
    #     if year < MIN_YEAR or year > MAX_YEAR:
    #         continue
    #
    #     pano_info = {'pano_id': pano_id}
    #     pano_info['year'] = year
    #     pano_info['month'] = month
    #     pano_info['lat'] = float(meta['location']['lat'])
    #     pano_info['lng'] = float(meta['location']['lng'])
    #
    #     final_panoids_timepruned.append(pano_info)

    final_panoids = []
    for pano_info in final_panoids_timepruned:
        pano_lat_lng = (pano_info['lat'], pano_info['lng'])
        dist_to_origin = distance.distance(pano_lat_lng, orig_lat_lng).m
        if dist_to_origin > MAX_DIST:
            continue
        pano_info['dist2origin'] = dist_to_origin
        final_panoids.append(pano_info)

    print('After pruning by distance:', len(final_panoids))

    print('Sorting panolist by distance')
    final_panoids = sorted(final_panoids, key=lambda i: i['dist2origin'])

    if len(final_panoids) > 10:
        print('>>>>>>>>>>>>>> Subsample panolist by 2 with some heuristics...')
        # half_point = len(final_panoids) // 2
        # final_panoids = final_panoids[:half_point:2] + final_panoids[half_point:]
        final_panoids = final_panoids[::4]

    with open(pruned_panos_json_path, 'w') as file:
        file.write(json.dumps(final_panoids, indent=4))


def download_panos(pano_json, out_dir):
    # read file
    with open(pano_json, 'r') as file:
        data = file.read()

    # parse file
    flat_panoids = json.loads(data)

    print('All panos:', len(flat_panoids))

    print('Sorting panolist by distance')
    flat_panoids = sorted(flat_panoids, key=lambda i: i['dist2origin'])

    print('Start downloading')
    idx = 0
    for _, panodict in enumerate(tqdm(flat_panoids)):
        panoid = panodict['panoid']
        pano_out_path = os.path.join(out_dir, "{:06d}_{}.{}".format(idx, panoid, 'png'))
        if os.path.exists(pano_out_path):
            print('Skipping', panoid)
            idx += 1
            continue
        # print('Downloading {}, dist2origin {} (m)'.format(panoid, panodict['dist2origin']))
        # try:
        depth_info, size_img, size_tile = panoid_to_depthinfo(panoid=panoid)
        # except:
        #     continue
        pano_img = panoid_to_img(panoid, API_KEY, zoom=5, size_img=size_img, flip=False)
        # print('Saving to file')
        if pano_img:
            pano_img.save(os.path.join(out_dir, "{:06d}_{}.{}".format(idx, panoid, 'png')))  # save pano
        else:
            print("!!!! FAILED\t{}".format(panoid))
        idx += 1

        # Prevent timeout from the downloading server
        # Since it's unofficial, it's unclear how much we can download at once, so better be safe than sorry...
        time.sleep(2)


def convert_equirect_perspective(pano_dir, pano_name, output_dir):
    # Convert from equirectangular to perspective
    panoimg_path = os.path.join(pano_dir, pano_name + '.png')
    assert os.path.exists(panoimg_path)

    pano_output_dir = os.path.join(output_dir, pano_name)
    os.makedirs(pano_output_dir, exist_ok=True)

    equ = E2P.Equirectangular(panoimg_path)

    # theta_range = np.arange(-180, 180, 20)  # yaw
    # phi_range = np.array([-15, 0, 10])  # pitch
    theta_range = np.arange(-180, 180, 30)  # yaw
    phi_range = np.array([0])  # pitch
    idx = 0
    for phi in phi_range:
        for theta in theta_range:
            # if idx % 30 == 0:
            #     print('{}/{}'.format(idx, len(theta_range) * len(phi_range)))
            img = equ.GetPerspective(90, theta, phi, 1080, 1920)  # Specify parameters(FOV, theta, phi, height, width)
            perspective_path = os.path.join(pano_output_dir, "{}_{:06d}.png".format(pano_name, idx))
            cv2.imwrite(perspective_path, img)
            idx += 1
