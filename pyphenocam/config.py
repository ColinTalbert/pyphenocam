import os
import appdirs

import ConfigParser

def initialize_config(fname):
    config = ConfigParser.ConfigParser()
    config.add_section('URLS')
    config.set('URLS', 'SITEURL', "http://phenocam.sr.unh.edu/webcam/roi/roilistinfo/?format=csv")
    config.set('URLS', 'ROIURL', "http://phenocam.sr.unh.edu/data/archive/{}/ROI")
    config.set('URLS', 'FNAMEURL', "http://phenocam.sr.unh.edu/data/archive/{}/{}/{}/{}")
    config.set('URLS', 'FNAMEURL_BROWSE', "http://phenocam.sr.unh.edu/webcam/browse/{}/{}/{}/{}")
    config.set('URLS', 'IRFNAMEURL', "http://phenocam.sr.unh.edu/webcam/browse/{}_IR/{}/{}/{}")
    config.add_section('CACHEDNAME')
    config.set('CACHEDNAME', 'folder', appdirs.user_data_dir("pyphenocam", "USGS"))

    # Writing our configuration file to 'example.cfg'
    with open(fname, 'wb') as configfile:
        config.write(configfile)

def get_config_fname():
    '''Find the location of the package config file
    '''
    config_fname = os.path.join(appdirs.user_data_dir("pyphenocam", "USGS"), '.pyphenocamconfig')
    
    if not os.path.exists(config_fname):
        initialize_config(config_fname)
    
    return config_fname

def get_config():
    config = ConfigParser.ConfigParser()
    config.read(get_config_fname())
    return config
    
def get_cache_dname():
    config = get_config()
    return config.get('CACHEDNAME', 'folder')
    
def set_cache_dname(cache_dname):
    if not os.path.exists(cache_dname):
        raise Exception, "{} does not exist!".format(cache_dname)
    
    config = get_config()
    config.set('CACHEDNAME', 'folder', cache_dname)
    
    with open(get_config_fname(), 'wb') as configfile:
        config.write(configfile)

def get_url(which='SITEURL'):
    config = get_config()
    return config.get('URLS', which)

def set_url(url, which='SITEURL'):
    config = get_config()
    config.set('URLS', which, url)
    
    with open(get_config_fname(), 'wb') as configfile:
        config.write(configfile)