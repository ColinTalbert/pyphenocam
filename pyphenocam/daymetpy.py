#!/usr/bin/env python
import os, sys ,csv, time, urllib.request, urllib.parse, urllib.error
        
def download_Daymet(site="Daymet",lat=36.0133,lon=-84.2625,start_yr=1980,end_yr=2012, as_dataframe=True):
    """
    Function to read download daymet data for a single pixel location
    """
    
    # calculate the end of the range of years to download
    max_year = int(time.strftime("%Y"))-2

    # check validaty of the range of years to download
    # I'm not sure when new data is released so this might be a
    # very conservative setting, remove it if you see more recent data
    # on the website
    if start_yr >= 1980 & end_yr <= max_year:        
    
        # if the year range is valid, create a string of valid years
        year_range = list(range(start_yr, end_yr))
    
        # convert to string and strip the brackets and spaces
        year_range = str(year_range).strip('[]')
        year_range = year_range.replace(' ','')

        # create download string / url
        download_string = "https://daymet.ornl.gov/data/send/saveData?lat=my_lat&lon=my_lon&measuredParams=tmax,tmin,dayl,prcp,srad,swe,vp&year=year_range"
  
        # substitute input variables in download string
        download_string = download_string.replace("my_lat",str(lat))
        download_string = download_string.replace("my_lon",str(lon))
        download_string = download_string.replace("year_range",year_range)
  
        # create filename for the output file
        daymet_file = str(site)+"_"+str(start_yr)+"_"+str(end_yr)+'.csv' 
  
        # download the daymet data (if available)
        urllib.request.urlretrieve(download_string,daymet_file)

        if os.path.getsize(daymet_file) == 0:
            os.remove(daymet_file)
            raise NameError("You requested data is outside DAYMET coverage, the file is empty --> check coordinates!")
    
        if as_dataframe:
            import pandas as pd
            df = pd.read_csv(daymet_file, header=6)
            df.index = pd.to_datetime(df.year.astype(str) + '-' + df.yday.astype(str), format="%Y-%j")
            df.columns = [c[:c.index('(')].strip() if '(' in c else c for c in df.columns ]
            return df
        else:
            return daymet_file

    else:
        raise NameError("Year values are out of range!")

if __name__ == "__main__":
    
    # set maximum year -- or update according to the website's
    # info
    max_year = int(time.strftime("%Y"))-2

    # create empty matrix
    sites = []   
    
    # the file name with sites should be the first argument to the
    # program executed without the python command
    my_sites_file = sys.argv[1]

    # do a basic check to see if the file exists, does not check if it
    # is comma separated or contains the correct data, you are on your own
    # there
    if os.path.isfile(my_sites_file):    
        with open(my_sites_file, 'rb') as f:
            reader = csv.reader(f)
            for row in reader:
                sites.append(row)
    
    for site in sites:
        try:
            download_Daymet(site=str(site[0]), lat=site[1], lon=site[2],
                            start_yr=1980, end_yr=max_year)
        except NameError:
            print("Error: check error messages!")
            
            # uncomment if you want extensive debugging info
            # raise 
