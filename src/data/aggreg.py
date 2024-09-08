#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Ivo Marvan"
__email__ = "ivo@wmarvan.cz"
__description__ = '''
    Aggregates data from multiple sources.
    
    Vstup:
    - soubor Timeseries_48.172_17.181_SA2_2000kWp_crystSi_14_38deg_-2deg_2005_2020.csv.gz.
    - se sloupci: time,P,Gb(i),Gd(i),Gr(i),H_sun,T2m,WS10m,Int
    - ve tvaru:
        20050101:0710,0.0,0.0,0.0,0.0,0.0,3.9,3.17,0.0
        20050101:0810,55800.0,3.62,41.68,0.99,9.91,4.42,3.38,0.0
        20050101:0910,66520.0,0.0,51.87,1.22,15.1,4.96,3.59,0.0
        
    Výstup:
    Pro všechny sloupce mimo time udělá průměr pro každou minutu ze všech roků (2005-2020).
    Výstup bude uložen do souboru nogit_data/aggregated.csv.gz. 
        
'''
import os
import pandas as pd

# root of project repository
from git_root_to_syspath import agr; PROJECT_ROOT = agr()
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
NOGIT_DIR = os.path.join(PROJECT_ROOT, 'nogit_data')


# Načtení souboru s komprimovaným obsahem
file_path = os.path.join(DATA_DIR, 'Timeseries_48.172_17.181_SA2_2000kWp_crystSi_14_38deg_-2deg_2005_2020.csv.gz')

# Načtení dat do DataFrame
df = pd.read_csv(file_path, compression='gzip')
df['time'] = pd.to_datetime(df['time'], format='%Y%m%d:%H%M', errors='coerce')

# Přidání sloupce pro měsíc
df['month'] = df['time'].dt.month
df['hour_minute'] = df['time'].dt.strftime('%H:%M')

# Agregace podle měsíce a času (hodina:minuta)
aggregated_df = df.groupby(['month', 'hour_minute']).mean()

# Uložení do souboru
output_path = os.path.join(NOGIT_DIR, 'aggregated.csv.gz')
aggregated_df.to_csv(output_path, compression='gzip')


