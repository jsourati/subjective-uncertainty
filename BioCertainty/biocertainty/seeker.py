#-*- coding: utf-8 -*-
#!/usr/bin/env python


import re
import requests
import xml.etree.ElementTree as ET

RE = '(10[.][0-9]{4,}(?:[.][0-9]+)*/(?:(?!["&\'<>])\S)+)'

def doi_pmc_seeker(pmc, doi = False):
    if doi == True:
        if bool(re.match(RE, doi)) == True:
            return doi
        else:
            raise ValueError("This is not a valid PMC")
    else:
        try: 
            url = requests.get('https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?ids='+pmc)
            x = ET.fromstring(url.text)
            for atype in x.findall('record'):
                doi = (atype.get('doi'))
            if doi == None:
                return pmc 
            else:
                return doi
        except AssertionError as error:
            print(error)
