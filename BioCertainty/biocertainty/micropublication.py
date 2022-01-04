#-*- coding: utf-8 -*-
#!/usr/bin/env python

from .certainty import Certainty
from .seeker import doi_pmc_seeker
import requests
import re
import json
import md5
import uuid
import datetime


RE = '(10[.][0-9]{4,}(?:[.][0-9]+)*/(?:(?!["&\'<>])\S)+)'


def Micropublication(pmc_doi, CLAIM):
    categorization = Certainty(CLAIM)
    doi = doi_pmc_seeker(pmc_doi)
    index = str(uuid.uuid1()) 
    try:
        if bool(re.match(RE, doi)) == True:
            index = str(uuid.uuid1()) 
            url = requests.get('https://doi.org/'+doi, headers={'Accept': 'application/json'}) #cambiar pmc por pmc
            json_data = url.json()
            date = (json_data['created']['date-time']).encode('utf-8')
            if type(json_data['title']) == list and len(json_data['title']) >= 1:
                title = json_data['title'][0]
            elif type(json_data['title']) == list and len(json_data['title']) == 0:
                title = doi
            else:
                title = json_data['title'].encode('utf-8')
            micropub = ("""<http://linkeddata.systems/micropubs_mario/graph_CertID_{0}> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://purl.org/mp/Statement> .

<http://linkeddata.systems/micropubs_mario/micropub_{0}> <http://purl.org/mp/hasAttributionAsPublisher> <https://github.com/Guindillator/Certainty> <http://linkeddata.systems/micropubs_mario/micropubgraph_{0}>.
<http://linkeddata.systems/micropubs_mario/micropub_{0}> <http://purl.org/mp/argues> <http://linkeddata.systems/micropubs_mario/claim_{0}> <http://linkeddata.systems/micropubs_mario/micropubgraph_{0}>.
<http://linkeddata.systems/micropubs_mario/micropub_{0}> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://purl.org/mp/Micropublication> <http://linkeddata.systems/micropubs_mario/micropubgraph_{0}>.
<http://linkeddata.systems/micropubs_mario/micropub_{0}> <http://purl.org/mp/hasSupportGraphGraphElement> <http://linkeddata.systems/micropubs_mario/graph_{0}> <http://linkeddata.systems/micropubs_mario/micropubgraph_{0}>.

<https://github.com/Guindillator/Certainty> <http://www.w3.org/2000/01/rdf-schema#label> "The Wilkinson laboratory micropub certainty annotator"@en  <http://linkeddata.systems/micropubs_mario/micropubgraph_{0}>.
<https://github.com/Guindillator/Certainty> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://www.w3.org/ns/prov#Agent>  <http://linkeddata.systems/micropubs_mario/micropubgraph_{0}>.


<http://linkeddata.systems/micropubs_mario/claim_{0}> <http://purl.org/mp/hasAttribution> <https://doi.org/{1}> <http://linkeddata.systems/micropubs_mario/graph_{0}>.
<http://linkeddata.systems/micropubs_mario/claim_{0}> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://purl.org/mp/Claim> <http://linkeddata.systems/micropubs_mario/graph_CertID_{0}>.
<http://linkeddata.systems/micropubs_mario/claim_{0}> <http://www.w3.org/2000/01/rdf-schema#label> "{2}" <http://linkeddata.systems/micropubs_mario/graph_CertID_{0}>.
<http://linkeddata.systems/micropubs_mario/claim_{0}> <http://w3id.org/orca-x#hasConfidenceLevel> <http://linkeddata.systems/micropubs_mario/conf_CertID_{0}> <http://linkeddata.systems/micropubs_mario/graph_CertID_{0}>.


<http://linkeddata.systems/micropubs_mario/conf_CertID_{0}> <http://www.w3.org/2000/01/rdf-schema#label> "Confidence {3}"@en <http://linkeddata.systems/micropubs_mario/graph_CertID_{0}> .
<http://linkeddata.systems/micropubs_mario/conf_CertID_{0}> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://w3id.org/orca-x#{3}> <http://linkeddata.systems/micropubs_mario/graph_CertID_{0}> .
<https://doi.org/{1}> <http://schema.org/identifier> '{1}'@en <http://linkeddata.systems/micropubs_mario/graph_CertID_{0}> .
<https://doi.org/{1}> <http://purl.org/dc/terms/date> '{5}'^^<http://www.w3.org/2001/XMLSchema#dateTime> <http://linkeddata.systems/micropubs_mario/graph_CertID{0}> .
<https://doi.org/{1}> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://purl.org/mp/ArticleText> <http://linkeddata.systems/micropubs_mario/graph_CertID_{0}> .
<https://doi.org/{1}> <http://purl.org/dc/terms/identifier> '{1}'@en <http://linkeddata.systems/micropubs_mario/graph_CertID_{0}> .
<https://doi.org/{1}> <http://purl.org/dc/terms/title> '{4}'@en <http://linkeddata.systems/micropubs_mario/graph_CertID_{0}> .""").format(index, doi, CLAIM, categorization, title, date)
            return micropub, index

        else:
            raise ValueError("This article does not have DOI. It is neccesary a DOI.")


    except AssertionError as error:
        print(error)