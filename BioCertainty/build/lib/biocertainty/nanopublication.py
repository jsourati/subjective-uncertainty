#-*- coding: utf-8 -*-
#!/usr/bin/env python

from .certainty import Certainty
from .seeker import doi_pmc_seeker
import re
import md5
import uuid
import datetime


RE = '(10[.][0-9]{4,}(?:[.][0-9]+)*/(?:(?!["&\'<>])\S)+)'


def Nanopublication(pmc_doi, CLAIM, this):
    categorization = Certainty(CLAIM)
    doi = doi_pmc_seeker(pmc_doi)
    index = str(uuid.uuid1()) 
    try:
        if bool(re.match(RE, doi)) == True:
            text = ("https://dx.doi.org/%s") % (doi)
            nanopub = ("""@prefix this: <{8}CertID_{0}> .
@prefix sub: <{8}CertID_{0}#> .
@prefix void: <http://rdfs.org/ns/void#> .
@prefix dcterms: <http://purl.org/dc/terms/> .
@prefix dcelem: <http://purl.org/dc/elements/1.1/> .
@prefix np: <http://www.nanopub.org/nschema#> .
@prefix pav: <http://swan.mindinformatics.org/ontologies/1.2/pav/> .
@prefix prov: <http://www.w3.org/ns/prov#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix dcat: <http://www.w3.org/ns/dcat#> .
@prefix schema: <https://schema.org/> .
@prefix thispub: <https://dx.doi.org/{1}#> .
@prefix orca-x: <http://w3id.org/orca-x#> .

sub:Head {6}
        this: np:hasAssertion sub:assertion ;
        np:hasProvenance sub:provenance ;
        np:hasPublicationInfo sub:pubinfo ;
        a np:Nanopublication .
{7}

sub:assertion {6}
		orca-x:asserts-{0} rdf:singletonPropertyOf orca-x:asserts .
		thispub: orca-x:asserts-{0} "{9}" .
		orca-x:asserts-{0} orca-x:hasConfidenceLevel orca-x:{3} .
{7}

sub:provenance {6}   
        sub:assertion dcterms:author "Certainty Classifier" ;
        dcterms:title "Automated Certainty Classification of Statement from https:dx.doi.org/{1}" ;
        dcterms:license <https://creativecommons.org/publicdomain/zero/1.0/> ;
        schema:identifier this: ;
        dcat:distribution sub:assertion ;
        prov:wasDerivedFrom sub:_1 .

        sub:_1   dcelem:format "application/pdf" ;
        a void:Dataset , dcat:Distribution ;
        dcat:downloadURL <{4}> .

{7}

sub:pubinfo {6}
        this: dcterms:created '{5}'^^xsd:date ;
        dcterms:rights <https://creativecommons.org/publicdomain/zero/1.0> ;
        dcterms:rightsHolder <https://orcid.org/0000-0002-9416-6743> ;
        pav:authoredBy "Mario Prieto" , <https://orcid.org/0000-0002-9416-6743> ;
        pav:versionNumber "1" ;
        prov:wasGeneratedBy "Mario Prieto's Certainty Classifier" .
{7}
    """).format(index, doi, md5.new(CLAIM).hexdigest(), categorization, text, datetime.date.today(), '{', '}', this, CLAIM)
            return nanopub, index
        else:
            pmc = pmc_doi
            text = ("https://www.ebi.ac.uk/europepmc/webservices/rest/%s/fullTextXML") % (pmc)
            url = ("https://www.ncbi.nlm.nih.gov/pmc/articles/%s") % (pmc)
            nanopub = ("""@prefix this: <{8}CertID_{0}> .
@prefix sub: <{8}CertID_{0}#> .
@prefix void: <http://rdfs.org/ns/void#> .
@prefix dcterms: <http://purl.org/dc/terms/> .
@prefix dcelem: <http://purl.org/dc/elements/1.1/> .
@prefix np: <http://www.nanopub.org/nschema#> .
@prefix pav: <http://swan.mindinformatics.org/ontologies/1.2/pav/> .
@prefix prov: <http://www.w3.org/ns/prov#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix dcat: <http://www.w3.org/ns/dcat#> .
@prefix schema: <https://schema.org/> .
@prefix orca-x: <http://w3id.org/orca-x#> .
@prefix thispub: <{9}> .

sub:Head {6}
        this: np:hasAssertion sub:assertion ;
        np:hasProvenance sub:provenance ;
        np:hasPublicationInfo sub:pubinfo ;
        a np:Nanopublication .
{7}

sub:assertion {6}
		orca-x:asserts-{0} rdf:singletonPropertyOf orca-x:asserts .
		thispub: orca-x:asserts-{0} "{10}" .
		orca-x:asserts-{0} orca-x:hasConfidenceLevel orca-x:{3} .
{7}

sub:provenance {6}   
        sub:assertion dcterms:author "Certainty Classifier" ;
        dcterms:title "Automated Certainty Classification of Statement from {9}" ;
        dcterms:license <https://creativecommons.org/publicdomain/zero/1.0/> ;
        schema:identifier this: ;
        dcat:distribution sub:assertion ;
        prov:wasDerivedFrom sub:_1 .

        sub:_1   dcelem:format "application/pdf" ;
        a void:Dataset , dcat:Distribution ;
        dcat:downloadURL <{9}> .

{7}

sub:pubinfo {6}
        this: dcterms:created '{5}'^^xsd:date ;
        dcterms:rights <https://creativecommons.org/publicdomain/zero/1.0> ;
        dcterms:rightsHolder <https://orcid.org/0000-0002-9416-6743> ;
        pav:authoredBy "Mario Prieto" , <https://orcid.org/0000-0002-9416-6743> ;
        pav:versionNumber "1" ;
        prov:wasGeneratedBy "Mario Prieto's Certainty Classifier" .

{7}
    """).format(index, pmc, md5.new(CLAIM).hexdigest(), categorization, text, datetime.date.today(), '{', '}', this, url, CLAIM)
            return nanopub, index

    except AssertionError as error:
        print(error)
