import os
import errno
import time
import urllib
import xml.etree.ElementTree as ET

#
# Script for importing metadata for arXiv preprints
#


# Specify folder and query

### hep-ph
filebase = 'datasets/hepph-2018/dataset-'
query = 'set=physics:hep-ph&from=2018-01-01&until=2018-12-31'
# .
# .
# .
# filebase = 'datasets/hepph-2012/dataset-'
# query = 'set=physics:hep-ph&from=2012-01-01&until=2012-12-31'

### hep-th
# filebase = 'datasets/hepth-2018/dataset-'
# query = 'set=physics:hep-th&from=2018-01-01&until=2018-12-31'
# .
# .
# .
# filebase = 'datasets/hepth-2012/dataset-'
# query = 'set=physics:hep-th&from=2012-01-01&until=2012-12-31'


# Create the specified folder if necessary

if not os.path.exists(os.path.dirname(filebase)):
    try:
        os.makedirs(os.path.dirname(filebase))
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise


# Query the database through the Open Archives Initiative OAI-PMH interface
 
print 'PRIMARY REQUEST with filebase = '+filebase+' and query = '+query
urlbase = 'http://export.arxiv.org/oai2?verb=ListRecords&'
url = urlbase+'metadataPrefix=arXiv&'+query
data = urllib.urlopen(url).read()
tree = ET.fromstring(data)
returntype = tree.tag
ns = {'OAI': 'http://www.openarchives.org/OAI/2.0/'}
resum = tree.find('OAI:ListRecords',ns).find('OAI:resumptionToken',ns)

# If the query results are _not_ split over multiple web-pages, 
# then return all results to a single file
if resum == None:
	outfile = open('../'+filebase+'all.txt','w')
	outfile.write(data)
	outfile.close()
	print '...written to file: '+filebase+'all.txt'
	exit()

# If the query results are split over multiple web-pages, 
# then return the results of each page to a separate file
token = resum.text
cursor = resum.attrib['cursor']
size = resum.attrib['completeListSize']
print 'List size = '+size
outfile = open('../'+filebase+cursor+'.txt','w')
outfile.write(data)
outfile.close()
print '...written to file: '+filebase+cursor+'.txt'
time.sleep(10)

while token != None:
	print 'RESUME REQUEST with token = '+token
	url = urlbase+'resumptionToken='+token
	data = urllib.urlopen(url).read()
	tree = ET.fromstring(data)
	returntype = tree.tag
	if returntype == 'html':
		print 'return type = '+returntype
		time.sleep(10)
		continue
	ns = {'OAI': 'http://www.openarchives.org/OAI/2.0/'}
	resum = tree.find('OAI:ListRecords',ns).find('OAI:resumptionToken',ns)
	token = resum.text
	cursor = resum.attrib['cursor']
	size = resum.attrib['completeListSize']
	outfile = open('../'+filebase+cursor+'.txt','w')
	outfile.write(data)
	outfile.close()
	print '...written to file: '+filebase+cursor+'.txt'
	time.sleep(10)