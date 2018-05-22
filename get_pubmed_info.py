import os
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import io
import requests
from xml.etree import ElementTree
import time

max_pmids = 100

pmid_files = {
    "./data/ETC/mutation2pubtator_pmids.txt",
}

def main(pmid_file_addr):
    #pmids_list = load_pmids(pmid_file_addr)
    #print('Total %s PMIDs loaded' % len(pmids_list))
    #check_valid_pmids(pmids_list)
    # #
    # output_file_addr = pmid_file_addr[:-4] + '_toText.txt'
    # pmids_list_to_file(pmids_list, output_file_addr)
    pmid_file_to_file(pmid_file_addr)
    # temp_str = pmids_list_to_articles(pmids_list)
    #print(temp_str)


def pmids_list_to_articles (pmids_list) :
    # take a pmid list as an input, and return a string of processed articles
    print('Changing %s PMIDs to articles.' % len(pmids_list))
    all_urls = url_generation(pmids_list, max_pmids)
    return_articles = ''
    counter=0
    last_progress = -1
    for url1 in all_urls:
        # print('%s / %s' % (counter, len(all_urls)))
        progress = int((counter * 100) / len(all_urls))
        if progress!=last_progress :
            print('Retrieving Article Information from PubMed : %d %%' % progress)
            last_progress = progress
        counter=counter+1
        try :
            returned_xml = url_to_content(url1)
            # returned_xml = preprocess_xml(returned_xml)
            temp_articles = xml_to_text(returned_xml)
        except :
            print('error')
            print(url1)

        return_articles = return_articles + temp_articles
    return return_articles

def pmids_list_to_file (pmids_list, output_file_addr) :
    # take a pmid list as an input, and write a FILE of processed articles
    all_urls = url_generation(pmids_list, max_pmids)

    f_out = open(output_file_addr, 'w')
    counter = 0
    last_progress = -1
    for url1 in all_urls:
        # print('%s / %s' % (counter, len(all_urls)))
        progress = int((counter * 100) / len(all_urls))
        if progress!=last_progress :
            print('Retrieving Article Information from PubMed : %d %%' % progress)
            last_progress = progress
        counter = counter + 1
        returned_xml = url_to_content(url1)
        # returned_xml = preprocess_xml(returned_xml)
        temp_articles = xml_to_text(returned_xml)
        #temp_articles = xml_to_text_only_journal_name(returned_xml)

        f_out.write(temp_articles)
    f_out.close()
    print('results are written in : %s' % output_file_addr)

def pmid_file_to_file (input_file_addr) :
    # take a pmid list FILE as an input, and write a FILE of processed articles
    print('Reading PMID File : %s' % input_file_addr)
    output_file_addr = input_file_addr[:-4] + '_toText.txt'
    pmids_list = load_pmids(input_file_addr)
    print('Total %s PMIDs loaded' % len(pmids_list))
    #pmids_list = list(pmids)
    check_valid_pmids(pmids_list)
    all_urls = url_generation(pmids_list, max_pmids)

    f_out = open(output_file_addr, 'w')
    counter = 0
    last_progress = -1
    for url1 in all_urls:
        # print('%s / %s' % (counter, len(all_urls)))
        progress = int((counter * 100) / len(all_urls))
        if progress!=last_progress :
            print('Retrieving Article Information from PubMed : %d %%' % progress)
            last_progress = progress
        counter = counter + 1
        try:
            returned_xml = url_to_content(url1)
            # returned_xml = preprocess_xml(returned_xml)
            temp_articles = xml_to_text(returned_xml)
            #temp_articles = xml_to_text_only_journal_name(returned_xml)
            f_out.write(temp_articles)
        except :
            print('error')
            print(url1)

    f_out.close()
    print('results are written in : %s' % output_file_addr)


def url_to_content (url) :
    time.sleep(0.5)
    response = requests.get(url)
    # print(url)
    if '[200]' in response :
        print(url)
        print('response : %s' % response)
        sleep_time = 10
        print ("Retrying in %s seconds." % sleep_time)
        time.sleep(sleep_time)  # seconds

    #print('response content : \n %s' % response.content)
    return response.content


def url_generation(pmids, max_pmids):
    urls = []

    pmids_used_counter = 0
    while pmids_used_counter<len(pmids) :
        temp_query = ''
        temp_query_counter=0
        while ((temp_query_counter<max_pmids)&(pmids_used_counter<len(pmids))) :
            if (pmids_used_counter%max_pmids)==0 :
                temp_query=pmids[pmids_used_counter]
            else :
                temp_query = temp_query + '+' + pmids[pmids_used_counter]
            pmids_used_counter=pmids_used_counter+1
            temp_query_counter=temp_query_counter+1
        base_url_temp = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=" + temp_query + "&retmode=xml"
        urls.append(base_url_temp)
    return urls

def load_pmids(pmid_file_addr):
    # Load data from files
    pmid_file = list(io.open(pmid_file_addr, "r").readlines())
    pmids = [s.strip() for s in pmid_file]
    pmids_list=list(pmids)
    return pmids_list

def check_valid_pmids(pmid_list) :
    print("Checking PMIDs.")
    for pmid in pmid_list :
        if (pmid.isdigit() == False) & (pmid != '') :
            print('!!!!!ERROR!!!!!')
            print('%s <== this is NOT a valid pmid. please use only numbers for PMIDs.' % pmid)
            sys.exit()
    print("All the PMIDs seem okay.")

def xml_to_text(xml_raw) :
    xml_raw = preprocess_xml(xml_raw)
    parsed_articles = ''
    root = ElementTree.fromstring(xml_raw)
    for child in root:  # PubmedArticle OR PubmedBookArticle
        _pmid = ''
        _edat = ''
        _journal_title = ''
        _article_types = ''
        _abstract = ''
        _title = ''
        vtitle = ''
        regtitle = ''

        for gchild in child:
            if gchild.tag == "MedlineCitation":
                pmid = gchild.find('PMID').text
                for ggchild in gchild:
                    if ggchild.tag == "Article":
                        for gggchild in ggchild:
                            if gggchild.tag == "Journal":
                                if gggchild.find('ISOAbbreviation') != None :
                                    journal_title = str(gggchild.find('ISOAbbreviation').text)
                                elif gggchild.find('Journal') != None :
                                    journal_title = str(gggchild.find('Journal').text)
                                elif gggchild.find('Title') != None :
                                    journal_title = str(gggchild.find('Title').text)
                                else :
                                    print('No Journal Info for this article PMID %s' % pmid)

                                _journal_title = journal_title.replace(' ', '_')
                            if gggchild.tag == "ArticleTitle":
                                _title = gggchild.text
                            if gggchild.tag == "VernacularTitle":
                                vtitle = gggchild.text
                            if gggchild.tag == "Title":
                                regtitle = gggchild.text

                            if gggchild.tag == "Abstract":
                                for ggggchild in gggchild.findall('AbstractText'):
                                    _abstract = _abstract + str(ggggchild.text) + ' '
                            if gggchild.tag == "PublicationTypeList":
                                for ggggchild in gggchild.findall('PublicationType'):
                                    _article_types = _article_types + ggggchild.attrib['UI'] + ' '

            if gchild.tag == "PubmedData":
                for ggchild in gchild:
                    if ggchild.tag == "History":
                        for gggchild in ggchild:
                            tempstr = gggchild.attrib['PubStatus']
                            if 'entrez' in tempstr:
                                year = gggchild.find('Year').text
                                month = gggchild.find('Month').text
                                if len(month) == 1:
                                    month = '0' + month
                                day = gggchild.find('Day').text
                                if len(day) == 1:
                                    day = '0' + day
                                _edat = year + month + day

                    if ggchild.tag == "ArticleIdList":
                        for gggchild in ggchild:
                            tempstr = gggchild.attrib['IdType']
                            if 'pubmed' in tempstr:
                                _pmid = gggchild.text
        # print("============")
        # print(_pmid)
        # print(_edat)
        # print(_journal_title)
        # print(_article_types)
        # print(_title)
        # print(_abstract)
        if _title == '' :
            _title = vtitle

        if (_title != None) & (_abstract != '') :
            current_article = _pmid + "\t" + _edat + "\t" + _journal_title + "\t" + _article_types + "\t" + _title + "\t" + _abstract
            parsed_articles = parsed_articles + current_article + "\n"
        # if _abstract != '' :
        #     parsed_articles = parsed_articles + current_article + "\n"

    return parsed_articles

def xml_to_text_only_journal_name(xml_raw) :
    xml_raw = preprocess_xml(xml_raw)
    parsed_articles = ''
    root = ElementTree.fromstring(xml_raw)
    for child in root:  # PubmedArticle OR PubmedBookArticle
        _pmid = ''
        _edat = ''
        _journal_title = ''
        _article_types = ''
        _abstract = ''
        _title = ''
        vtitle = ''
        regtitle = ''

        for gchild in child:
            if gchild.tag == "MedlineCitation":
                pmid = gchild.find('PMID').text
                for ggchild in gchild:
                    if ggchild.tag == "Article":
                        for gggchild in ggchild:
                            if gggchild.tag == "Journal":
                                if gggchild.find('Title') != None:
                                    journal_title = str(gggchild.find('Title').text)
                                elif gggchild.find('ISOAbbreviation') != None :
                                    journal_title = str(gggchild.find('ISOAbbreviation').text)
                                elif gggchild.find('Journal') != None :
                                    journal_title = str(gggchild.find('Journal').text)
                                else :
                                    print('No Journal Info for this article PMID %s' % pmid)

                                _journal_title = journal_title.replace(' ', '_')
                            if gggchild.tag == "ArticleTitle":
                                _title = gggchild.text
                            if gggchild.tag == "VernacularTitle":
                                vtitle = gggchild.text
                            if gggchild.tag == "Title":
                                regtitle = gggchild.text

                            if gggchild.tag == "Abstract":
                                for ggggchild in gggchild.findall('AbstractText'):
                                    _abstract = _abstract + str(ggggchild.text) + ' '
                            if gggchild.tag == "PublicationTypeList":
                                for ggggchild in gggchild.findall('PublicationType'):
                                    _article_types = _article_types + ggggchild.attrib['UI'] + ' '

            if gchild.tag == "PubmedData":
                for ggchild in gchild:
                    if ggchild.tag == "History":
                        for gggchild in ggchild:
                            tempstr = gggchild.attrib['PubStatus']
                            if 'entrez' in tempstr:
                                year = gggchild.find('Year').text
                                month = gggchild.find('Month').text
                                if len(month) == 1:
                                    month = '0' + month
                                day = gggchild.find('Day').text
                                if len(day) == 1:
                                    day = '0' + day
                                _edat = year + month + day

                    if ggchild.tag == "ArticleIdList":
                        for gggchild in ggchild:
                            tempstr = gggchild.attrib['IdType']
                            if 'pubmed' in tempstr:
                                _pmid = gggchild.text
        # print("============")
        # print(_pmid)
        # print(_edat)
        # print(_journal_title)
        # print(_article_types)
        # print(_title)
        # print(_abstract)
        if _title == '' :
            _title = vtitle

        if (_title != None) & (_abstract != '') :
            #current_article = _pmid + "\t" + _edat + "\t" + _journal_title + "\t" + _article_types + "\t" + _title + "\t" + _abstract
            current_article = _pmid + "\t" + _edat + "\t" + _journal_title + "\t" + _article_types + "\t" + _title #+ "\t" + _abstract
            parsed_articles = parsed_articles + current_article + "\n"
        # if _abstract != '' :
        #     parsed_articles = parsed_articles + current_article + "\n"

    return parsed_articles


def preprocess_xml (xml_raw) :
    xml_raw = xml_raw.replace('<i>', "")
    xml_raw = xml_raw.replace('</i>', "")
    xml_raw = xml_raw.replace('<b>', "")
    xml_raw = xml_raw.replace('</b>', "")
    xml_raw = xml_raw.replace('<sup>', "")
    xml_raw = xml_raw.replace('</sup>', "")
    xml_raw = xml_raw.replace('<sub>', "")
    xml_raw = xml_raw.replace('</sub>', "")
    xml_raw = xml_raw.replace('&nbsp', "")
    xml_raw = xml_raw.replace('\n', " ")
    while xml_raw.find("  ") != -1:
        xml_raw = xml_raw.replace("  ", " ")

    return xml_raw

def sleep_temp(sleep_time) :
    # Sleeping time before starting the code. #default = 0
    print ("Sleeping for %s seconds." % sleep_time)
    time.sleep(sleep_time) # seconds

if __name__ == '__main__':
    for pmidf in pmid_files :
        print("Processing %s" % pmidf)
        main(pmidf)