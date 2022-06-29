# Membuat file link.py
# File link.py digunakan untuk crawling link tugas akhir
import scrapy

class QuotesSpider(scrapy.Spider):
    name = "quotes"

    def start_requests(self):
        start_urls = ['https://pta.trunojoyo.ac.id/c_search/byprod/10/1']
        for i in range (2,9):
            tambah = 'https://pta.trunojoyo.ac.id/c_search/byprod/10/'+ str(i)
            start_urls.append(tambah)
        for url in start_urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        for i in range(1, 6):
            yield {
                'link':response.css('#content_journal > ul > li:nth-child(' +str(i)+ ') > div:nth-child(3) > a::attr(href)').extract()
            }
# Membuat file link.py
# File link.py digunakan untuk crawling link tugas akhir
import scrapy

class QuotesSpider(scrapy.Spider):
    name = "quotes"

    def start_requests(self):
        start_urls = ['https://pta.trunojoyo.ac.id/c_search/byprod/10/1']
        for i in range (2,9):
            tambah = 'https://pta.trunojoyo.ac.id/c_search/byprod/10/'+ str(i)
            start_urls.append(tambah)
        for url in start_urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        for i in range(1, 6):
            yield {
                'link':response.css('#content_journal > ul > li:nth-child(' +str(i)+ ') > div:nth-child(3) > a::attr(href)').extract()
            }
# Membuat file link.py
# File link.py digunakan untuk crawling link tugas akhir
import scrapy

class QuotesSpider(scrapy.Spider):
    name = "quotes"

    def start_requests(self):
        start_urls = ['https://pta.trunojoyo.ac.id/c_search/byprod/10/1']
        for i in range (2,9):
            tambah = 'https://pta.trunojoyo.ac.id/c_search/byprod/10/'+ str(i)
            start_urls.append(tambah)
        for url in start_urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        for i in range(1, 6):
            yield {
                'link':response.css('#content_journal > ul > li:nth-child(' +str(i)+ ') > div:nth-child(3) > a::attr(href)').extract()
            }
# Membuat file link.py
# File link.py digunakan untuk crawling link tugas akhir
import scrapy

class QuotesSpider(scrapy.Spider):
    name = "quotes"

    def start_requests(self):
        start_urls = ['https://pta.trunojoyo.ac.id/c_search/byprod/10/1']
        for i in range (2,9):
            tambah = 'https://pta.trunojoyo.ac.id/c_search/byprod/10/'+ str(i)
            start_urls.append(tambah)
        for url in start_urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        for i in range(1, 6):
            yield {
                'link':response.css('#content_journal > ul > li:nth-child(' +str(i)+ ') > div:nth-child(3) > a::attr(href)').extract()
            }
