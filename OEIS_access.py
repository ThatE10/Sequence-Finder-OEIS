import requests
from bs4 import BeautifulSoup
import re


def get_results(search_array_doubles):
    print(search_array_doubles)
    if (len(search_array_doubles) == 0):
        return []
    print(search_array_doubles)
    search = [int(num) for num in search_array_doubles]
    string = "%2c".join(str(num) for num in search)

    url = "https://oeis.org/search?q=" + string + "&start=0&fmt=data"
    page = requests.get(url)

    # print(page.text)
    print(url)
    soup = BeautifulSoup(page.content, "html.parser")
    return parse_page(soup)


def parse_page(soup):
    if soup.find("Sorry, but the terms do not match anything in the table."):
        return []
    tables = soup.find_all('table', attrs={'width': '100%'})
    # parsing out the data from the table...
    ##
    # information comes in tables of 2 at a time
    # information goes after table #2 and stops at 23

    """
    print(tables[3].find('a').get_text(strip=True))
    print("################################")
    print((tables[3].find_all('tr')[2]).find('td').find('td').find('td').find('td').get_text(strip=True))
    print("##########################")
    print(tables[3].find_all('tr')[5].find('tt').get_text(strip=True))
    """
    results = []
    for i in range(3, len(tables) - 2, 2):
        print(i)
        results.append(
            {
                'ID': tables[i].find('a').get_text(strip=True),
                'Title': (tables[i].find_all('tr')[2]).find('td').find('td').find('td').find('td').get_text(strip=True),
                'Author': 'Null',
                'Link': 'https://oeis.org/' + tables[i].find('a').get_text(strip=True),
                'Sequence': tables[i].find_all('tr')[5].find('tt').get_text(strip=True).replace(" ", "").split(",")
            }
        )
    return results

