import os
import re
import sys
import time
import requests
import pandas as pd
import concurrent.futures
from bs4 import BeautifulSoup
from alive_progress import alive_bar
from colorama import init, Fore, Style

os.system('cls')
try:
    os.chdir(os.path.dirname(__file__))
except:
    pass

# colors

init(autoreset=True)
red = Fore.RED + Style.BRIGHT
green = Fore.GREEN + Style.BRIGHT
yellow = Fore.YELLOW + Style.BRIGHT
blue = Fore.BLUE + Style.BRIGHT
magenta = Fore.MAGENTA + Style.BRIGHT
cyan = Fore.CYAN + Style.BRIGHT
white = Fore.WHITE + Style.BRIGHT

ROOT_URL = 'http://accent.gmu.edu/'
BROWSE_LANGUAGE_URL = 'browse_language.php?function=find&language={}'
WAIT = 1.2
DEBUG = True


def get_htmls(urls):

    htmls = []
    for url in urls:
        if DEBUG:
            print(f'  {white}> {magenta}downloading from {white}{url}')
        htmls.append(requests.get(url).text)
        # time.sleep(WAIT)

    return(htmls)


def build_search_urls(languages):

    return([ROOT_URL+BROWSE_LANGUAGE_URL.format(language) for language in languages])


def parse_p(p_tag):

    text = p_tag.text.replace(' ', '').split(',')
    return([ROOT_URL+p_tag.a['href'], text[0], text[1]])


def get_bio(hrefs):

    htmls = get_htmls(hrefs)
    bss = [BeautifulSoup(html, 'html.parser') for html in htmls]
    rows = []
    bio_row = []
    for bs in bss:
        rows.append([li.text for li in bs.find('ul', 'bio').find_all('li')])
    for row in rows:
        bio_row.append(parse_bio(row))

    return(pd.DataFrame(bio_row))


def parse_bio(row):

    cols = []
    for col in row:
        try:
            tmp_col = re.search((r"\:(.+)", col.replace(' ', '')).group(1))
        except:
            tmp_col = col
        cols.append(tmp_col)
    return(cols)


def multithread(function, list):

    futures = []
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=100)
    for item in list:
        futures.append(executor.submit(function, item))
    with alive_bar(int(len(futures))) as bar:
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
                bar()
            except:
                bar()


def create_dataframe(languages):

    htmls = get_htmls(build_search_urls(languages))
    bss = [BeautifulSoup(html, 'html.parser') for html in htmls]
    persons = []

    for bs in bss:
        for p in bs.find_all('p'):
            if p.a:
                persons.append(parse_p(p))

    df = pd.DataFrame(persons, columns=['href', 'language_num', 'sex'])

    bio_rows = get_bio(df['href'])

    if DEBUG:
        print('loading finished')

    df['birth_place'] = bio_rows.iloc[:, 0]
    df['native_language'] = bio_rows.iloc[:, 1]
    df['other_languages'] = bio_rows.iloc[:, 2]
    df['age_sex'] = bio_rows.iloc[:, 3]
    df['age_of_english_onset'] = bio_rows.iloc[:, 4]
    df['english_learning_method'] = bio_rows.iloc[:, 5]
    df['english_residence'] = bio_rows.iloc[:, 6]
    df['length_of_english_residence'] = bio_rows.iloc[:, 7]

    df['birth_place'] = df['birth_place'].apply(
        lambda x: x[:-6].split(' ')[-2:])

    df['native_language'] = df['native_language'].apply(
        lambda x: x.split(' ')[2])

    df['other_languages'] = df['other_languages'].apply(
        lambda x: x.split(' ')[2:])

    df['age_sex'], df['age'] = df['age_sex'].apply(lambda x: x.split(
        ' ')[2:]), df['age_sex'].apply(lambda x: x.replace('sex:', '').split(',')[1])

    df['age_of_english_onset'] = df['age_of_english_onset'].apply(
        lambda x: float(x.split(' ')[-1]))

    df['english_learning_method'] = df['english_learning_method'].apply(
        lambda x: x.split(' ')[-1])

    df['english_residence'] = df['english_residence'].apply(
        lambda x: x.split(' ')[2:])

    df['length_of_english_residence'] = df['length_of_english_residence'].apply(
        lambda x: float(x.split(' ')[-2]))

    return(df)


if __name__ == '__main__':

    df = None

    destination_file = sys.argv[1]

    try:
        languages = sys.argv[2:]
    except:
        languages = ['mandarin']
        pass

    try:
        df = pd.read_csv(destination_file)
        df = df.append(create_dataframe(
            languages=languages), ignore_index=True)
    except:
        df = create_dataframe(languages=languages)

    df.drop_duplicates(subset='language_num', inplace=True)

    df.to_csv(destination_file, index=False)
