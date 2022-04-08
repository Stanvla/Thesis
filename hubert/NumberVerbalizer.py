# %%
import requests
from num2words import num2words
import pandas as pd
from icecream import ic


def get_request_morphodita(word):
    url = 'http://lindat.mff.cuni.cz/services/morphodita/api/generate?'
    params = dict(
        data=word,
        convert_tagset='pdt_to_conll2009',
        output='json'
    )
    response = requests.get(url, params)
    # return parse_response(response), response.json()['result'][0]
    if response.status_code != 200:
        raise RuntimeError(f'response code is {response.status_code}')
    return parse_response(response), response.json()

def parse_tag(raw_tag):
    raw_tag_lst = raw_tag.split('|')
    result = dict(gen=None, num=None, cas=None)
    allowed = ['C', 'N']
    for sub_tag in raw_tag_lst:
        k, v = sub_tag.split('=')
        k = k.lower()
        if k in result:
            result[k] = v
        if k == 'pos' and v not in allowed:
            return None
    if result['cas'] is None:
        return None
    return result

def parse_response(response):
    dicts = response.json()['result'][0]
    parsed_dicts = []
    for d in dicts:
        tags = parse_tag(d['tag'])
        if tags is None:
            continue
        for k, v in tags.items():
            d[f'tag_{k}'] = v
        del d['tag']
        parsed_dicts.append(d)
    return pd.DataFrame(parsed_dicts)


def gen_all_comb(num_str, df):
    nums = num_str.split()
    dfs = []
    for i, n in enumerate(nums):
        tmp_df = df[df.lemma == n]
        tmp_df = tmp_df.rename(columns={c: f'{c}{i}' for c in tmp_df.columns if c !='tag_cas'})
        dfs.append(tmp_df)

    result = dfs[0]
    for i, df in enumerate(dfs[1:]):
        result = result.merge(df, how='left')
        result.form0 = result.form0 + ' ' + result[f'form{i+1}']
        result = result[['tag_cas', 'form0']]
    str_results = result['form0'].unique()
    return str_results


def num2words_digits_morphodita(num):
    # num can be string or int
    num_verbalized = num2words(num, lang='cz')
    num_verbalized = num_verbalized.replace('jedna', 'jeden')
    df = pd.concat([get_request_morphodita(n) for n in num_verbalized.split()])
    return gen_all_comb(num_verbalized, df)

def num2words_dates_morphodita(num):
    # num can be string or int
    pass

def num2words_floats_morphodita(num):
    pass

# %%
if __name__ == '__main__':
    # %%
    # url = 'http://lindat.mff.cuni.cz/services/morphodita/api/generate?'
    # data_lst = ['sto', 'dvacet', 'jeden', 'dva']
    # dfs = []
    # for data in data_lst:
    #     params = dict(
    #         data=data,
    #         convert_tagset='pdt_to_conll2009',
    #         output='json'
    #     )
    #     response = requests.get(url, params)
    #     x = parse_response(response)
    #     dfs.append(x)
    # df = pd.concat(dfs)
    # %%
    results = []
    for i in range(0, 10 * 10**6):
        s = num2words(i, lang='cz')
        results.extend(s.split(' '))
        if i % 10000 == 0:
            tmp = set(results)
            results = list(tmp)
    # %%
    results = sorted(list(set(results)))
    # %%
    results = [
        'deset',
        'devadesát',
        'devatenáct',
        'devět',
        'devětset',
        'dva',
        'dvacet',
        'dvanáct',
        'dvěstě',
        'jedenáct',
        'jedn',
        'milión',
        'nula',
        'osm',
        'osmdesát',
        'osmnáct',
        'osmset',
        'padesát',
        'patnáct',
        'pět',
        'pětset',
        'sedm',
        'sedmdesát',
        'sedmnáct',
        'sedmset',
        'sto',
        'tisíc',
        'tři',
        'třicet',
        'třináct',
        'třista',
        'čtrnáct',
        'čtyři',
        'čtyřicet',
        'čtyřista',
        'šedesát',
        'šest',
        'šestnáct',
        'šestset'
    ]

    # %%
    for r in results:
        print(r)
        x, y = get_request_morphodita(r)
        print(x)
        print(y)
        print('---'*40)
        # results.append(num2words_digits_morphodita(i))
