# %%
import re

import numpy as np
import requests
from num2words import num2words
import pandas as pd
from icecream import ic
import copy
import time


def cartesian_product_basic(left, right):
    return left.assign(key=1).merge(right.assign(key=1), on='key').drop('key', 1)


def get_request_morphodita(words, num=True):
    url = 'http://lindat.mff.cuni.cz/services/morphodita/api/generate?'
    params = dict(
        data="\n".join(words.split(' ')),
        convert_tagset='pdt_to_conll2009',
        output='json'
    )
    response = requests.get(url, params)
    # return parse_response(response), response.json()['result'][0]
    if response.status_code != 200:
        raise RuntimeError(f'response code is {response.status_code}')

    result = parse_response(response, num_flag=num)
    return result


def parse_tag_num(raw_tag, num=True):
    raw_tag_lst = raw_tag.split('|')
    result = dict(gen=None, num=None, cas=None)
    allowed = ['C', 'N']
    for sub_tag in raw_tag_lst:
        k, v = sub_tag.split('=')
        k = k.lower()
        if k in result:
            result[k] = v
        if num and k == 'pos' and v not in allowed:
            return None

        if k == 'gra' and v != '1':
            return None
        if k == 'neg' and v == 'N':
            return None
    if result['cas'] is None:
        return None
    return result


def parse_response(response, num_flag=True):
    dicts_num = response.json()['result']
    parsed_dicts = []
    for num in dicts_num:
        for d in num:
            tags = parse_tag_num(d['tag'], num_flag)
            if tags is None:
                continue
            for k, v in tags.items():
                d[f'tag_{k}'] = v
            del d['tag']
            parsed_dicts.append(d)
    df = pd.DataFrame(parsed_dicts)

    if len(df[df.tag_cas == 'X']) != 0:
        new_entries = [df[df.tag_cas != 'X']]
        for x in ['1', '2', '3', '4', '5', '6', '7']:
            tmp_df = df[df.tag_cas == 'X'].copy()
            tmp_df.tag_cas = x
            new_entries.append(tmp_df)
        df = pd.concat(new_entries)

    tags_sub = dict(
        tag_gen=dict(
            fem=['H', 'F', 'Q', 'T', 'X'],
            masc_i=['Z', 'X', 'Y', 'T', 'I'],
            masc_a=['Z', 'X', 'Y', 'M'],
            neut=['Z', 'X', 'Q', 'N', 'H'],
        ),
        tag_num=dict(
            pl=['D', 'P', 'X'],
            sing=['D', 'S', 'X', 'W'],
        )
    )
    for tag_name, tag_dict in tags_sub.items():
        tmp_dfs = []
        df[tag_name].fillna('X', inplace=True)
        for k, val_lst in tag_dict.items():
            tmp_df = df[df[tag_name].isin(val_lst)].copy()
            tmp_df[tag_name] = k
            tmp_dfs.append(tmp_df)
        df = pd.concat(tmp_dfs)

    df['tag'] = df['tag_cas'] + '|' + df['tag_gen'] + '|' + df['tag_num']
    # df['tag'] = df['tag_cas'] + '|' + df['tag_num']
    bad_patterns = [
        r'^tis$', r'tis\s+',
        r'^mil$', r'mil\s+',
        r'^bil$', r'bil\s+',
        r'^hod$', r'^h$',
        r'^min$', r'^m$',
        r'^nul$',
    ]
    for p in bad_patterns:
        mask = df['form'].str.contains(p)
        df = df[~mask]

    return df[['form', 'lemma', 'tag']]


def filter_num_df(df, nums_verb):
    if isinstance(nums_verb, str):
        nums_verb = nums_verb.split(' ')

    filters = dict(
        tag_gen=dict(
            fem=['H', 'F', 'Q', 'T', 'X'],
            masc_i=['Z', 'X', 'Y', 'T', 'I'],
            masc_a=['Z', 'X', 'Y', 'M'],
            neut=['Z', 'X', 'Q', 'N', 'H'],
        ),
        tag_num=dict(
            plural=['D', 'P', 'X'],
            singular=['D', 'S', 'X', 'W'],
        )
    )

    filtered_dfs = []
    for tag_name, tag_dict in filters.items():
        for k, val_lst in tag_dict.items():
            tmp_df = df.copy()
            for i in range(len(nums_verb)):
                tag = f'{tag_name}{i}'
                tmp_df[tag].fillna('X', inplace=True)
                tmp_df = tmp_df[tmp_df[tag].isin(val_lst)]
            filtered_dfs.append(tmp_df)
    return pd.concat(filtered_dfs)


def merge_tag_subset(df1, df2, df1_src_col, df2_src_col, tag_subset, trg_col='time'):
    def new_tag(df):
        df['tmp_tag'] = ''
        for i, t in enumerate(tag_subset):
            df['tmp_tag'] += df[t]
            if i != len(tag_subset) -1:
                df['tmp_tag'] += '|'
        return df

    df1 = new_tag(df_tag_split(df1))
    df2 = new_tag(df_tag_split(df2))

    result = df1.merge(df2, how='left', on='tmp_tag')
    result[trg_col] = result[df1_src_col] + ' ' + result[df2_src_col]
    return result[result[trg_col].notna()].drop_duplicates()


def gen_all_comb(num_str, source_df):
    nums = num_str.split(' ')
    dfs = []
    for i, n in enumerate(nums):
        tmp_df = source_df[source_df.lemma == n]
        tmp_df = tmp_df.rename(columns={c: f'{c}{i}' for c in tmp_df.columns if c != 'tag'})
        dfs.append(tmp_df)

    result = dfs[0]
    for i, df in enumerate(dfs[1:]):
        result = merge_tag_subset(result.copy(), df.copy(), 'form0', f'form{i+1}', tag_subset=['cas', 'gen'], trg_col='form0')
        result['tag'] = result.tag_y
        result = result[['tag', 'form0']]
        # result = result.merge(df, how='left', on='tag')
        # result.form0 = result.form0 + ' ' + result[f'form{i+1}']
        # result = result.dropna()

    return result


def n2w(n):
    return num2words(n, lang='cz')


def expand_group(group, result, ord_num, german_version):
    highest = None
    if ord_num[group['units']] != 0:
        if ord_num[group['tens']] != 0:
            tmp = ord_num[group['tens']] * 10 + ord_num[group['units']]
            # basic variant in right order
            result[group['tens']].append(n2w(tmp))
            # alternative variant with inverse order
            if german_version and 20 < tmp < 100 and tmp % 10 != 0:
                result[group['tens']].append(f'{n2w(ord_num[group["units"]])}a{n2w(ord_num[group["tens"]] * 10)}')
            highest = group['tens']
        else:
            result[group['units']].append(n2w(ord_num[group['units']]))
            highest = group['units']
    else:
        if ord_num[group['tens']] != 0:
            result[group["tens"]].append(n2w(ord_num[group["tens"]] * 10))
            highest = group['tens']

    if ord_num[group['hundreds']] != 0:
        # basic variant without whitespace
        tmp_results = [n2w(ord_num[group["hundreds"]] * 100)]
        # variant with a whitespace
        if ord_num[group["hundreds"]] > 1:
            tmp_results.append(n2w(ord_num[group['hundreds']]) + ' ' + n2w(100))
        for tmp_r in tmp_results:
            if highest is not None:
                for x in result[highest]:
                    result[group['hundreds']].append(tmp_r + ' ' + x)
            else:
                result[group["hundreds"]].append(tmp_r)
        result[highest] = []
        highest = group['hundreds']
    return result, highest


def get_orders(num):
    orders = [
        'units_units', 'tens_units', 'hundreds_units',
        'units_thousands', 'tens_thousands', 'hundreds_thousands',
        'units_millions', 'tens_millions', 'hundreds_millions',
        'units_milliards', 'tens_milliards', 'hundreds_milliards',
        'units_billions', 'tens_billions', 'hundreds_billions',
    ]
    orders_groups = {
        'units': orders[0:3],
        'thousands': orders[3:6],
        'millions': orders[6:9],
        'milliards': orders[9:12],
        'billions': orders[12:15]
    }

    orders_groups_names = {
        'units': '',
        'thousands': 'tisíc',
        'millions': 'milión',
        'milliards': 'miliarda',
        'billions': 'bilión'
    }

    ord_base = {k: 10 ** i for i, k in enumerate(orders)}
    ord_num = {k: 0 for k in ord_base}
    acc = 0
    for order_name, order_val in ord_base.items():
        ord_num[order_name] = (num % (order_val * 10) - acc) // order_val
        acc += ord_num[order_name]

    return orders_groups, orders_groups_names, ord_base, ord_num


def alt_hundreds_1100_2000(num, orders_groups, ord_base, ord_num, german_version):
    results = []
    types = ['units', 'tens', 'hundreds']
    tmp_ord_num = copy.deepcopy(ord_num)
    if 1100 <= num < 2000:
        hundreds = tmp_ord_num['hundreds_units'] + tmp_ord_num['units_thousands'] * 10
        tmp_ord_num['units_thousands'] = 0
        tmp_ord_num['hundreds_units'] = 0
        result_dict = {k: [] for k in ord_base}
        group = {k: v for k, v in zip(types, orders_groups["units"])}
        result_dict, highest = expand_group(group, result_dict, tmp_ord_num, german_version)

        prefix = n2w(hundreds) + ' ' + n2w(100)
        if highest is not None:
            results.extend([prefix + ' ' + x for x in result_dict[highest]])
        else:
            results = [prefix]
    return results


def natural_num_verbalization(num, orders_groups, orders_groups_names, ord_base, ord_num, german_version=True, alt_hundreds=True):
    # orders_groups = {'units': ['units_units', 'tens_units', 'hundreds_units'],
    #  'thousands': ['units_thousands', 'tens_thousands', 'hundreds_thousands'],
    #  'millions': ['units_millions', 'tens_millions', 'hundreds_millions'],
    #  'milliards': ['units_milliards', 'tens_milliards', 'hundreds_milliards'],
    #  'billions': ['units_billions', 'tens_billions', 'hundreds_billions']}

    # orders_group_names = {'units': '',
    #  'thousands': 'tisíc',
    #  'millions': 'milión',
    #  'milliards': 'miliarda',
    #  'billions': 'bilión'}

    # ord_base = {'units_units': 1,
    #  'tens_units': 10,
    #  'hundreds_units': 100,
    #  'units_thousands': 1000,
    #  'tens_thousands': 10000,
    #  'hundreds_thousands': 100000,
    #  'units_millions': 1000000,
    #  'tens_millions': 10000000,
    #  'hundreds_millions': 100000000,
    #  'units_milliards': 1000000000,
    #  'tens_milliards': 10000000000,
    #  'hundreds_milliards': 100000000000,
    #  'units_billions': 1000000000000,
    #  'tens_billions': 10000000000000,
    #  'hundreds_billions': 100000000000000}

    # num can be string or int
    if isinstance(num, str):
        num = int(num)

    result_dict = {k: [] for k in ord_base}
    result_lst = ['']

    types = ['units', 'tens', 'hundreds']

    for order_name, order_lst in orders_groups.items():
        if ord_base[order_lst[0]] > num:
            break
        # example of a group {units: units_thousands, tens: tens_thousands, hundreds: hundreds_thousands}
        group = {k: v for k, v in zip(types, order_lst)}

        # when expanding a group we generate all variants of verbal forms of a number restricted to a group
        # result_dict[highest] contains all verbal versions of a number restricted to a group
        result_dict, highest = expand_group(group, result_dict, ord_num, german_version)

        if highest is not None:
            # first add group name to all variants of a number restricted to a group
            # dva -> dva tisíc
            group_with_name_lst = []
            for x in result_dict[highest]:
                # todo check
                # this condition will add a variant without number
                if 'units' in highest and orders_groups_names[order_name] != '':
                    group_with_name_lst.append(orders_groups_names[order_name])
                group_with_name_lst.append(x + ' ' + orders_groups_names[order_name])
            result_dict[highest] = group_with_name_lst

            # now prepend new group verbalization to the results
            result_lst = [x + ' ' + r for x in group_with_name_lst for r in result_lst]

    # if requested generate alternative version for hundreds and thousands: 1100 -> jedenact set
    if alt_hundreds:
        result_lst.extend(alt_hundreds_1100_2000(num, orders_groups, ord_base, ord_num, german_version))

    # num2word library almost always generates jedna instead of jeden, so replace it back
    result_lst = [r.replace('jedna ', 'jeden ').replace('jednaa', 'jedena') for r in result_lst]
    result_lst = [re.sub(r'jedna$', 'jeden', r) for r in result_lst]

    return [r.rstrip() for r in result_lst]


def n2w_cardinal_morphodita(num, german_version=True):
    def get_all_variants_morphodita(num_lst):
        result = []
        for verb_num in num_lst:
            verb_num = verb_num.rstrip()
            df = get_request_morphodita(verb_num)
            result.append(gen_all_comb(verb_num, df))
        return pd.concat(result).reset_index(drop=True)

    # Ve větné souvislosti se víceslovné číslovkové výrazy zpravidla skloňují. Máme více možností:
    #
    #   case_1) Skloňujeme všechny části výrazu.
    #    - "Teploty vystoupí k 27 °C" čteme a píšeme k dvaceti sedmi stupňům nebo i obráceně sedmadvaceti stupňům (v takovém případě píšeme číslovku dohromady);
    #    - "před 365 lety"  před třemi sty šedesáti pěti (před třemi sty pětašedesáti) lety;
    #    - "bez 1 847 Kč" bez tisíce osmi set čtyřiceti sedmi korun.
    #
    #   case_2) Skloňujeme jen jméno počítaného předmětu a část číslovkového výrazu (obvykle řád desítek a jednotek), zbytek ponecháváme nesklonný.
    #    - "před 365 lety" můžeme také přečíst před tři sta šedesáti pěti (pětašedesáti) lety
    #    - "bez 1 847 Kč" bez tisíc osm set čtyřiceti sedmi (sedmačtyřiceti) korun;
    #    - "o 1 358 423 dokladech" o milion tři sta padesát osm tisíc čtyři sta  dvaceti třech (třiadvaceti) dokladech.
    #
    #   case_3) V některých situacích, většinou v mluvené řeči, např. při diktování a při početních operacích s čísly,
    #   může zůstat celý víceslovný číslovkový výraz neskloňovaný,
    #    - např. k tisíc sedm set dvacet dva korunám.

    # case_1 will be solved by generating all variants using morphodita
    # case_2 for tens and units use morphodita, the rest will be generated from n2w (exceptions is if 1100 <= num < 2000)
    # case_3 will be included in case_2 by default

    if isinstance(num, str):
        num = int(num)
    orders_groups, orders_groups_names, ord_base, ord_num = get_orders(num)
    if sum(ord_num.values()) == 0:
        case_1 = ['nula']
    else:
        case_1 = natural_num_verbalization(num, orders_groups, orders_groups_names, ord_base, ord_num, german_version)
    all_results = get_all_variants_morphodita(case_1)

    # if number is smaller than 100, then case_2 and case_3 are done by default in case_1
    case_2 = None
    if num > 100 and num % 100 != 0:
        # need to handle tisic/jedna tisic, milion/jeden milion,
        new_num = num % 100
        num_reminder = num - new_num
        rest = n2w(num_reminder)
        case_2_copy = None
        base_tag = '1|sing'
        if new_num != 0:
            orders_groups, orders_groups_names, ord_base, ord_num = get_orders(new_num)
            case_2 = natural_num_verbalization(new_num, orders_groups, orders_groups_names, ord_base, ord_num, german_version, alt_hundreds=False)
            case_2 = get_all_variants_morphodita(case_2)
            case_2_copy = case_2.copy()
            case_2.form0 = rest + ' ' + case_2.form0
        else:
            case_2 = pd.DataFrame(dict(tag=base_tag, form0=rest))

        if num // 10**3 == 1:
            tmp_df = case_2.copy()
            tmp_df.form0 = tmp_df.form0.str.replace('tisíc', 'jeden tisíc')
            case_2 = pd.concat([case_2, tmp_df])
        if num // 10**6 == 1:
            tmp_df = case_2.copy()
            tmp_df.form0 = tmp_df.form0.str.replace('milion', 'jeden milion')
            case_2 = pd.concat([case_2, tmp_df])
        if num // 10**9 == 1:
            tmp_df = case_2.copy()
            tmp_df.form0 = tmp_df.form0.str.replace('miliarda', 'jedna miliarda')
            case_2 = pd.concat([case_2, tmp_df])
        if num // 10**12 == 1:
            tmp_df = case_2.copy()
            tmp_df.form0 = tmp_df.form0.str.replace('bilion', 'jeden bilion')
            case_2 = pd.concat([case_2, tmp_df])

        # case for 1100 <= num < 2000
        orders_groups, orders_groups_names, ord_base, ord_num = get_orders(num_reminder)
        alt_hundreds = alt_hundreds_1100_2000(num_reminder, orders_groups, ord_base, ord_num, german_version=False)
        if alt_hundreds:
            if case_2_copy is not None:
                tmp_dfs = []
                for alt_h in alt_hundreds:
                    tmp_df = case_2_copy.copy()
                    tmp_df.form0 = alt_h.replace('sto', 'set') + ' ' + tmp_df.form0
                    tmp_dfs.append(tmp_df)
                tmp_dfs = pd.concat(tmp_dfs)
                case_2 = pd.concat([case_2, tmp_dfs])
            else:
                tmp_df = pd.DataFrame([dict(form0=alh.replace('sto', 'set'), tag=base_tag) for alh in alt_hundreds])
                case_2 = pd.concat([case_2, tmp_df])

    if case_2 is not None:
        all_results = pd.concat([all_results, case_2])
    return all_results.drop_duplicates(ignore_index=True).sort_values('tag').reset_index(drop=True)


def num_with_noun(num_df, noun_df):
    valid_cols = ['time', 'tag']
    result_dfs = []
    num_df = df_tag_split(num_df)
    noun_df = df_tag_split(noun_df)
    tags = ['num', 'cas', 'gen']

    # case 1: join by tag
    tmp_df1 = num_df.merge(noun_df, how='left', on='tag')
    tmp_df1['time'] = tmp_df1.form0 + ' ' + tmp_df1.form
    result_dfs.append(tmp_df1[tmp_df1.time.notna()][valid_cols])

    # case 2: now for each num add noun in the second case in correct num and gen
    tmp_noun_df = noun_df[(noun_df.cas == '2')].drop(['tag'], 1)
    tmp_noun_df = tmp_noun_df.drop_duplicates()

    tmp_df2 = num_df.merge(tmp_noun_df, how='left', on='gen')
    tmp_df2['time'] = tmp_df2.form0 + ' ' + tmp_df2.form
    result_dfs.append(tmp_df2[tmp_df2.time.notna()][valid_cols])

    return pd.concat(result_dfs).drop_duplicates()


def n2w_ordinal_morphodita(num, base_variant_only=True):
    # Den v měsíci se v češtině označuje řadovou číslovkou
    # Je-li ve větě označení dne v pozici podmětu nebo předmětu, používá se někdy i tvar, v němž se název měsíce dostává do základního tvaru (1. pádu),
    # jako by řadová číslovka označovala pořadí měsíce a ne pořadí dne v měsíci („Patnáctý srpen je den mého narození. Proto se těším na každý patnáctý srpen.“)
    # num can be string or int
    if isinstance(num, str):
        num = int(num)
    if num > 999:
        raise NotImplementedError(f'Can generate only ordinal numbers smaller than 1000, passed num={num}')

    base_variants = {
        1: ['první', 'prvý'], 2: ['druhý'], 3: ['třetí'], 4: ['čtvrtý'], 5: ['pátý'], 6: ['šestý'], 7: ['sedmý'], 8: ['osmý'], 9: ['devátý'],
        10: ['desátý'], 11: ['jedenáctý'], 12: ['dvanáctý'], 13: ['třináctý'], 14: ['čtrnáctý'], 15: ['patnáctý'], 16: ['šestnáctý'], 17: ['sedmnáctý'], 18: ['osmnáctý'], 19: ['devatenáctý'],
        20: ['dvacátý'], 30: ['třicátý'], 40: ['čtyřicátý'], 50: ['padesátý'], 60: ['šedesátý'], 70: ['sedmdesátý'], 80: ['osmdesátý'], 90: ['devadesátý'],
        100: ['stý'], 200: ['dvoustý'], 300: ['třístý'], 400: ['čtyřstý'], 500: ['pětistý'], 600: ['šestistý'], 700: ['sedmistý'], 800: ['osmistý'], 900: ['devítistý'],
    }
    # n2w + dict[20]/dict[30]...
    for i in range(21, 100):
        if i % 10 == 0:
            continue
        units = i % 10
        tens = i - units
        if i not in base_variants:
            base_variants[i] = []

        for t in base_variants[tens]:
            for u in base_variants[units]:
                base_variants[i].append(t + ' ' + u)
            base_variants[i].append(n2w(units).replace('jedna', 'jeden') + 'a' + t)

    for i in range(101, 1000):
        if i % 100 == 0:
            continue
        tens_units = i % 100
        hundreds = i - tens_units
        base_variants[i] = [h + ' ' + v for v in base_variants[tens_units] for h in base_variants[hundreds]]
    if base_variant_only:
        return base_variants[num]
    else:
        return pd.concat([get_request_morphodita(v, num=False) for v in base_variants[num]])


def df_tag_split(df):
    df[['cas', 'gen', 'num']] = df.tag.str.split('|', expand=True)
    return df


def add_words_df(col, words, df, prefix=True):
    dfs = []
    for w in words:
        tmp_df = df.copy()
        if prefix:
            tmp_df[col] = w + ' ' + tmp_df[col]
        else:
            tmp_df[col] += ' ' + w
        dfs.append(tmp_df)
    return pd.concat(dfs)


def n2w_times_morphodita(hours, minutes, unique=False):
    valid_columns = ['tag', 'time']
    hours_int = int(hours)
    minutes_int = int(minutes)
    hours_morph_df = n2w_cardinal_morphodita(hours)

    # handle hours_int % 12
    if hours_int > 12:
        hours_morph_df = pd.concat([hours_morph_df, n2w_cardinal_morphodita(hours_int % 12)])

    minutes_morph_df = n2w_cardinal_morphodita(minutes)
    hours_words_df = get_request_morphodita('hodina', num=False)
    minutes_words_df = get_request_morphodita('minuta', num=False)

    hours_morph_df = df_tag_split(hours_morph_df)
    hours_morph_df.tag = hours_morph_df.cas + '|' + hours_morph_df.gen

    minutes_morph_df = df_tag_split(minutes_morph_df)
    minutes_morph_df.tag = minutes_morph_df.cas + '|' + minutes_morph_df.gen

    # handling zeros
    if '0' == minutes[0]:
        if minutes == '00':
            minutes_morph_df.form0 = 'nula nula'
        else:
            tmp_df = minutes_morph_df.copy()
            tmp_df.form0 = 'nula ' + tmp_df.form0
            minutes_morph_df = pd.concat([minutes_morph_df, tmp_df])


    # case: no "hours", "minutes", both numbers in the same case
    nums_by_case_short_df = hours_morph_df.merge(minutes_morph_df, how='left', on='tag')
    nums_by_case_short_df['time'] = nums_by_case_short_df.form0_x + ' ' + nums_by_case_short_df.form0_y
    # can have "hodin" at the end
    tmp_df = nums_by_case_short_df.copy()
    tmp_df.time = tmp_df.time + ' ' + 'hodin'
    nums_by_case_short_df = pd.concat([nums_by_case_short_df, tmp_df])
    nums_by_case_short_df = nums_by_case_short_df[valid_columns]
    nums_by_case_short_df = nums_by_case_short_df.dropna()

    # case: two groups, hours_int with hours_word in the correct case and minutes_int and minutes_word in the correct case
    hours_int_word_df = n2w_cardinal_morphodita(hours_int)
    if hours_int == 1:
        hours_int_word_df = add_words_df('form0', ['hodina', 'hodinu'], hours_int_word_df, False)
    elif 1 < hours_int < 5:
        hours_int_word_df.form0 += ' hodiny'
    else:
        hours_int_word_df.form0 += ' hodin'

    # also handle hours_int % 12
    if hours_int > 12:
        hours_int_alt = hours_int % 12
        tmp_df = n2w_cardinal_morphodita(hours_int_alt)
        if hours_int_alt == 1:
            tmp_df = add_words_df('form0', ['hodina', 'hodinu'], tmp_df, False)
        elif 1 < hours_int_alt < 5:
            tmp_df.form0 += ' hodiny'
        else:
            tmp_df.form0 += ' hodin'
        hours_int_word_df = pd.concat([hours_int_word_df, tmp_df])
    hours_int_word_df = df_tag_split(hours_int_word_df)
    hours_int_word_df.tag = hours_int_word_df.cas + '|' + hours_int_word_df.gen

    minutes_int_word_df = minutes_morph_df.copy()
    if minutes_int == 1:
        minutes_int_word_df = add_words_df('form0', ['minuta', 'minutu'], minutes_int_word_df, False)
    elif 1 < minutes_int < 5:
        minutes_int_word_df.form0 += ' minuty'
    else:
        minutes_int_word_df.form0 += ' minut'

    nums_by_case_full_df = hours_int_word_df.merge(minutes_int_word_df, how='left', on='tag')
    nums_by_case_full_df['time'] = nums_by_case_full_df.form0_x + ' ' + nums_by_case_full_df.form0_y

    hours_int_word_conj_df = hours_int_word_df.copy()
    hours_int_word_conj_df.form0 += ' a'

    tmp_df = hours_int_word_conj_df.merge(minutes_int_word_df, how='left', on='tag')
    tmp_df['time'] = tmp_df.form0_x + ' ' + tmp_df.form0_y

    nums_by_case_full_df = pd.concat([nums_by_case_full_df, tmp_df])
    nums_by_case_full_df = nums_by_case_full_df[valid_columns]
    nums_by_case_full_df = nums_by_case_full_df.dropna()


    # case: hours only (can have word "hodin")
    hours_only_df = hours_morph_df.copy()
    hours_only_df = pd.concat([hours_only_df, hours_int_word_df])
    hours_only_df['time'] = hours_only_df.form0
    hours_only_df = hours_only_df[valid_columns]
    hours_only_df = hours_only_df.dropna()

    if hours_only_df.time.isnull().values.any():
        raise RuntimeError('nan values in the hours_only')

    result_df = pd.concat([nums_by_case_short_df, nums_by_case_full_df, hours_only_df])

    # case: ordinal numbers
    if minutes_int != 0:
        hours_ordinal_df = n2w_ordinal_morphodita(hours_int, base_variant_only=False)
        hours_ordinal_word_df = hours_ordinal_df.merge(hours_words_df, how='left', on='tag')
        hours_ordinal_word_df['time'] = hours_ordinal_word_df.form_x + ' ' + hours_ordinal_word_df.form_y

        minutes_ordinal_df = n2w_ordinal_morphodita(minutes_int, base_variant_only=False)
        minutes_ordinal_word_df = minutes_ordinal_df.merge(minutes_words_df, how='left', on='tag')
        minutes_ordinal_word_df['time'] = minutes_ordinal_word_df.form_x + ' ' + minutes_ordinal_word_df.form_y

        tmp_df = minutes_ordinal_word_df.copy()
        tmp_df.time = 'a ' + tmp_df.time
        minutes_ordinal_word_df = pd.concat([minutes_ordinal_word_df, tmp_df])

        ordinal_df = hours_ordinal_word_df.merge(minutes_ordinal_word_df, how='left', on='tag')
        ordinal_df['time'] = ordinal_df.time_x + ' ' + ordinal_df.time_y
        ordinal_df = df_tag_split(ordinal_df)
        ordinal_df = ordinal_df[ordinal_df.cas.isin(['2', '4']) & (ordinal_df.gen == 'fem')]
        ordinal_df = ordinal_df[valid_columns]
        result_df = pd.concat([result_df, ordinal_df])

    # case: quarters and halves
    if minutes_int % 15 == 0 and minutes_int > 0:
        hours_int_alt = (hours_int + 1) % 12
        if hours_int_alt == 0:
            hours_int_alt = 12

        if minutes_int == 30:
            # handle halves
            if hours_int_alt == 1:
                hours_int_alt_df = n2w_cardinal_morphodita(hours_int_alt)
                hours_int_alt_df['form'] = hours_int_alt_df.form0
            else:
                hours_int_alt_df = n2w_ordinal_morphodita(hours_int_alt, base_variant_only=False)
            # extract only correct forms
            hours_int_alt_df = hours_int_alt_df[hours_int_alt_df.tag == '2|fem|sing']
            hours_int_alt_df['time'] = 'půl ' + hours_int_alt_df.form
            alt_df = hours_int_alt_df[valid_columns]
        else:
            # handle quarters
            quarters = minutes_int // 15
            quarters_word_df = df_tag_split(get_request_morphodita('čtvrt', num=False))
            quarters_word_df = quarters_word_df[quarters_word_df.cas.isin(['2', '4'])]
            if quarters == 1:
                quarters_word_df = quarters_word_df[quarters_word_df.num == 'sing']
            else:
                quarters_word_df = quarters_word_df[quarters_word_df.num == 'pl']
            # use preposition "na"
            quarters_word_df.form = quarters_word_df.form + ' na'

            hours_int_alt_df = df_tag_split(n2w_cardinal_morphodita(hours_int_alt))
            hours_int_alt_df = hours_int_alt_df[(hours_int_alt_df.cas == '4') & (hours_int_alt_df.gen == 'fem')]
            unique_hours_alt = hours_int_alt_df.form0.unique()
            if len(unique_hours_alt) == 0:
                raise RuntimeError('Quarters verbalization: no num verbalization for hours, unique_hours_alt is empty.')
            alt_df = add_words_df('form', unique_hours_alt, quarters_word_df, prefix=False)

            if quarters > 1:
                cardinal_num_df = n2w_cardinal_morphodita(quarters)
                alt_df = add_words_df('form0', alt_df.form.unique(), cardinal_num_df, prefix=False)
                alt_df['form'] = alt_df.form0

            alt_df['time'] = alt_df.form
            alt_df = alt_df.dropna()

            alt_df = alt_df[valid_columns]

        result_df = pd.concat([result_df, alt_df])
    result_df = result_df.drop_duplicates()

    if unique:
        return result_df.time.unique()
    return result_df


def n2w_floats_morphodita(whole, decimal, unique=False):
    valid_cols = ['tag', 'time']
    whole_int = int(whole)
    decimal_int = int(decimal)
    result_dfs = []

    # handle decimals
    decimal_morpho_df = n2w_cardinal_morphodita(decimal_int, german_version=False)
    leading_zeros = ' '.join((len(decimal) - len(decimal.lstrip('0'))) * ['nula'])
    if leading_zeros != '':
        decimal_morpho_with_zeros_df = add_words_df('form0', [leading_zeros], decimal_morpho_df.copy(), prefix=True)
        decimal_morpho_df = pd.concat([decimal_morpho_df, decimal_morpho_with_zeros_df])

    decimal_int_word_df = None
    if len(decimal) < 4:
        if len(decimal) == 1:
            decimal_word = 'desetina'
        elif len(decimal) == 2:
            decimal_word = 'setina'
        else:
            decimal_word = 'tisícina'

        decimal_word_df = get_request_morphodita(decimal_word, num=False)
        decimal_int_word_df = num_with_noun(decimal_morpho_df.copy(), decimal_word_df.copy())
        if whole_int == 0:
            result_dfs.append(decimal_int_word_df.copy())
    # dvěstě třicet tř
    # handle whole part
    whole_morpho_df = n2w_cardinal_morphodita(whole_int, german_version=False)
    whole_word_df = get_request_morphodita('celý', num=False)

    if whole_int == 0:
        alt_whole_morpho_df = get_request_morphodita('žádný', num=False)
        alt_whole_morpho_df = df_tag_split(alt_whole_morpho_df)
        alt_whole_morpho_df = alt_whole_morpho_df[(alt_whole_morpho_df.gen == 'fem') & (alt_whole_morpho_df.num == 'sing')]
        alt_whole_morpho_df['form0'] = alt_whole_morpho_df.form
        whole_morpho_df = pd.concat([whole_morpho_df, alt_whole_morpho_df])
        whole_morpho_df = whole_morpho_df.drop('form', 1)

    whole_int_word_df = num_with_noun(whole_morpho_df.copy(), whole_word_df.copy())

    # case 0: halves
    if decimal_int == 5:
        tmp_df = pd.concat([whole_morpho_df, whole_int_word_df])
        tmp_df = add_words_df('form0', ['a půl'], tmp_df.copy(), prefix=False)
        tmp_df['time'] = tmp_df.form0
        result_dfs.append(tmp_df[valid_cols].dropna())

    # prepare tags, num in tag is not useful,
    # since whole part can be singular and decimal part can be plural
    new_tags = ['cas']
    # case 1: whole_int decimal_int
    short_df = merge_tag_subset(whole_morpho_df.copy(), decimal_morpho_df.copy(), 'form0_x', 'form0_y', new_tags)
    short_df['tag'] = short_df['tag_x']
    result_dfs.append(short_df[valid_cols])

    # case 2: whole_int whole_word decimal_int
    mid_df = merge_tag_subset(whole_int_word_df.copy(), decimal_morpho_df.copy(), 'time', 'form0', new_tags)
    mid_df['tag'] = mid_df['tag_x']
    result_dfs.append(mid_df[valid_cols])

    # case 3: whole_int whole_word decimal_int decimal_word
    if decimal_int_word_df is not None:
        long_df = merge_tag_subset(whole_int_word_df.copy(), decimal_int_word_df.copy(), 'time_x', 'time_y', new_tags)
        long_df['tag'] = long_df['tag_x']
        result_dfs.append(long_df[valid_cols])

    result_df = pd.concat(result_dfs)
    if unique:
        return result_df.time.unique()
    return result_df


def test_time():
    tests = dict(
        only_num_same_case=[
            ['12.50', 'dvanácti padesáti'],
            ['12.50', 'dvanácti padesáti'],
            ['12.50', 'dvanácti padesáti hodin'],
            ['12.45', 'dvanáct čtyřicet pět'],
            ['9.30', 'devíti třiceti'],
            ['14.30', 'čtrnácti třiceti'],
            ['12.30', 'dvanáct třicet'],
            ['12.50', 'dvanáct padesát'],
            ['21.21', 'jedenadvaceti jedenadvaceti'],
            ['17.45', 'sedmnácti čtyřiceti pěti'],
            ['12.15', 'dvanácti patnácti'],
            ['10.15', 'deseti patnácti'],
            ['10.01', 'deset jedna'],
            ['13.05', 'třináct nula pět'],
            ['13.02', 'třináct dvě'],
            ['13.03', 'třináct tři'],
        ],
        two_groups=[
            ['14.02', 'čtrnáct hodin a dvě minuty'],
            ['13.02', 'třináct hodin dvě minuty'],
            ['13.03', 'třináct hodin tři minuty'],
            ['13.05', 'třináct hodin pět minut'],
            ['18.45', 'osmnácti hodin čtyryceti pěti minut'],
            ['23.15', 'dvaceti tří hodin patnácti minut'],
            ['11.15', 'jedenácti hodin patnácti minut'],
            ['12.45', 'dvanáct hodin čtyřicet pět minut'],
            ['12.50', 'dvanáct hodin padesát minut'],
            ['13.15', 'třináct hodin patnáct minut'],
            ['13.05', 'třináct hodin pět minut'],
            ['13.05', 'třináct hodin nula pět minut'],
            ['12.01', 'dvanáct hodin jedna minuta'],
        ],
        zeros=[
            ['14.04', 'čtrnáct nula čtyři'],
            ['13.00', 'třináct nula nula'],
            ['19.00', 'devatenácti nula nula'],
            ['13.00', 'třinácti nula nula hodin'],
        ],
        hours_only=[
            ['13.00', 'jednu hodinu'],
            ['14.00', 'dvě hodiny'],
            ['15.00', 'tři hodiny'],
            ['16.00', 'čtyři hodiny'],
            ['15.00', 'patnáct hodin'],
            ['12.50', 'dvanáct hodin'],
            ['14.00', 'čtrnáct hodin'],
            ['11.00', 'jedenácti'],
        ],
        quarters=[
            ['12.45', 'tří čtvrti na jednu'],
            ['14.45', 'tří čtvrti na tři'],
            ['11.45', 'tří čtvrti na dvanáct'],
            ['11.15', 'čtvrt na dvanáct'],
        ],
        halves=[
            ['12.30', 'půl jedné'],
            ['17.30', 'půl šesté'],
            ['15.30', 'půl čtvrté'],
            ['11.30', 'půl dvanácté'],
            ['10.30', 'půl jedenácté']
        ],
        ordinal=[
            ['16.30', 'šestnáctou hodinu třicátou minutu'],
            ['17.30', 'sedmnáctou hodinu třicátou minutu'],
        ],
    )
    for test_name, values in tests.items():
        print(test_name)
        for i, (k, v) in enumerate(values):
            hours, minutes = k.split('.')
            result = n2w_times_morphodita(hours, minutes, unique=True)
            if v not in result:
                print(f'XX: {k:>5}, {v:<40} {len(result):>6}')
            else:
                print(f'OK: {k:>5}, {v:<40} {len(result):>6}')
            time.sleep(1)
        print('---' * 40)


def test_float():

    tests = [
        # todo check
        #   ["1 , 82", "jedním celým osmdesáti dvěma setinamy"],

        # todo hard
        #   ["1 , 163", "jedna celá sto šedesát třech"],

        #  ["0 , 0042", "nula celá nula nula čtyřicet dva"],
        #  ["2 , 072", "dvě celé nula sedmdesát dva"],
        #  ["9 , 209", "devět celých dvěstě devět"],
        #  ["32 , 5", "třicet dva a půl"],
        #  ["1 , 233", "jedna celá dvěstě třicet tři"],
        #  ["3 , 164", "tři celé sto šedesát čtyři"],
        #  ["1 , 023", "jedna celá nula dvacet tři"],
        #
        # ["0 , 1",  "jednu desetinu"],
        # ["0 , 001", "nula celá nula nula jedna"],
        # ["0 , 0001", "nula celá nula nula nula jedna"],
        # ["0 , 005", "nula celá nula nula pět"],
        # ["0 , 004", "nula celá nula nula čtyři"],
        # ["0 , 0006", "nula celá nula nula nula šest"],
        #
        # ["0 , 055", "nula celá nula padesát pět tisícin"],
        # ["0 , 3",  "žádná celá tři desetiny"],
        # ["1 , 4",  "jedna celá čtyři desetiny"],
        # ["80 , 4",  "osmdesát celých čtyři desetin"],
        #
        # ["0 , 9",  "žádná celá devět"],
        # ["0 , 7",  "žádná celá sedm"],
        # ["0 , 77",  "žádná celá sedmdesát sedm"],
        # ["0 , 8",  "žádná celá osm"],
        # ["0 , 4",  "žádná celá čtyři"],
        # ["0 , 11",  "žádná celá jedenáct"],
        #
        # ["1 , 5", "jeden a půl"],
        # ["2 , 5", "dva a půl"],
        # ["8 , 5", "osm a půl"],
        # ["3 , 5", "tři a půl"],
        # ["13 , 5", "třináct a půl"],
        # ["11 , 5", "jedenácti a půl"],
        # ["2 , 5", "dvě a půl"],
        #
        # ["70 , 6", "sedmdesát celých šest"],
        #
        #
        # ["2 , 1",  "dvě celé jedna desetina"],
        # ["4 , 3",  "čtyři celé tři desetiny"],
        # ["2 , 7",  "dvě celé sedm desetin"],
        # ["9 , 3",  "devět celé tři desetiny"],
        # ["3 , 4",  "tři celé čtyři desetiny"],
        # ["0 , 9",  "žádná celá devět desetin"],
        # ["3 , 6",  "tři celé šest desetin"],
        #
        # ["0 , 00103", "nula celá nula nula sto tři"],
        #
        # ["1 , 440", "jedna celá čtyřista čtyřicet"],
        # ["1 , 285", "jedna celá dvěstě osmdesát pět"],
        # ["1 , 748", "jedna celá sedmset čtyřicet osm"],
        # ["0 , 012", "nula celá nula dvanáct"],
        # ["0 , 506", "nula celá pětset šest"],
        # ["0 , 521", "nula celá pětset dvacet jedna"],
        # ["1 , 031", "jedna celá nula třicet jedna"],
        # ["4 , 185", "čtyři celé sto osmdesát pět"],
        # ["5 , 265", "pět celý dvěstě šedesát pět"],
    ]

    for k, v in tests:
        whole, decimal = k.split(' , ')
        results = n2w_floats_morphodita(whole, decimal, True)
        f = f'{whole}.{decimal}'
        if v in results:
            print(f'OK, {f:>10}, len={len(results):>5}, <{v}>')
        else:
            print(f'--, {f:>10}, len={len(results):>5}, <{v}>')
        time.sleep(1)

# %%
if __name__ == '__main__':
    # %%
    # n2w_ordinal_morphodita(999)
    # y = n2w_cardinal_morphodita(1001001001321)
    # %%
    # n2w_floats_morphodita('0', '05')
    # result = n2w_times_morphodita('23', '15', unique=True)

    # %%
    # get_request_morphodita('šestset')
    test_float()
    # test_time()

    # %%
    # cases:
    # V 1. [only_num_same_case] no "hours", "minutes", both numbers in the same case (can have "hours" at the end)
    #   V   do 12.50 -> do dvanácti padesáti
    #   V   Na 12.45 hodin -> na dvanáct čtyřicet pět
    #   V   do 9.30 -> do devíti třiceti
    #   V   od 14.30 -> od čtrnácti třiceti
    #   V   ve 12.30 -> ve dvanáct třicet
    #       se dostaneme ve 12.50 -> se dostavil do let ve dvanáct padesát
    #       do dneška do 21.21 -> do jedna dvaceti jedná dvaceti
    #       na 12.50 -> na dvanáct padesát
    #   V   do 21.21 -> do jedna dvaceti jedná dvaceti
    #       od 17.45 -> od sedmnácti čtyřiceti pěti
    #       do 10.15 -> do deseti patnácti
    #       od 12.15 -> od dvanácti patnácti
    #       do 10.15 -> do deseti patnácti
    #       do 10.15 -> do deseti patnácti
    #       ve 13.45 hodin -> ve třináct čtyřicet pět hodin
    #   2. [two_groups] two groups, hours_int with hours word in the same case and minutes_int and minutes_word in the same case
    #   V   od 18.45 -> od osmnácti hodin čtyři pěti minut
    #   V   od 23.15 hodin -> od do dvaceti tří hodin patnácti minut
    #   V   od 11.15 hodin -> od jedenácti hodin patnácti minut
    #       na 12.45 hodin -> na dvanáct hodin čtyřicet pět minut
    #       na 12.50 hodin -> na dvanáct hodin padesát minut
    #       ve 13.15 hodin -> třináct hodin patnáct minut
    #   V   ve 13.05 -> třináct hodin pět minut
    #   X   12.01 hodin -> dvanáct hodin jedna minuta
    #   3. zeros pronounced
    #   V   do 13.00 -> do třináct nula nula
    #   X   pauzu do 14.04 -> pauzu do chtěl nás nula čtyři
    #       do 19.00 -> do devatenácti nula nula
    #       do 13.00 -> do třinácti nula nula
    # ? 4. ordinal numbers for hours and minutes
    #       až na 18.15 -> až na osmnáctou patnáct
    #   5. [minutes_ignored] minutes are not pronounced
    #       ve 12.50 hodin -> ve dvanáct hodin
    #       do 15.00 hodin -> do patnáct hodin
    #       ve 14.00 hodin -> ve čtrnáct hodin
    #       od 11.00 -> od jedenácti
    # ? 6. [quaters] quarters, use preposition "na"
    #       od 14.45 -> od tří čtvrtina tři
    #       od 11.45 -> od tří čtvrtina dvanáct
    # V 8. halfs with ordinal -> (hours_int % 12) + 1
    #       do 17.30 -> do půl šesté
    # ? 9. hours_int, hours_word, minutes_int, minutes_word in the same case, numbers should be ordinal
    #       na 16.30 . -> na šestnáctou hodinu třicátou minutu
    #       na sedmnáctou hodinu třicátou minutu
    #       -> 7|fem|sing

    # %%
    # s = "tisíc sto jeden"
    # x, y = gen_all_comb(s, get_request_morphodita(s))
    #
    # # %%
    # url = 'http://lindat.mff.cuni.cz/services/morphodita/api/generate?'
    # params = dict(
    #     data="\n".join(['dvacet', 'dva']),
    #     convert_tagset='pdt_to_conll2009',
    #     output='json'
    # )
    # response = requests.get(url, params)
    # dicts = response.json()['result']
    #
