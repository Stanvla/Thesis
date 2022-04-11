# %%
import requests
from num2words import num2words
import pandas as pd
from icecream import ic
import copy

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
    try:
        result = parse_response(response)
    except:
        ic(word)
        ic(response)
        raise RuntimeError('can not parse response')
    return result

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
    # num can be string or int
    if isinstance(num, str):
        num = int(num)

    result_dict = {k: [] for k in ord_base}
    result_lst = ['']

    types = ['units', 'tens', 'hundreds']

    for order_name, order_lst in orders_groups.items():
        if ord_base[order_lst[0]] > num:
            break

        group = {k: v for k, v in zip(types, order_lst)}
        result_dict, highest = expand_group(group, result_dict, ord_num, german_version)
        if highest is not None:
            group_with_name_lst = []
            for x in result_dict[highest]:
                if 'units' in highest and orders_groups_names[order_name] != '':
                    group_with_name_lst.append(orders_groups_names[order_name])
                group_with_name_lst.append(x + ' ' + orders_groups_names[order_name])
            result_dict[highest] = group_with_name_lst
            new_result_lst = []
            for x in group_with_name_lst:
                for r in result_lst:
                    new_result_lst.append(x + ' ' + r)
            result_lst = new_result_lst
    if alt_hundreds:
        result_lst.extend(alt_hundreds_1100_2000(num, orders_groups, ord_base, ord_num, german_version))
    result_lst = [r.replace('jedna ', 'jeden ').replace('jednaa', 'jedena') for r in result_lst]
    return [r.rstrip() for r in result_lst]


def n2w_cardinal_morphodita(num, german_version=True):
    def get_all_variants_morphodita(num_lst):
        result = []
        for verb_num in num_lst:
            verb_num = verb_num.rstrip()
            df = pd.concat([get_request_morphodita(n) for n in verb_num.split(' ')])
            result.extend(gen_all_comb(verb_num, df))
        return result

    # Ve větné souvislosti se víceslovné číslovkové výrazy zpravidla skloňují. Máme více možností:
    #   case_1) Skloňujeme všechny části výrazu.
    #    - "Teploty vystoupí k 27 °C" čteme a píšeme k dvaceti sedmi stupňům nebo i obráceně sedmadvaceti stupňům (v takovém případě píšeme číslovku dohromady);
    #    - "před 365 lety"  před třemi sty šedesáti pěti (před třemi sty pětašedesáti) lety;
    #    - "bez 1 847 Kč" bez tisíce osmi set čtyřiceti sedmi korun.
    #   case_2) Skloňujeme jen jméno počítaného předmětu a část číslovkového výrazu (obvykle řád desítek a jednotek), zbytek ponecháváme nesklonný.
    #    - "před 365 lety" můžeme také přečíst před tři sta šedesáti pěti (pětašedesáti) lety
    #    - "bez 1 847 Kč" bez tisíc osm set čtyřiceti sedmi (sedmačtyřiceti) korun;
    #    - "o 1 358 423 dokladech" o milion tři sta padesát osm tisíc čtyři sta  dvaceti třech (třiadvaceti) dokladech.
    #   case_3) V některých situacích, většinou v mluvené řeči, např. při diktování a při početních operacích s čísly,
    #   může zůstat celý víceslovný číslovkový výraz neskloňovaný,
    #    - např. k tisíc sedm set dvacet dva korunám.

    # case_1 will be solved by generating all variants using morphodita
    # case_2 for tens and units use morphodita, the rest will be generated from n2w (exceptions is if 1100 <= num < 2000)
    # case_3 will be included in case_2 by default

    if isinstance(num, str):
        num = int(num)
    orders_groups, orders_groups_names, ord_base, ord_num = get_orders(num)
    case_1 = natural_num_verbalization(num, orders_groups, orders_groups_names, ord_base, ord_num, german_version)
    all_results = get_all_variants_morphodita(case_1)

    tmp_results = []
    # if number is smaller than 100, then case_2 and case_3 are done by default in case_1
    if num > 100:
        # need to handle tisic/jedna tisic, milion/jeden milion,
        new_num = num % 100
        num_reminder = num - new_num
        rest = n2w(num_reminder)

        if new_num != 0:
            orders_groups, orders_groups_names, ord_base, ord_num = get_orders(new_num)
            case_2 = natural_num_verbalization(new_num, orders_groups, orders_groups_names, ord_base, ord_num, german_version, alt_hundreds=False)
            case_2 = get_all_variants_morphodita(case_2)
            tmp_results = [rest + ' ' + x for x in case_2]
        else:
            tmp_results.append(rest)
            case_2 = []

        if num // 10**3 == 1:
            tmp_results.extend([rest.replace('tisíc', 'jedna tisíc') + ' ' + x for x in case_2])
        if num // 10**6 == 1:
            tmp_results.extend([rest.replace('milion', 'jeden milion') + ' ' + x for x in case_2])
        if num // 10**9 == 1:
            tmp_results.extend([rest.replace('miliarda', 'jedna miliarda') + ' ' + x for x in case_2])
        if num // 10**12 == 1:
            tmp_results.extend([rest.replace('bilion', 'jeden bilion') + ' ' + x for x in case_2])

        # case for 1100 <= num < 2000
        orders_groups, orders_groups_names, ord_base, ord_num = get_orders(num_reminder)
        alt_hundreds = alt_hundreds_1100_2000(num_reminder, orders_groups, ord_base, ord_num, german_version=False)
        for alt_h in alt_hundreds:
            if case_2:
                for x in case_2:
                    tmp_results.append(alt_h.replace('sto', 'set') + ' ' + x)
            else:
                tmp_results.append(alt_h.replace('sto', 'set'))

    all_results.extend(tmp_results)
    # ic(tmp_results)
    ic(all_results)
    ic(len(all_results))
    ic(len(set(all_results)))
    return all_results


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
    ic(base_variants)
    if base_variant_only:
        return base_variants[num]
    return None


def n2w_floats_morphodita(num):
    pass

# %%
if __name__ == '__main__':
    # %%
    # nans 1103 and error for 1100
    y = n2w_cardinal_morphodita(1106)
    # n2w_ordinal_morphodita(999)

