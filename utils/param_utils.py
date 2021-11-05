import ast
import collections.abc

parse_dict = {'True': True, 'False': False, 'None': None}


def update_nested_dict(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_nested_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def fix_parsing_values_to_int(d):
    assert isinstance(d, dict), 'input it not a dictionary type'
    for k, v in d.items():
        if isinstance(v, dict):
            fix_parsing_values_to_int(v)
        else:
            if is_int(v):
                d.update({k: int(v)})
            elif is_float(v):
                d.update({k: float(v)})
            elif is_list(v):
                d.update({k: ast.literal_eval(v)})
            else:
                # is string
                v = parse_dict[v] if v in parse_dict.keys() else v
                d.update({k: v})
    return d


def is_list(v):
    try:
        isinstance(ast.literal_eval(v), list)
    except (ValueError, SyntaxError):
        return False
    else:
        return isinstance(ast.literal_eval(v), list)


def is_int(n):
    try:
        float_n = float(n)
        int_n = int(float_n)
    except (ValueError, SyntaxError):
        return False
    else:
        return float_n == int_n


def is_float(n):
    try:
        float_n = float(n)
    except (ValueError, SyntaxError):
        return False
    else:
        return True
