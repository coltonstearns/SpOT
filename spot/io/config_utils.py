'''
argparse options.
'''
import argparse
import yaml


def get_general_options():
    '''
    Adds general options to the given argparse parser.
    These are options that are shares across train, test, and visualization time.
    '''
    parser = argparse.ArgumentParser(allow_abbrev=False)

    # Regular Config Files
    parser.add_argument('--config', type=str, required=True)
    args, unknown_args = parser.parse_known_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # overwrite config arguments with any extra unknown arguments
    unknown_args = hacky_format_unknown_args(unknown_args)
    unified_config = overwrite_configfile_fields(config, unknown_args)

    return unified_config


def overwrite_configfile_fields(unified_config, extra_args):
    for arg, val in extra_args.items():
        arg_keys = arg.split(".")
        _, success = recursive_dict_update(unified_config, arg_keys, val)
        if not success:
            print("!!!!!!!!!!!!!!!")
            print("Command Line Argument %s Did Not Match a Config Key!" % arg)

    return unified_config


def hacky_format_unknown_args(unknown):
    def format_value(val):
        # list check
        if isinstance(val, str) and len(val.split(" ")) > 1:
            vals = val.split(" ")
            tmp = []
            for v in vals:
                tmp.append(format_value(v))
            val = tmp

        # boolean check
        if val == "True":
            val = True
        elif val == "False":
            val = False

        # int / float check
        try:
            val = float(val)
        except ValueError:  # it's a string
            pass


        return val

    formatted = {}
    for arg in unknown:
        if arg.startswith(("--")):
            name, val = arg.split('=')
            name = name[2:]
            val = format_value(val)
            formatted[name] = val
    return formatted


def recursive_dict_update(d, u, val):
    if len(u) == 0:
        return
    k = u[0]
    if len(u) > 1:
        if k in d:
            d[k], flag = recursive_dict_update(d[k], u[1:], val)
    else:
        if k in d:
            try:
                val_type = type(d[k])
                d[k] = val_type(val)
                flag = True
            except TypeError as e:
                print("Changing config input type for key %s, from %s to %s." % (k, str(val_type), str(type(val))))
                d[k] = val
                flag = True
        else:
            flag = False

    return d, flag