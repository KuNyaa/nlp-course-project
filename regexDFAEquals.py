import subprocess

def unprocess_regex(regex):
    regex = regex.replace("<VOW>", " ".join('AEIOUaeiou'))
    regex = regex.replace("<NUM>", " ".join('0-9'))
    regex = regex.replace("<LET>", " ".join('A-Za-z'))
    regex = regex.replace("<CAP>", " ".join('A-Z'))
    regex = regex.replace("<LOW>", " ".join('a-z'))

    regex = regex.replace("<M0>", " ".join('dog'))
    regex = regex.replace("<M1>", " ".join('truck'))
    regex = regex.replace("<M2>", " ".join('ring'))
    regex = regex.replace("<M3>", " ".join('lake'))

    regex = regex.replace(" ", "")

    return regex

def regex_equiv(gold, predicted):
    if gold == predicted:
        return True
    out = subprocess.check_output(['java', '-jar', 'regex_dfa_equals.jar', '{}'.format(gold), '{}'.format(predicted)])
    if '\\n1' in str(out):
        return True
    else:
        return False

def regex_equiv_from_raw(gold, predicted):
    gold = unprocess_regex(gold)
    predicted = unprocess_regex(predicted)
    return regex_equiv(gold, predicted)

