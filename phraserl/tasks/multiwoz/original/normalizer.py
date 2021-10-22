"""
Original code:
https://github.com/budzianowski/multiwoz/blob/master/utils/nlp.py
"""

import re

timepat = re.compile(r"\d{1,2}[:]\d{1,2}")
pricepat = re.compile(r"\d{1,3}[.]\d{1,2}")

fin = open("phraserl/tasks/multiwoz/original/mapping.txt")
replacements = []
for line in fin.readlines():
    tok_from, tok_to = line.replace("\n", "").split("\t")
    replacements.append((" " + tok_from + " ", " " + tok_to + " "))


def insertSpace(token, text):
    sidx = 0
    while True:
        sidx = text.find(token, sidx)
        if sidx == -1:
            break
        if (
            sidx + 1 < len(text)
            and re.match("[0-9]", text[sidx - 1])
            and re.match("[0-9]", text[sidx + 1])
        ):
            sidx += 1
            continue
        if text[sidx - 1] != " ":
            text = text[:sidx] + " " + text[sidx:]
            sidx += 1
        if sidx + len(token) < len(text) and text[sidx + len(token)] != " ":
            text = text[: sidx + 1] + " " + text[sidx + 1 :]
        sidx += 1
    return text


def normalize(text):
    # lower case every word
    text = text.lower()

    # replace white spaces in front and end
    text = re.sub(r"^\s*|\s*$", "", text)

    # hotel domain pfb30
    text = re.sub(r"b&b", "bed and breakfast", text)
    text = re.sub(r"b and b", "bed and breakfast", text)

    # normalize phone number
    ms = re.findall(r"\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4,5})", text)
    if ms:
        sidx = 0
        for m in ms:
            sidx = text.find(m[0], sidx)
            if text[sidx - 1] == "(":
                sidx -= 1
            eidx = text.find(m[-1], sidx) + len(m[-1])
            text = text.replace(text[sidx:eidx], "".join(m))

    # normalize postcode
    ms = re.findall(
        (
            r"([a-z]{1}[\. ]?[a-z]{1}[\. ]?\d{1,2}[, ]+\d{1}[\. ]?[a-z]{1}"
            r"[\. ]?[a-z]{1}|[a-z]{2}\d{2}[a-z]{2})"
        ),
        text,
    )
    if ms:
        sidx = 0
        for m in ms:
            sidx = text.find(m, sidx)
            eidx = sidx + len(m)
            text = text[:sidx] + re.sub(r"[,\. ]", "", m) + text[eidx:]

    # weird unicode bug
    text = re.sub(u"(\u2018|\u2019)", "'", text)

    # replace time and and price
    text = re.sub(timepat, " [value_time] ", text)
    text = re.sub(pricepat, " [value_price] ", text)
    # text = re.sub(pricepat2, '[value_price]', text)

    # replace st.
    text = text.replace(";", ",")
    text = re.sub(r"$\/", "", text)
    text = text.replace("/", " and ")

    # replace other special characters
    text = text.replace("-", " ")
    text = re.sub(r'[":\<>@\(\)]', "", text)

    # insert white space before and after tokens:
    for token in ["?", ".", ",", "!"]:
        text = insertSpace(token, text)

    # insert white space for 's
    text = insertSpace("'s", text)

    # replace it's, does't, you'd ... etc
    text = re.sub(r"^'", "", text)
    text = re.sub(r"'$", "", text)
    text = re.sub(r"'\s", " ", text)
    text = re.sub(r"\s'", " ", text)
    for fromx, tox in replacements:
        text = " " + text + " "
        text = text.replace(fromx, tox)[1:-1]

    # remove multiple spaces
    text = re.sub(" +", " ", text)

    # concatenate numbers
    tokens = text.split()
    i = 1
    while i < len(tokens):
        if re.match(r"^\d+$", tokens[i]) and re.match(r"\d+$", tokens[i - 1]):
            tokens[i - 1] += tokens[i]
            del tokens[i]
        else:
            i += 1
    text = " ".join(tokens)

    return text
