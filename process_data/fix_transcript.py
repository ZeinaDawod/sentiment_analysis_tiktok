

import re

CORRECTIONS = {

    "cerevive": "CeraVe",
    "sarah vee": "CeraVe",
    "sarah v": "CeraVe",
    "cera vay": "CeraVe",
    "serave": "CeraVe",
    "cera v": "CeraVe",
    "sera v": "CeraVe",
    "CeraVeal":"CeraVe",


    "hyaluronic assets": "hyaluronic acid",
    "hyaluronic assid": "hyaluronic acid",
    "hyalouronic acid": "hyaluronic acid",
    "hyaluronic assyria": "hyaluronic acid",


    "seramides": "ceramides",
    "sarah midas": "ceramides",
    "ceramics": "ceramides",


    "nyah cinnamide": "niacinamide",
    "nyacinamide": "niacinamide",
    "nia cinnamide": "niacinamide",
    "nice in a mide": "niacinamide",


    "sal a silic acid": "salicylic acid",
    "salicylic assid": "salicylic acid",


    "read a nol": "retinol",
    "read in all": "retinol",


    "a h a": "AHA",
    "b h a": "BHA",
    "s p f": "SPF",
}


def fix_transcript(text: str) -> str:

    if not isinstance(text, str):
        return text

    corrected = text
    for wrong, right in CORRECTIONS.items():
        pattern = re.compile(re.escape(wrong), re.IGNORECASE)
        corrected = pattern.sub(right, corrected)

    return corrected