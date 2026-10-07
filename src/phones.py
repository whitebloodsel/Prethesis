"""Shared phone inventory (ARPAbet without stress digits). Id 0 is reserved for padding.
Both corpora are mapped into this list so M2, M3 and M4 use the same output space."""
ARPABET = ("AA AE AH AO AW AY B CH D DH EH ER EY F G HH IH IY JH K L M N NG OW OY "
           "P R S SH T TH UH UW V W Y Z ZH").split()
PHONE_LIST = ["<pad>"] + ARPABET
PHONE_ID = {p: i for i, p in enumerate(PHONE_LIST)}

# Indonesian corpus (GiVe) writes phones in IPA. DECISION TO CONFIRM WITH YOUR FRIEND:
# /ə/ and /ʌ/ are both mapped to AH, because Speechocean762 merges them once stress digits are removed.
IPA_TO_ARPABET = {
    "aɪ": "AY", "aʊ": "AW", "b": "B", "d": "D", "dʒ": "JH", "eɪ": "EY", "f": "F", "g": "G",
    "h": "HH", "i": "IY", "j": "Y", "k": "K", "l": "L", "m": "M", "n": "N", "oʊ": "OW",
    "p": "P", "r": "R", "s": "S", "t": "T", "tʃ": "CH", "u": "UW", "v": "V", "w": "W",
    "z": "Z", "æ": "AE", "ð": "DH", "ŋ": "NG", "ɑ": "AA", "ɔ": "AO", "ɔɪ": "OY", "ə": "AH",
    "ɛ": "EH", "ɜ": "ER", "ɪ": "IH", "ʃ": "SH", "ʊ": "UH", "ʌ": "AH", "θ": "TH",
}
