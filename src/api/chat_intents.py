"""Local interpretations for standalone questions with interchangeable answers.

Only grammatical wrappers are discarded. The requested relation, object and
rank are compared independently. Object names never receive spelling edits.
"""
from dataclasses import dataclass
from functools import lru_cache
import re

_ORDINALS = "first second third fourth fifth sixth seventh eighth ninth tenth".split()
_RANK = r"(?:" + "|".join(_ORDINALS) + r"|[1-9]\d?(?:st|nd|rd|th))"
_ROLES = {
    "pm": "prime minister", "ceo": "chief executive officer", "cfo": "chief financial officer",
    **{role: role for role in ("prime minister", "president", "vice president", "governor",
        "chief minister", "mayor", "chancellor", "secretary general", "chairperson",
        "chief executive officer", "chief financial officer", "captain", "monarch")},
}
_QUALIFIED_ROLE = re.compile(r"(?:(female|male|elected|acting|interim|appointed) )?(.+)")
# Context-dependent objects cannot identify a reusable standalone answer. Leave
# unsupported qualifiers and compound requests intact for the normal chat path.
_RELATIVE = re.compile(
    r"\b(?:i|me|my|our|you|your|he|him|his|she|her|they|them|it|its|their|there|"
    r"this|that|these|those|current|currently|today|latest|yesterday|tomorrow|recent|"
    r"now|next|last|newest|previous|following|above|below|and|or|with|without|"
    r"please|also|explain|include|give|show|example|examples|why|how|who|what|"
    r"where|when|which|before|after|using|under|not)\b"
)
_NAME = r"[^\W_]+(?:[-' ][^\W_]+)*"

_POLITE = re.compile(
    r"^(?:(?:(?:can|could|would) you (?:please )?tell me |(?:please )?tell me |"
    r"i(?:'d| would) like to know )(?=(?:who|what|where|when|why|how|which)\b)|"
    r"please (?=(?:define|compare|explain|name)\b))", re.I)


def direct_question(text):
    return _POLITE.sub("", text.strip(), count=1)


def wording_key(question):
    """Keep the entire proposition, including case, quantities and modifiers."""
    direct = direct_question(question)
    match = re.fullmatch(r"(who|what|where|when|why|how|which|define|compare|explain) "
                         r"([\w '-]{1,240})[?.!]?", direct, re.I)
    return (match[1].lower(), match[2].strip()) if match else None


def _object(text):
    if (len(text) > 100 or len(text.split()) > 10 or not re.fullmatch(_NAME, text)
            or _RELATIVE.search(text.casefold()) or text.isdigit()):
        return None
    return text


def _ordinal(text):
    if text in _ORDINALS:
        return _ORDINALS.index(text) + 1
    n = int(text[:-2])
    return n if text.endswith(_ordinal_suffix(n)) else None


def _ordinal_suffix(n):
    return "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def _role(text):
    qualifier, role = _QUALIFIED_ROLE.fullmatch(text).groups()
    if role not in _ROLES:
        # Repair an accidentally repeated character only if it recovers a
        # recognized role unambiguously. Never approximate a country or title.
        repairs = {role[:m.start()] + role[m.start()+1:] for m in re.finditer(r"(.)\1", role)}
        candidates = repairs.intersection(_ROLES)
        if len(candidates) != 1:
            return None
        role = candidates.pop()
    return " ".join(filter(None, (qualifier, _ROLES[role])))


@dataclass(frozen=True)
class QuestionIntent:
    relation: str
    subject: str
    rank: int = 0
    precision: str = ""

    def accepts(self, source, answer):
        if self == source:
            return True
        if (not source or not self.precision or not source.precision
                or (self.relation, self.subject, self.rank) != (source.relation, source.subject, source.rank)):
            return False
        if self.precision == "when":
            return True
        # A short, unambiguous date can answer a differently phrased time
        # question. A year alone cannot supply a day, and two years may be a range.
        years = re.findall(r"\b(?:1[0-9]{3}|20[0-9]{2})\b", answer)
        if len(answer.split()) > 45 or len(years) != 1:
            return False
        if self.precision == "year":
            return True
        month = r"(?:Jan(?:uary)?|Feb(?:ruary)?|March|April|May|June|July|Aug(?:ust)?|Sept?(?:ember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
        return bool(re.search(rf"\b(?:{month} \d{{1,2}}|\d{{1,2}} {month}|\d{{4}}-\d{{2}}-\d{{2}})\b", answer, re.I))

    def scoped_question(self, subject):
        """A provider-ready question for an otherwise incomplete ranked role."""
        n = self.rank
        suffix = _ordinal_suffix(n)
        return f"Who was the {n}{suffix} {self.relation} of {subject}?"


_PATTERNS = {
    "birthplace": (r"where was (.+) born", r"(.+)'s (?:place of birth|birthplace)",
                  r"(?:place of birth|birthplace) of (.+)"),
    "birth date": (r"when was (.+) born", r"(.+)'s (?:date of birth|birth date)",
                   r"(?:date of birth|birth date) of (.+)"),
    "death place": (r"where did (.+) (?:die|pass away)", r"(.+)'s place of death",
                    r"place of death of (.+)"),
    "death date": (r"when did (.+) (?:die|pass away)", r"(.+)'s (?:date of death|death date)",
                   r"date of death of (.+)", r"when (.+) died"),
    "author": (r"who wrote (.+)", r"(?:who (?:is|was) )?the author of (.+)",
               r"who was (.+) written by"),
    "calculation method": (r"how (?:do|can) i (?:calculate|compute) (.+)",
                           r"how (?:is|are) (.+) (?:calculated|computed)"),
}


@lru_cache(maxsize=512)
def question_intent(question):
    if len(question) > 220:
        return None
    text = " ".join(direct_question(question).replace("’", "'").split()).rstrip("?.!").strip()
    time_prefix = re.match(r"^(?:(?:in |on )?(?:what|which) (year|date)|when) (?=(?:did|was|were|had) )", text, re.I)
    precision = (time_prefix[1] or "when").lower() if time_prefix else ""
    timed = "when " + text[time_prefix.end():] if time_prefix else text
    for relation, patterns in _PATTERNS.items():
        phrasing = timed
        if relation.startswith(("birth", "death")):
            phrasing = re.sub(r"^what (?:is|was) (?:the )?", "", timed, flags=re.I)
        for pattern in patterns:
            match = re.fullmatch(pattern, phrasing, re.I)
            if match and (subject := _object(match[1])):
                # Preserve the casing of symbols such as VaR and VAR in methods.
                if relation != "calculation method":
                    subject = subject.casefold()
                grain = (precision or "date") if relation in {"birth date", "death date"} else ""
                return QuestionIntent(relation, subject, precision=grain)

    if time_prefix:
        auxiliary, clause = text[time_prefix.end():].split(" ", 1)
        if _object(clause):
            return QuestionIntent("event time " + auxiliary.lower(), clause, precision=precision)
    text = text.casefold()
    text = re.sub(r"^(?:who (?:was|is) |name )?(?:the )?", "", text)
    possessive = re.fullmatch(rf"(.+)'s ({_RANK}) (.+)", text)
    if possessive:
        entity, ordinal, role = possessive.groups()
    else:
        ranked = re.fullmatch(rf"({_RANK}) (.+)", text)
        if not ranked:
            return event_intent(question)
        ordinal, tail = ranked.groups()
        role, _, entity = tail.partition(" of ")
    rank = _ordinal(ordinal)
    relation = _role(role)
    if rank and relation and (not entity or _object(entity)):
        return QuestionIntent(relation, entity, rank)
    return None


# In past-event questions, keep the requested answer unit and every word of
# the predicate. Preposition movement is reversible ("in which city" and
# "which city ... in"). Directional prepositions are part of the identity.
_UNIT = r"(?:year|month|century|decade|day|city|country|state|province|town|team|university|college|company|organization|language)"
_TIME_UNITS = {"year", "month", "century", "decade"}
_PAST_AUX = r"(?:was|were|did|had)"


def event_intent(question):
    direct = direct_question(question).strip().rstrip("?.!").strip()
    match = re.fullmatch(
        rf"(?:(in|at|from|to|for) )?(?:what|which) ({_UNIT}) ({_PAST_AUX} [\w '-]+)",
        direct, re.I)
    if not match or len(direct) > 200:
        return None
    front, unit, predicate = match.groups()
    unit = unit.lower()
    # Ignore grammatical capitalization at the start of the auxiliary only.
    aux, clause = predicate.split(" ", 1)
    if (_RELATIVE.search(clause.casefold()) or len(clause.split()) > 20
            or not clause.strip()):
        return None
    tail = re.search(r" (in|at|from|to|for)$", clause, re.I)
    if front and tail:
        return None
    if tail:
        clause = clause[:tail.start()]
    prep = (front or (tail[1] if tail else "")).lower()
    if unit in _TIME_UNITS and prep == "in":
        prep = ""
    relation = " ".join(filter(None, (prep, unit, aux.lower())))
    return QuestionIntent(relation, clause.strip())


def scope_reply(question):
    """A name-only reply to the immediately preceding clarification."""
    name = question.strip().rstrip(".").strip()
    if len(name.split()) > 6 or re.match(
        r"^(?:hello|hi|thanks|thank|ok|okay|define|compare|calculate|list|name|"
        r"describe|tell|write|can|could|would|should|is|are|do|does)\b", name, re.I
    ):
        return None
    return _object(name)


# Negative or prospective answers can change when a position gets filled.
_UNRESOLVED = re.compile(
    r"\b(?:unknown|unclear|unborn|not yet|has(?:n't| not)|never (?:had|been)|"
    r"no (?:such|one|record|known)|there (?:is|was) no|yet to|will be|"
    r"cannot|can't|couldn't|don't know|not sure|does not exist|could refer to|can you clarify|which (?:country|person|organization))\b", re.I)



def reusable_fact_answer(content):
    return not _UNRESOLVED.search(content.replace("’", "'")) and not content.rstrip().endswith("?")
