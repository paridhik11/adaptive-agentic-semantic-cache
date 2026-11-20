"""Ordered keyword signatures for standalone live-chat requests."""

from functools import lru_cache
import re

from src.api.chat_intents import direct_question, question_intent

# Only strip framing with a recognized request form. Content and qualifiers
# remain in the key, even when they look like common English stopwords.
_POLITE_ACTION = re.compile(r"^(?:can|could|would) you (?:please )?(?=explain |show |give |define )", re.I)
_DEPENDENT = re.compile(
    r"\b(?:this|that|these|those|it|its|they|them|their|you|your|i|my|me|our|us|above|below)\b",
    re.I,
)
_METHOD = re.compile(
    r"(?:(?:explain |show (?:me )?)?how to |how (?:do|can) (?:i|we|you|one) |"
    r"what(?: is|'s) the way to )(.+)", re.I,
)
# Admit standalone questions about reproducible operations. Other methods can
# still be indexed when the existing stability classifier confirms reuse.
_METHOD_ACTIONS = frozenset(
    "aggregate average calculate cast combine convert count divide extract filter format "
    "group join measure merge multiply normalize parse rename replace round select sort split subtract".split()
)
_FORMS = (
    ("mechanism", (
        r"how (?:does|do) (.+) (?:work|operate|function)",
        r"(?:explain )?how (.+) (?:works?|operates?|functions?)",
    )),
    ("purpose", (
        r"what (?:is|are) (.+) used for",
        r"what (?:is|are) the use of (.+)",
        r"what do (?:people|you) use (.+) for",
        r"(?:what(?: is|'s) the |explain the )?purpose of (.+)",
        r"what(?: is|'s) (.+)'s purpose",
    )),
    ("formula", (
        r"(?:what(?: is|'s) the |give (?:me )?the )formula for (.+)",
        r"how is the formula for (.+) written",
    )),
    ("definition", (
        r"what does (.+) mean",
        r"(?:how would you define |what do (?:you|u) mean by )(.+)",
        r"(?:what(?: is|'s) the (?:definition|meaning) of |meaning of |define |"
        r"give (?:me )?the definition of |what(?: is|'s) |what are )(.+)",
        r"(.+) (?:meaning|definition)",
    )),
)

# Only explicit attributes with a predictable possessive reading. "Of" in an
# arbitrary name or technical phrase is not an ownership marker.
_ATTRIBUTE = r"capital(?: city)?|atomic number|chemical formula|date of birth|place of birth|birthplace|original author|publication date"
_ATTRIBUTES = (
    re.compile(rf"(?:what(?: is|'s) the )?({_ATTRIBUTE}) of (.+)", re.I),
    re.compile(rf"(?:what(?: is|'s) )?(.+)'s ({_ATTRIBUTE})", re.I),
)
_ATTRIBUTE_CANONICAL = {"capital city": "capital", "place of birth": "birthplace"}

# For a regular past verb, active and passive questions can express the
# same agent relationship. Keep the verb and every word of its object.
_AGENT = (
    re.compile(r"who ([a-z]{1,23}ed) (.+)", re.I),
    re.compile(r"who (?:was|were) (.+) ([a-z]{1,23}ed) by", re.I),
)

_WHY = re.compile(r"(?:explain )?why (.+)", re.I)
_COPULA = re.compile(r"\b(?:is|are|was|were)\b", re.I)
_COMPARATIVE = re.compile(r"\b(?:more|less|better|worse|lower|higher|different|similar)\b", re.I)

_REVERSE_CONTAINER = re.compile(
    r"(?:make|build) (?:a |an )?(tuple|list) (?:out of|from) (?:a |an )?(tuple|list)", re.I
)
_CONVERSION = (
    re.compile(r"from (.+?) to (.+)", re.I),
    re.compile(r"to (.+?) from (.+)", re.I),
    re.compile(r"(.+?) (?:to|into) (.+)", re.I),
)
_PART = re.compile(r"[\w][\w +#./%'°-]{0,220}")
_SEPARATORS = re.compile(r"\b(?:from|to|into)\b", re.I)


def _ordered_words(phrase):
    if not _PART.fullmatch(phrase) or _DEPENDENT.search(phrase):
        return ()
    # Keep acronyms and mixed-case identifiers distinct. Leave repeated
    # keywords in place and retain punctuation inside symbols and versions.
    words = tuple(word if word.isupper() or any(ch.isupper() for ch in word[1:])
                  else word.casefold() for word in phrase.split())
    if phrase.split()[0] in ("a", "an", "the") and (len(words) == 1 or words[1][:1].islower()):
        words = words[1:]
    return words


def _method_keywords(text):
    match = _REVERSE_CONTAINER.fullmatch(text)
    if match and match[1].casefold() != match[2].casefold():
        return "method", ("convert", (match[2].casefold(),), (match[1].casefold(),))
    action, _, target = text.partition(" ")
    action = action.casefold()
    if action == "compute":
        action = "calculate"
    if action in {"convert", "cast"}:
        for index, pattern in enumerate(_CONVERSION):
            if match := pattern.fullmatch(target):
                source, destination = reversed(match.groups()) if index == 1 else match.groups()
                # Embedded direction can describe a different operation. Keep
                # such a request in its original order.
                if not (_SEPARATORS.search(source) or _SEPARATORS.search(destination)):
                    start, end = _ordered_words(source), _ordered_words(destination)
                    if start and end:
                        return "method", (action, start, end)
                break
    words = _ordered_words(target)
    return ("method", (action, *words)) if words else None


@lru_cache(maxsize=512)
def keyword_signature(question):
    """Keep request kind, argument order, values and qualifications."""
    if len(question) > 260 or "\n" in question or "\r" in question:
        return None
    text = " ".join(direct_question(question).replace("’", "'").split())
    text = _POLITE_ACTION.sub("", text).rstrip("?.!").strip()
    if re.search(r"[.!?]\s", text):
        return None
    # A ranked officeholder request needs a named jurisdiction or organization.
    if (intent := question_intent(text)) and intent.rank and intent.subject:
        return "ranked", (intent.rank, intent.relation, *intent.subject.split())
    for index, pattern in enumerate(_ATTRIBUTES):
        if match := pattern.fullmatch(text):
            attribute, subject = reversed(match.groups()) if index == 1 else match.groups()
            if words := _ordered_words(subject):
                attribute = attribute.casefold()
                return "attribute", (_ATTRIBUTE_CANONICAL.get(attribute, attribute), *words)
    for index, pattern in enumerate(_AGENT):
        if match := pattern.fullmatch(text):
            verb, subject = reversed(match.groups()) if index == 1 else match.groups()
            if words := _ordered_words(subject):
                return "agent", (verb.casefold(), *words)
    if match := _METHOD.fullmatch(text):
        return _method_keywords(match[1])
    if match := _WHY.fullmatch(text):
        # In an English why-question the copula can precede the subject.
        # Retain the tense and the order of all words around it.
        clause = match[1]
        copulas = list(_COPULA.finditer(clause))
        if len(copulas) == 1 and not re.search(r"\b(?:and|or|that|if)\b", clause, re.I):
            copula = copulas[0]
            before, after = clause[:copula.start()].strip(), clause[copula.end():].strip()
            # Infer the boundary only where a comparative marker identifies
            # the predicate in both inverted and declarative forms.
            if marker := _COMPARATIVE.search(after):
                if not before or marker.start() == 0:
                    subject = _ordered_words(before or after[:marker.start()].strip())
                    predicate = _ordered_words(after[marker.start():])
                    if subject and predicate:
                        return "why", (copula[0].casefold(), subject, predicate)
        if words := _ordered_words(clause):
            return "why", words
    for kind, patterns in _FORMS:
        for pattern in patterns:
            if match := re.fullmatch(pattern, text, re.I):
                words = _ordered_words(match[1])
                return (kind, words) if words else None
    return None


def educational_keywords(signature):
    if signature is None:
        return False
    kind, words = signature
    return kind != "method" or words[0] in _METHOD_ACTIONS
