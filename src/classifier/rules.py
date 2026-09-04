"""Rule-based first pass classifier for query stability.

Detects clearly dynamic/volatile queries as well as clearly invariant/historical
queries using explicit patterns, temporal signals, and domain heuristics.
Includes mixed-intent dynamic clause guarding.
"""

from __future__ import annotations

from datetime import datetime
import re
from typing import List, Pattern, Tuple

from src.classifier.models import RuleMatch, StabilityLabel


class StabilityRuleEngine:
    """Explainable rule engine for first-pass query stability classification."""

    def __init__(self) -> None:
        self._compile_rules()

    def _compile_rules(self) -> None:
        # Dynamic past year regex: matches 4-digit years (1000-2999) preceded by prepositions
        self.year_anchor_pattern = re.compile(
            r"\b(in|during|before|after|until|of\s+the|of|census\s+of|treaty\s+of|the|the\s+year)\s+([12][0-9]{3})\b",
            re.IGNORECASE,
        )

        # Clause boundary splitter for mixed-intent queries
        self.clause_splitter = re.compile(
            r"[?;\n]+|\b(?:and\s+also|and\s+what\s+is|and\s+how\s+much|and\s+who\s+is|along\s+with|as\s+well\s+as)\b",
            re.IGNORECASE,
        )

        # 1. Historical & Invariant Overrides (Guards against naive temporal false positives)
        self.historical_named_patterns: List[Tuple[Pattern[str], str, str]] = [
            (
                re.compile(
                    r"\b(world\s+war\s+(i|ii|1|2)|great\s+depression|declaration\s+of\s+independence|spanish\s+armada|thirty\s+years'?\s+war|byzantine\s+empire|apollo\s+11|chernobyl|battle\s+of\s+waterloo|ancient\s+civilization|machu\s+picchu)\b",
                    re.IGNORECASE,
                ),
                "historical_named_event",
                "Query references a completed historical event or landmark ({anchor}).",
            ),
            (
                re.compile(
                    r"\b(in\s+what\s+year|which\s+year|what\s+year|what\s+was\s+the\s+year)\b",
                    re.IGNORECASE,
                ),
                "historical_year_inquiry",
                "Query inquires about a historical calendar year for a completed past occurrence.",
            ),
            (
                re.compile(
                    r"\bwho\s+won\s+the\b.*\b(world\s+cup|olympics|super\s+bowl|championship|tournament|election)\b",
                    re.IGNORECASE,
                ),
                "historical_sports_tournament_completed",
                "Query asks about a past completed sports or tournament outcome.",
            ),
            (
                re.compile(
                    r"\b(who\s+was\s+the|who\s+were\s+the|what\s+was\s+the\s+capital\s+of|what\s+ancient\s+civilization|what\s+treaty\s+ended)\b",
                    re.IGNORECASE,
                ),
                "historical_past_fact",
                "Past-tense inquiry about a completed historical period, treaty, ancient civilization, or past entity.",
            ),
        ]

        # Invariant scientific / mathematical / CS / geography conceptual patterns
        self.invariant_concept_patterns: List[Tuple[Pattern[str], str, str]] = [
            # Mathematics
            (
                re.compile(
                    r"\b(time\s+complexity|space\s+complexity|big-o|asymptotic|euclidean\s+algorithm|bayes'?\s+theorem|central\s+limit\s+theorem|eigenvalues?|eigenvectors?|euler'?s\s+(identity|number)|fundamental\s+theorem|quadratic\s+equation|prime\s+numbers?|peano\s+arithmetic|standard\s+deviation|derivative\s+of|taylor\s+series|sum\s+of\s+angles|type\s+1\s+and\s+type\s+2\s+error|irrational|digits\s+of\s+pi|fluid\s+ounces|compound\s+interest|pythagorean\s+theorem|matrix\s+multiplication)\b",
                    re.IGNORECASE,
                ),
                "invariant_math_algorithm",
                "Query addresses a proven mathematical theorem, invariance, or algorithmic complexity property.",
            ),
            # Physical science & medicine
            (
                re.compile(
                    r"\b(chemical\s+formula|speed\s+of\s+light|atomic\s+(weight|number)|periodic\s+table|newton'?s\s+.*laws?|mechanism\s+of\s+action|mitosis|meiosis|blood\s+groups?|hemoglobin|appendicitis|adaptive\s+immunity|cell\s+membrane|beta\s+blockers|ace\s+inhibitors|resting\s+heart\s+rate|diabetes\s+mellitus|streptococcal\s+pharyngitis|acetaminophen|metformin|carbon-14|half-life|chromosomes|doppler\s+effect|conservation\s+of\s+energy|photosynthesis|dna\s+replication)\b",
                    re.IGNORECASE,
                ),
                "invariant_science_medicine",
                "Query addresses fundamental physical, chemical, biological, or established medical invariants.",
            ),
            # Geography
            (
                re.compile(
                    r"\b(capital\s+(city\s+)?of|deepest\s+ocean|longest\s+river|largest\s+land\s+area|mountain\s+range\s+separates)\b",
                    re.IGNORECASE,
                ),
                "invariant_geography",
                "Query inquires about invariant physical geography or national capital cities.",
            ),
            # Computer science & protocols
            (
                re.compile(
                    r"\b(difference\s+between\s+inner\s+join|symmetric\s+multiprocessing|optimistic\s+and\s+pessimistic\s+locking|process\s+and\s+thread|pure\s+functions|mutex\s+and\s+semaphore|rsa\s+encryption|dns\s+resolution|osi\s+model|http\s+status\s+code|rfc\s+[0-9]+|cryptography|symmetric\s+and\s+asymmetric|binary\s+search\s+tree|dijkstra|stack\s+and\s+a\s+queue|tcp\s+vs\s+udp|tcp\s+three-way\s+handshake|git\s+rebase|machine\s+learning|breadth-first\s+search|depth-first\s+search|quicksort|mergesort)\b",
                    re.IGNORECASE,
                ),
                "invariant_cs_concept",
                "Query asks for timeless computer science, cryptography, and network architecture principles.",
            ),
            # Legal doctrines and codified statutes
            (
                re.compile(
                    r"\b(stare\s+decisis|statute\s+of\s+frauds|beyond\s+a\s+reasonable\s+doubt|miranda\s+warnings?|difference\s+between\s+copyright|berne\s+convention|article\s+17|rule\s+10b-5|section\s+101|section\s+107|form\s+1040|definition\s+of\s+negligence|habeas\s+corpus|res\s+judicata)\b",
                    re.IGNORECASE,
                ),
                "codified_legal_statute",
                "Query inquires about established legal doctrines, codified treaties, or statutory definitions.",
            ),
            # Standard sysadmin & operations syntax
            (
                re.compile(
                    r"^how\s+do\s+i\s+(check|change|find|create|configure|set\s+up|write|implement|inspect)\b.*\b(linux|command|cron|chmod|ssh|tar\.gz|find|free|ss|systemd|netplan|dockerfile|cors\s+headers|type\s+annotations|journalctl|df\s+-h|apirouter)\b",
                    re.IGNORECASE,
                ),
                "standard_sysadmin_command",
                "Query asks for standard sysadmin command syntax or deterministic tool configuration.",
            ),
        ]

        # 2. Dynamic Volatility Detectors
        self.dynamic_volatility_patterns: List[Tuple[Pattern[str], str, str]] = [
            # High-intensity real-time markers and common shorthands
            (
                re.compile(
                    r"\b(right\s+now|currently|at\s+the\s+moment|today|this\s+week|this\s+upcoming|this\s+calendar\s+year|last\s+night'?s|past\s+24\s+hours|last\s+24\s+hours|underway|\brn\b|\batm\b)\b",
                    re.IGNORECASE,
                ),
                "realtime_temporal_marker",
                "Query explicitly requests live, real-time, or intra-day temporal state.",
            ),
            # Financial quotes, prices, rates, indicators
            (
                re.compile(
                    r"\b(current\s+stock\s+price|price\s+of\s+bitcoin|price\s+of\s+ethereum|spot\s+exchange\s+rate|live\s+exchange\s+rate|market\s+capital(ization)?\s+of|p/e\s+ratio\s+of|fear\s+&\s+greed\s+index|consumer\s+sentiment\s+index|discount\s+rate\s+set\s+by|target\s+federal\s+funds|mortgage\s+rate|treasury\s+note|annual\s+inflation\s+rate|crude\s+oil\s+per\s+barrel|crude\s+palm\s+oil|physical\s+gold|ounce\s+of\s+silver|exchange\s+rate\s+between|valuation\s+of\s+the\s+s&p\s+500|gas\s+price\s+average|minimum\s+wage|btc\s+price|price\s+per\s+gallon|price\s+of\s+one\s+gram|treasury\s+bonds\s+today|refinance\s+index|current\s+price)\b",
                    re.IGNORECASE,
                ),
                "dynamic_financial_quote",
                "Query seeks live financial prices, equity valuations, exchange rates, or macroeconomic metrics.",
            ),
            # Real-time weather, natural events, transit, flight, queues
            (
                re.compile(
                    r"\b(weather\s+forecast|air\s+quality\s+index|uv\s+index|earthquake\s+activity|tornado\s+warnings?|wildfire\s+danger|level\s+of\s+water\s+in|flight\s+status|live\s+flight\s+status|traffic\s+on\b.*\b(freeway|highway|road)|status\s+of\s+the\b.*\b(line|train|metro)|wait\s+time\s+for\s+security|disaster\s+declaration|visa\s+appointment\s+wait\s+times|road\s+closures|emergency\s+room\s+wait\s+times|seismic\s+magnitude|travel\s+advisory|queue\s+length)\b",
                    re.IGNORECASE,
                ),
                "dynamic_realtime_environment",
                "Query asks about real-time environmental, meteorological, transit, or operational conditions.",
            ),
            # Breaking news and trending topics
            (
                re.compile(
                    r"\b(breaking\s+headlines|headline\s+story\s+on|trending\s+topics\s+on|trending\s+hashtags|latest\s+score\s+in|super\s+bowl\s+this\s+upcoming)\b",
                    re.IGNORECASE,
                ),
                "dynamic_news_sports_trends",
                "Query requests breaking news headlines, trending social media topics, or live sports scores.",
            ),
            # Current personnel / executives / political leaders / sports rosters
            (
                re.compile(
                    r"\b(who\s+is\s+the\s+current\s+(prime\s+minister|president|ceo|chief\s+executive|chancellor|governor|mayor)|who\s+is\s+the\s+chief\s+executive\s+officer|starting\s+quarterback)\b",
                    re.IGNORECASE,
                ),
                "dynamic_current_leadership",
                "Query asks for current political, corporate, or sports roster personnel subject to rotation.",
            ),
            # Moving software versions and dynamic API rates
            (
                re.compile(
                    r"\b(current\s+version\s+of\b.*\b(npm|mainline|pypi)|latest\s+version\s+of\b.*\b(recommended|lts|production|stable)|latest\s+updates\s+to\b.*\b(rate\s+limits|api)|newest\s+iphone)\b",
                    re.IGNORECASE,
                ),
                "dynamic_software_version",
                "Query inquires about rolling software releases or dynamic product/API specifications.",
            ),
            # Generic 'latest' or 'current' attached to volatile subjects
            (
                re.compile(
                    r"\b(what\s+is\s+the\s+(current|latest)\s+(price|status|rate|score|version|forecast|headline|cost|yield))\b",
                    re.IGNORECASE,
                ),
                "dynamic_generic_current_inquiry",
                "Query requests current/latest status of an inherently dynamic entity.",
            ),
        ]

        # 3. Generalized Answer-Type Taxonomy Patterns (Phase 1.1 Sub-stage C)
        # Tightened vocabulary guards prevent false positives on CS / math / theoretical terms.
        self.answer_type_patterns: List[Tuple[Pattern[str], str, str, Optional[Pattern[str]]]] = [
            # Financial & pricing state (excludes mathematical/CS values like "return value", "value of pi", "eigenvalue")
            (
                re.compile(
                    r"\b(pricing|worth|valuation|market\s+cap|trading\s+at|subscription\s+cost|how\s+much\s+does\s+.*\s+cost|how\s+much\s+is\s+(?:a|an|the|\$|[0-9]|one\b)|(?:current\s+)?price\s+of|(?:current\s+)?cost\s+of)\b",
                    re.IGNORECASE,
                ),
                "answer_type_financial_pricing",
                "Query asks about dynamic financial pricing, valuation, or asset cost.",
                re.compile(r"\b(formula|theory|definition|equation|proof|concept|return\s+value|eigenvalue|expected\s+value|value\s+of\s+(?:pi|e|\u03c0))\b", re.IGNORECASE),
            ),
            # Current-Role state (guarded to offices/positions, excludes author/inventor/founder/creator)
            (
                re.compile(
                    r"\b(who\s+is\s+the\s+(?:current\s+)?(?:president|prime\s+minister|ceo|chief\s+executive(?: officer)?|chairperson|governor|head\s+of\s+state|mayor|chancellor|minister|secretary\s+of\s+state)|who\s+(?:runs|leads)\s+(?:the\s+)?[A-Z][a-zA-Z0-9\s&]+|(?:president|prime\s+minister|ceo|chief\s+executive|chancellor|governor|mayor)\s+of\s+[A-Z][a-zA-Z0-9\s&]+)\b",
                    re.IGNORECASE,
                ),
                "answer_type_current_role",
                "Query asks about an office or leadership role currently held by an individual subject to rotation.",
                re.compile(r"\b(author|inventor|founder|creator|discoverer|pioneer|father\s+of|mother\s+of|first\s+president|first\s+ceo)\b", re.IGNORECASE),
            ),
            # Software version state (requires explicit versioning vocabulary)
            (
                re.compile(
                    r"\b(latest\s+version|current\s+version|newest\s+version|stable\s+version|lts\s+version|which\s+version)\b",
                    re.IGNORECASE,
                ),
                "answer_type_software_version",
                "Query inquires about latest software release or version specification.",
                None,
            ),
            # Bare Environmental / Weather state
            (
                re.compile(
                    r"^(?:what\s+is\s+the\s+)?(?:weather|temperature|humidity|air\s+quality|forecast)\s+(?:in|for|at)\s+[A-Za-z\s]+$|^(?:[A-Za-z\s]+)\s+(?:weather|forecast|temperature)$",
                    re.IGNORECASE,
                ),
                "answer_type_environmental_bare",
                "Query requests live environmental or meteorological condition for a location.",
                re.compile(r"\b(why|how|what\s+causes|explain|science\s+behind|physics\s+of|history\s+of)\b", re.IGNORECASE),
            ),
            # Statistic state (macroeconomic / demographics subject to periodic revisions)
            (
                re.compile(
                    r"\b(population\s+of|unemployment\s+rate\s+in|gdp\s+of|ranking\s+of|how\s+many\s+people\s+live\s+in)\b",
                    re.IGNORECASE,
                ),
                "answer_type_statistic",
                "Query seeks live macroeconomic or demographic statistics.",
                re.compile(r"\b(formula|definition|how\s+is\s+.*\s+calculated|why\s+is\s+gdp)\b", re.IGNORECASE),
            ),
            # Recommendation & purchase intent (guarded against CS/math/methodology phrasing)
            (
                re.compile(
                    r"\b(?:should\s+i\s+buy|which\s+.*\s+(?:is\s+better|should\s+i\s+buy)|best\s+(?:laptop|phone|camera|gpu|headphones?|tv|car|tool|smartphone|tablet|monitor|headset|processor|graphics\s+card|antivirus|vpn)|top\s+(?:laptops?|phones?|cameras?|gpus?|headphones?|tvs?|cars?|tools?|smartphones?|tablets?|monitors?|headsets?))\b",
                    re.IGNORECASE,
                ),
                "answer_type_recommendation",
                "Query seeks current product recommendation or purchase guidance.",
                re.compile(r"\b(algorithm|practice|practices|approach|method|pattern|sorting|complexity|data\s+structure|stack|architecture)\b", re.IGNORECASE),
            ),
            # Release & availability state
            (
                re.compile(
                    r"\b(is\s+.*\s+available|when\s+does\s+.*\s+(?:launch|release|come\s+out|ship)|when\s+is\s+.*\s+coming\s+out)\b",
                    re.IGNORECASE,
                ),
                "answer_type_release_availability",
                "Query asks for release timing or real-time availability of a product/feature.",
                re.compile(r"\b(press\s+release|memory\s+release|emotional\s+release|energy\s+release)\b", re.IGNORECASE),
            ),
        ]

    def _has_completed_temporal_anchor(self, query: str) -> bool:
        """Return True if query contains an explicit completed-period anchor."""
        # Live / real-time override cancels past anchor
        if re.search(r"\b(right\s+now|live|currently|today|this\s+week|\brn\b|\batm\b|current\s+price)\b", query, re.IGNORECASE):
            return False
        current_year = datetime.now().year
        for match in self.year_anchor_pattern.finditer(query):
            year_val = int(match.group(2))
            if year_val <= current_year:
                return True
        if re.search(r"\b(who\s+was|what\s+was|when\s+was|in\s+\d{4}|during\s+the\s+\d{4}|as\s+of\s+\d{4})\b", query, re.IGNORECASE):
            return True
        return False

    def evaluate(self, query: str) -> RuleMatch:
        """Evaluate the query against explainable stability and volatility rules.

        Returns:
            RuleMatch indicating whether a deterministic rule matched.
        """
        clean_query = query.strip()
        current_year = datetime.now().year

        # Step 1: Check for Mixed-Intent Dynamic Clauses
        # If any sub-clause contains an explicit volatile/dynamic request, the query MUST bypass cache
        sub_clauses = [c.strip() for c in self.clause_splitter.split(clean_query) if len(c.strip()) > 3]
        if len(sub_clauses) > 1:
            for clause in sub_clauses:
                for pattern, rule_name, reason in self.dynamic_volatility_patterns:
                    if pattern.search(clause):
                        return RuleMatch(
                            matched=True,
                            label=StabilityLabel.DYNAMIC,
                            rule_name=f"mixed_intent_guard_{rule_name}",
                            confidence=1.0,
                            rationale=f"Mixed-intent query: Sub-clause '{clause}' contains volatile request ({reason}). Entire query must bypass cache.",
                        )

        # Step 2: Dynamic Year Anchor Validation (Detects completed past calendar years)
        for match in self.year_anchor_pattern.finditer(clean_query):
            year_val = int(match.group(2))
            if year_val <= current_year:
                if not re.search(r"\b(right\s+now|live|currently|today|this\s+week|\brn\b|\batm\b|current\s+price)\b", clean_query, re.IGNORECASE):
                    return RuleMatch(
                        matched=True,
                        label=StabilityLabel.STABLE,
                        rule_name="guard_historical_past_year",
                        confidence=1.0,
                        rationale=f"Query references past historical year {year_val} (<= current year {current_year}) for completed event.",
                    )

        # Step 3: Named Historical & Past Fact Guards
        for pattern, rule_name, template in self.historical_named_patterns:
            match = pattern.search(clean_query)
            if match:
                if not re.search(r"\b(right\s+now|live|currently|today|this\s+week|\brn\b|\batm\b|current\s+price)\b", clean_query, re.IGNORECASE):
                    matched_text = match.group(0)
                    reason = template.format(anchor=matched_text)
                    return RuleMatch(
                        matched=True,
                        label=StabilityLabel.STABLE,
                        rule_name=f"guard_{rule_name}",
                        confidence=1.0,
                        rationale=reason,
                    )

        # Step 4: Check Dynamic Volatility Patterns across full query
        for pattern, rule_name, reason in self.dynamic_volatility_patterns:
            if pattern.search(clean_query):
                return RuleMatch(
                    matched=True,
                    label=StabilityLabel.DYNAMIC,
                    rule_name=f"dynamic_{rule_name}",
                    confidence=1.0,
                    rationale=reason,
                )

        # Step 4.5: Check Answer-Type Taxonomy Patterns (Phase 1.1 Sub-stage C)
        for pattern, rule_name, reason, exclusion in self.answer_type_patterns:
            if pattern.search(clean_query):
                # Check exclusion pattern first
                if exclusion and exclusion.search(clean_query):
                    continue
                # Check completed temporal anchor guard
                if self._has_completed_temporal_anchor(clean_query):
                    continue
                return RuleMatch(
                    matched=True,
                    label=StabilityLabel.DYNAMIC,
                    rule_name=f"dynamic_{rule_name}",
                    confidence=1.0,
                    rationale=reason,
                )

        # Step 5: Check Invariant Scientific / Mathematical / CS / Legal / Sysadmin Concepts
        for pattern, rule_name, reason in self.invariant_concept_patterns:
            if pattern.search(clean_query):
                return RuleMatch(
                    matched=True,
                    label=StabilityLabel.STABLE,
                    rule_name=f"invariant_{rule_name}",
                    confidence=1.0,
                    rationale=reason,
                )

        # Step 6: No high-confidence rule triggered -> Mark UNCERTAIN for fallback layer
        return RuleMatch(
            matched=False,
            label=StabilityLabel.UNCERTAIN,
            rule_name="no_rule_match",
            confidence=0.5,
            rationale="No high-confidence volatility or invariance pattern detected; deferred to fallback classifier.",
        )
