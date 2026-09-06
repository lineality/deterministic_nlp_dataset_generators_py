"""
Binary Intent Classification Dataset Generation Module (Version 2).

This module deterministically generates synthetic labeled training, testing,
and cross-validation datasets in JSON Lines (JSONL) format for a binary
question-type classification task:

    Class 0: 'KNOWN_EVENT_UNKNOWN_TIME'
        The user specifies an event name and asks for the temporal coordinate.
        (e.g., "What time does the board meeting start this evening?")

    Class 1: 'KNOWN_TIME_UNKNOWN_EVENT'
        The user specifies a temporal coordinate and asks for the scheduled event.
        (e.g., "What is scheduled at 20:00 UTC?")

Supported Command-Line Flags:
    --mode / -m:
        'split': Writes separate training and evaluation JSONL files.
        'unified': Writes a single consolidated JSONL dataset annotated with
                   deterministic K-Fold cross-validation indices.
    --standard-format:
        Transforms dictionary output schema into standardized NLP benchmark format:
        {'text': <query_string>, 'label': <0 or 1>, ...}
"""

import argparse
import hashlib
import json
import sys
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path

# =====================================================================
# SYSTEM CONSTANTS: QUESTION INTENT CLASS LABELS & INTEGER MAPPINGS
# =====================================================================

LABEL_KNOWN_EVENT_UNKNOWN_TIME: str = "KNOWN_EVENT_UNKNOWN_TIME"
LABEL_KNOWN_TIME_UNKNOWN_EVENT: str = "KNOWN_TIME_UNKNOWN_EVENT"

INTEGER_CLASS_LABEL_ZERO_EVENT_KNOWN: int = 0
INTEGER_CLASS_LABEL_ONE_TIME_KNOWN: int = 1

STANDARD_FORMAT_INTEGER_LABEL_MAPPING: dict[str, int] = {
    LABEL_KNOWN_EVENT_UNKNOWN_TIME: INTEGER_CLASS_LABEL_ZERO_EVENT_KNOWN,
    LABEL_KNOWN_TIME_UNKNOWN_EVENT: INTEGER_CLASS_LABEL_ONE_TIME_KNOWN,
}


# =====================================================================
# CLASS A DOMAIN DATA: EVENT NAMES, MODIFIERS, AND TIME TERMS
# =====================================================================

# Prefix modifiers for recombinant event name generation
EVENT_MODIFIER: tuple[str, ...] = (
    "",
    "the",
    "the first",
    "the charity",
    "the fundraiser",
    "the next",
    "the anticipated",
    "the indoor",
    "the outdoor",
    "the main floor",
    "the main deck",
    "the early deck",
    "the late deck",
)

DEFAULT_EVENT_MODIFIER_COLLECTION: tuple[str, ...] = EVENT_MODIFIER

# Standardized baseline event base names without hardcoded leading articles,
# formatted to accept recombinant prefix modifiers
DEFAULT_EVENT_NAME_COLLECTION: tuple[str, ...] = (
    "team standup",
    "marketing keynote",
    "quarterly review",
    "product launch",
    "security briefing",
    "sprint retrospective",
    "press conference",
    "board meeting",
    "system maintenance window",
    "project kickoff",
    "all-hands meeting",
    "architecture review",
    "client demo",
    "financial audit",
    "executive sync",
    "show",
    "breakfast",
    "movie",
    "dinner",
    "film",
    "ball",
    "dance",
    "comedy act",
    "game",
    "play",
    "reading",
    "book signing",
    "tennis match",
    "race",
    "fundraiser",
    "hay ride",
    "auction",
    "concert",
    "dog show",
    "singing contest",
    "talent show",
    "Birds screening",
    "murder mystery dinner",
)

DEFAULT_BASE_EVENT_NAME_COLLECTION: tuple[str, ...] = DEFAULT_EVENT_NAME_COLLECTION

DEFAULT_EVENT_TIME_TERM_COLLECTION: tuple[str, ...] = (
    "this morning",
    "this afternoon",
    "this evening",
    "tonight",
    "tomorrow",
    "tomorrow morning",
    "tomorrow afternoon",
    "tomorrow evening",
    "later today",
    "this Friday",
    "next Monday",
)


# =====================================================================
# CLASS B DOMAIN DATA: SYSTEMATIC MODULAR TIME EXPRESSIONS & NUMBERS
# =====================================================================

TIME_DIGIT_HOUR_COLLECTION: tuple[str, ...] = (
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "11",
    "12",
)

TIME_WORD_HOUR_COLLECTION: tuple[str, ...] = (
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "elevin",
    "twelve",
)

TIME_NUMBER_EXPRESSION_COLLECTION: tuple[str, ...] = (
    "1",
    "one",
    "2",
    "two",
    "3",
    "three",
    "4",
    "four",
    "5",
    "five",
    "6",
    "six",
    "7",
    "seven",
    "8",
    "eight",
    "9",
    "nine",
    "10",
    "ten",
    "11",
    "elevin",
    "eleven",
    "12",
    "twelve",
)

TIME_MINUTE_EXPRESSION_COLLECTION: tuple[str, ...] = (
    "15",
    "30",
    "45",
)

TIME_TWENTY_FOUR_HOUR_COLLECTION: tuple[str, ...] = (
    "08:00",
    "09:30",
    "10:00",
    "11:30",
    "13:00",
    "14:00",
    "15:30",
    "16:00",
    "17:00",
    "18:30",
    "19:00",
    "20:00",
    "21:30",
    "22:00",
    "23:00",
)

TIME_NAMED_INSTANT_COLLECTION: tuple[str, ...] = (
    "noon",
    "midday",
    "midnight",
)


def generate_systematic_time_expressions(
    hour_digit_strings: Sequence[str] = TIME_DIGIT_HOUR_COLLECTION,
    hour_word_strings: Sequence[str] = TIME_WORD_HOUR_COLLECTION,
    minute_strings: Sequence[str] = TIME_MINUTE_EXPRESSION_COLLECTION,
    twenty_four_hour_strings: Sequence[str] = TIME_TWENTY_FOUR_HOUR_COLLECTION,
    named_time_strings: Sequence[str] = TIME_NAMED_INSTANT_COLLECTION,
) -> tuple[str, ...]:
    """
    Systematically synthesize modular candidate time expressions.

    Constructs natural time expression strings across cardinal digits, English words,
    sub-hour minutes, AM/PM meridiem designators, military 24-hour designations,
    fractional relative phrasing (half past, quarter past, quarter to), and named
    temporal anchors.

    Parameters:
        hour_digit_strings: Sequence of 12-hour clock digit strings ('1' through '12').
        hour_word_strings: Sequence of 12-hour clock word strings ('one' through 'twelve').
        minute_strings: Sequence of clock minute intervals ('15', '30', '45').
        twenty_four_hour_strings: Sequence of 24-hour military clock expressions.
        named_time_strings: Sequence of lexical named times ('noon', 'midday', 'midnight').

    Returns:
        A deterministically sorted tuple of distinct time expression strings prefixed with 'at '.

    Raises:
        DatasetGenerationError: If time expression synthesis encounters an unhandled failure.
    """
    try:
        accumulated_time_expressions: list[str] = []

        # 1. Hour digits with direct, o'clock, meridiem, and fractional phrasing
        for digit_hour in sorted(set(hour_digit_strings)):
            accumulated_time_expressions.append(f"at {digit_hour}")
            accumulated_time_expressions.append(f"at {digit_hour} o'clock")
            accumulated_time_expressions.append(f"at {digit_hour}:00")
            accumulated_time_expressions.append(f"at {digit_hour} am")
            accumulated_time_expressions.append(f"at {digit_hour} pm")
            accumulated_time_expressions.append(f"at {digit_hour}:00 am")
            accumulated_time_expressions.append(f"at {digit_hour}:00 pm")
            accumulated_time_expressions.append(f"at half past {digit_hour}")
            accumulated_time_expressions.append(f"at quarter past {digit_hour}")
            accumulated_time_expressions.append(f"at quarter to {digit_hour}")

            for minute in sorted(set(minute_strings)):
                accumulated_time_expressions.append(f"at {digit_hour}:{minute}")
                accumulated_time_expressions.append(f"at {digit_hour}:{minute} am")
                accumulated_time_expressions.append(f"at {digit_hour}:{minute} pm")

        # 2. Hour words with direct, o'clock, and fractional phrasing
        for word_hour in sorted(set(hour_word_strings)):
            accumulated_time_expressions.append(f"at {word_hour}")
            accumulated_time_expressions.append(f"at {word_hour} o'clock")
            accumulated_time_expressions.append(f"at half past {word_hour}")
            accumulated_time_expressions.append(f"at quarter past {word_hour}")
            accumulated_time_expressions.append(f"at quarter to {word_hour}")

        # 3. 24-Hour military time expressions
        for military_time in sorted(set(twenty_four_hour_strings)):
            accumulated_time_expressions.append(f"at {military_time}")

        # 4. Lexical named time anchors
        for named_time in sorted(set(named_time_strings)):
            accumulated_time_expressions.append(f"at {named_time}")

        unique_sorted_expressions: tuple[str, ...] = tuple(sorted(set(accumulated_time_expressions)))
        return unique_sorted_expressions

    except Exception as time_synthesis_exception:
        detailed_traceback_string: str = traceback.format_exc()
        diagnostic_message: str = (
            f"Failed to systematically synthesize time expressions: "
            f"{time_synthesis_exception}\n{detailed_traceback_string}"
        )
        raise DatasetGenerationError(diagnostic_message) from time_synthesis_exception


DEFAULT_TIME_EXPRESSION_COLLECTION: tuple[str, ...] = generate_systematic_time_expressions()

DEFAULT_TIME_MODIFIER_COLLECTION: tuple[str, ...] = (
    "EST",
    "EDT",
    "CST",
    "CDT",
    "PST",
    "PDT",
    "UTC",
    "GMT",
    "CET",
    "CEST",
    "sharp",
    "local time",
)


# ---------------------------------------------------------------------
# Class A Templates: Known Event, Unknown Time
# ---------------------------------------------------------------------

TEMPLATES_KNOWN_EVENT_LEVEL_ONE: tuple[str, ...] = (
    # Direct Interrogatives
    "When is {event_name}?",
    "When is {event_name} taking place?",
    "When is {event_name} scheduled?",
    "When does {event_name} start?",
    "When does {event_name} begin?",
    "When does {event_name} kick off?",
    "When will {event_name} happen?",
    "When will {event_name} take place?",
    "What time is {event_name}?",
    "What time is {event_name} set for?",
    "What time is {event_name} scheduled for?",
    "What time does {event_name} start?",
    "What time does {event_name} begin?",
    "What time does {event_name} kick off?",
    "What time will {event_name} start?",
    "At what time is {event_name}?",
    "At what time does {event_name} start?",
    "At what time does {event_name} begin?",
    "At what time will {event_name} commence?",
    # Inverted Declarative / Subject First
    "{event_name} starts at what time?",
    "{event_name} begins at what time?",
    "{event_name} kicks off at what time?",
    "{event_name} is at what time?",
    "{event_name} is scheduled for what time?",
    "{event_name} starts when?",
    "{event_name} begins when?",
    # Nominal / Schedule Format
    "What is the start time of {event_name}?",
    "What is the kickoff time for {event_name}?",
    "What is the schedule for {event_name}?",
    # Indirect / Conversational
    "Could you tell me what time {event_name} starts?",
    "Can you check when {event_name} begins?",
    "Do you know what time {event_name} begins?",
    "Please let me know when {event_name} starts.",
    "I need to know what time {event_name} starts.",
    # Concise / Keyword
    "{event_name} start time",
    "{event_name} schedule time",
    "time for {event_name}",
)

TEMPLATES_KNOWN_EVENT_LEVEL_TWO: tuple[str, ...] = (
    # Postfixed Temporal Modifiers
    "When is {event_name} {time_term}?",
    "When does {event_name} start {time_term}?",
    "When does {event_name} begin {time_term}?",
    "What time is {event_name} {time_term}?",
    "What time does {event_name} start {time_term}?",
    "What time does {event_name} begin {time_term}?",
    "What time does {event_name} kick off {time_term}?",
    "At what time does {event_name} start {time_term}?",
    "{event_name} starts at what time {time_term}?",
    "{event_name} begins at what time {time_term}?",
    "{event_name} is at what time {time_term}?",
    "What is the start time for {event_name} {time_term}?",
    "Could you tell me what time {event_name} begins {time_term}?",
    "Please let me know when {event_name} starts {time_term}.",
    # Prefixed Temporal Modifiers
    "{time_term}, when is {event_name}?",
    "{time_term}, what time is {event_name}?",
    "{time_term}, what time does {event_name} start?",
    "{time_term}, at what time does {event_name} begin?",
    "{time_term}, could you tell me what time {event_name} starts?",
    # Prepositional Framing
    "For {time_term}, what time does {event_name} start?",
    "Looking at {time_term}, when does {event_name} take place?",
)


# ---------------------------------------------------------------------
# Class B Templates: Known Time, Unknown Event
# ---------------------------------------------------------------------

TEMPLATES_KNOWN_TIME_LEVEL_ONE: tuple[str, ...] = (
    # Direct Identity Queries
    "What is happening {time_expression}?",
    "What is taking place {time_expression}?",
    "What is scheduled {time_expression}?",
    "What is on the schedule {time_expression}?",
    "What is on the calendar {time_expression}?",
    "What is on the agenda {time_expression}?",
    "What occurs {time_expression}?",
    "What starts {time_expression}?",
    "What begins {time_expression}?",
    "What kicks off {time_expression}?",
    "What commences {time_expression}?",
    "What event is {time_expression}?",
    "What event starts {time_expression}?",
    "Which event is happening {time_expression}?",
    # Temporal Lead / Inverted Queries
    "{time_expression}, what is happening?",
    "{time_expression}, what is scheduled?",
    "{time_expression}, what event is taking place?",
    "{time_expression}, what meeting starts?",
    "{time_expression}, what is on the agenda?",
    "{time_expression}, what begins?",
    # Conversational & Polite Forms
    "Do you know what is scheduled {time_expression}?",
    "Do you know what is happening {time_expression}?",
    "Could you tell me what happens {time_expression}?",
    "Could you check what is scheduled {time_expression}?",
    "Can you tell me what event is {time_expression}?",
    "Please check what is taking place {time_expression}.",
    "Please let me know what starts {time_expression}.",
    "I need to know what event is {time_expression}.",
    # Schedule & Roster Inquiries
    "What does the schedule show {time_expression}?",
    "What do we have planned {time_expression}?",
    "What do we have scheduled {time_expression}?",
    "What do we have going on {time_expression}?",
    "What is booked {time_expression}?",
    "Is there an event scheduled {time_expression}?",
    "What takes place {time_expression}?",
)

TEMPLATES_KNOWN_TIME_LEVEL_TWO: tuple[str, ...] = (
    # Direct Queries with Timezone / Precision Modifiers
    "What is happening {time_expression} {time_modifier}?",
    "What is scheduled {time_expression} {time_modifier}?",
    "What is taking place {time_expression} {time_modifier}?",
    "What starts {time_expression} {time_modifier}?",
    "What begins {time_expression} {time_modifier}?",
    "What kicks off {time_expression} {time_modifier}?",
    "What event is scheduled {time_expression} {time_modifier}?",
    "What meeting starts {time_expression} {time_modifier}?",
    "What is on the calendar {time_expression} {time_modifier}?",
    "What is on the schedule {time_expression} {time_modifier}?",
    "What is on the agenda {time_expression} {time_modifier}?",
    "What session begins {time_expression} {time_modifier}?",
    "What do we have booked {time_expression} {time_modifier}?",
    # Prefixed Modifiers with Time Expression
    "{time_expression} {time_modifier}, what is happening?",
    "{time_expression} {time_modifier}, what is scheduled?",
    "{time_expression} {time_modifier}, what event takes place?",
    "{time_expression} {time_modifier}, what starts?",
    "{time_expression} {time_modifier}, what is on the agenda?",
    # Polite & Indirect Forms
    "Could you check what is scheduled {time_expression} {time_modifier}?",
    "Could you tell me what event begins {time_expression} {time_modifier}?",
    "Can you tell me what is happening {time_expression} {time_modifier}?",
    "Please let me know what event is {time_expression} {time_modifier}.",
    "Do you know what is taking place {time_expression} {time_modifier}?",
    "I need to verify what is scheduled {time_expression} {time_modifier}.",
)


# =====================================================================
# EXCEPTIONS
# =====================================================================

class DatasetGenerationError(Exception):
    """Base exception for pipeline operations."""


class TemplateValidationError(DatasetGenerationError):
    """Raised when templates do not contain mandatory replacement tokens."""


class ConfigurationParameterError(DatasetGenerationError):
    """Raised when an invalid argument value is configured."""


class FilePersistenceError(DatasetGenerationError):
    """Raised when writing output datasets to disk fails."""


# =====================================================================
# VALIDATION AND HASH UTILITIES
# =====================================================================

def validate_template_contains_placeholder_tokens(
    template_string: str,
    mandatory_placeholder_tokens: Sequence[str],
) -> None:
    """
    Ensure that a template string includes all mandatory placeholder tokens.

    Parameters:
        template_string: The string template to inspect.
        mandatory_placeholder_tokens: The expected token names without curly braces.

    Returns:
        None.

    Raises:
        TemplateValidationError: If any placeholder token is missing.
    """
    missing_tokens: list[str] = [
        f"{{{token}}}" for token in mandatory_placeholder_tokens if f"{{{token}}}" not in template_string
    ]
    if missing_tokens:
        diagnostic_message: str = (
            f"Template validation failed. The template '{template_string}' is missing "
            f"the required placeholder tokens: {missing_tokens}."
        )
        raise TemplateValidationError(diagnostic_message)


def compute_deterministic_dataset_split_assignment(
    unique_record_content_identifier: str,
    training_split_ratio: float,
    partition_salt_string: str,
) -> str:
    """
    Deterministically assign a record to 'train' or 'test' using salted SHA-256.

    Parameters:
        unique_record_content_identifier: A unique string identifier for the record.
        training_split_ratio: Fraction of records allocated to the training split.
        partition_salt_string: Cryptographic salt string for hash isolation.

    Returns:
        The partition label string: either 'train' or 'test'.

    Raises:
        ConfigurationParameterError: If training_split_ratio is out of range.
        DatasetGenerationError: If hash assignment fails.
    """
    if not (0.0 < training_split_ratio < 1.0):
        error_message: str = (
            f"Invalid training_split_ratio: {training_split_ratio}. Must be strictly between 0.0 and 1.0."
        )
        raise ConfigurationParameterError(error_message)

    try:
        hash_input_string: str = f"{partition_salt_string}::split::{unique_record_content_identifier}"
        digest_bytes: bytes = hashlib.sha256(hash_input_string.encode(encoding="utf-8")).digest()
        extracted_integer_value: int = int.from_bytes(digest_bytes[:8], byteorder="big", signed=False)
        maximum_possible_integer_value: int = 0xFFFFFFFFFFFFFFFF
        normalized_ratio: float = extracted_integer_value / maximum_possible_integer_value

        return "train" if normalized_ratio < training_split_ratio else "test"

    except Exception as runtime_exception:
        detailed_traceback: str = traceback.format_exc()
        diagnostic_message: str = (
            f"Failed to assign partition split for identifier '{unique_record_content_identifier}': "
            f"{runtime_exception}\n{detailed_traceback}"
        )
        raise DatasetGenerationError(diagnostic_message) from runtime_exception


def compute_deterministic_cross_validation_fold(
    unique_record_content_identifier: str,
    total_cross_validation_folds_count: int,
    partition_salt_string: str,
) -> int:
    """
    Deterministically assign a record to a zero-indexed cross-validation fold.

    Parameters:
        unique_record_content_identifier: A unique string identifier for the record.
        total_cross_validation_folds_count: Total integer count of folds (minimum 2).
        partition_salt_string: Cryptographic salt string for hash isolation.

    Returns:
        Zero-indexed fold integer in the range [0, total_cross_validation_folds_count - 1].

    Raises:
        ConfigurationParameterError: If total_cross_validation_folds_count is invalid.
        DatasetGenerationError: If fold calculation fails.
    """
    if total_cross_validation_folds_count < 2:
        error_message: str = (
            f"Invalid total_cross_validation_folds_count: {total_cross_validation_folds_count}. "
            "Cross-validation requires at least 2 distinct folds."
        )
        raise ConfigurationParameterError(error_message)

    try:
        hash_input_string: str = f"{partition_salt_string}::cv_fold::{unique_record_content_identifier}"
        digest_bytes: bytes = hashlib.sha256(hash_input_string.encode(encoding="utf-8")).digest()
        extracted_integer_value: int = int.from_bytes(digest_bytes[:8], byteorder="big", signed=False)
        return extracted_integer_value % total_cross_validation_folds_count

    except Exception as runtime_exception:
        detailed_traceback: str = traceback.format_exc()
        diagnostic_message: str = (
            f"Failed to compute CV fold for identifier '{unique_record_content_identifier}': "
            f"{runtime_exception}\n{detailed_traceback}"
        )
        raise DatasetGenerationError(diagnostic_message) from runtime_exception


# =====================================================================
# MODULAR-RECOMBINANT EVENT NAME BUILDER
# =====================================================================

def format_modular_recombinant_event_name(
    event_modifier_string: str,
    base_event_name_string: str,
) -> str:
    """
    Synthesize a recombinant event name from a prefix modifier and a base event name.

    Normalizes whitespace and determiners to prevent duplicate articles (e.g. 'the the').
    If the modifier is empty, the normalized base event name is returned.

    Parameters:
        event_modifier_string: Prefix modifier (e.g., 'the first', 'the charity', or '').
        base_event_name_string: Base event designation (e.g., 'team standup', 'breakfast').

    Returns:
        The combined recombinant event name string.

    Raises:
        DatasetGenerationError: If formatting encounters an unexpected failure.
    """
    try:
        normalized_modifier: str = event_modifier_string.strip()
        normalized_base_event: str = base_event_name_string.strip()

        # Prevent duplicate leading determiners if the modifier already specifies 'the'
        if normalized_modifier.lower().startswith("the ") and normalized_base_event.lower().startswith("the "):
            normalized_base_event = normalized_base_event[4:].strip()

        if not normalized_modifier:
            return normalized_base_event

        return f"{normalized_modifier} {normalized_base_event}"

    except Exception as formatting_exception:
        detailed_traceback: str = traceback.format_exc()
        diagnostic_message: str = (
            f"Failed to format modular recombinant event name from modifier '{event_modifier_string}' "
            f"and base '{base_event_name_string}': {formatting_exception}\n{detailed_traceback}"
        )
        raise DatasetGenerationError(diagnostic_message) from formatting_exception


def generate_modular_recombinant_event_names(
    base_event_names: Sequence[str],
    event_modifiers: Sequence[str],
) -> tuple[str, ...]:
    """
    Generate all Cartesian combinations of prefix modifiers and base event names.

    Parameters:
        base_event_names: Sequence of base event name strings.
        event_modifiers: Sequence of prefix modifier strings.

    Returns:
        A sorted tuple of unique recombinant event name strings.

    Raises:
        DatasetGenerationError: If modular event generation fails.
    """
    try:
        accumulated_event_names: list[str] = []
        for modifier_string in sorted(set(event_modifiers)):
            for base_event_string in sorted(set(base_event_names)):
                recombinant_name: str = format_modular_recombinant_event_name(
                    event_modifier_string=modifier_string,
                    base_event_name_string=base_event_string,
                )
                accumulated_event_names.append(recombinant_name)

        return tuple(sorted(set(accumulated_event_names)))

    except Exception as generation_exception:
        detailed_traceback: str = traceback.format_exc()
        diagnostic_message: str = (
            f"Failed to generate modular recombinant event names: "
            f"{generation_exception}\n{detailed_traceback}"
        )
        raise DatasetGenerationError(diagnostic_message) from generation_exception


# =====================================================================
# FORMAT TRANSFORMATION
# =====================================================================

def transform_canonical_record_to_standard_nlp_schema(
    canonical_record_dictionary: Mapping[str, object],
) -> dict[str, object]:
    """
    Convert a canonical internal record dictionary into standard NLP benchmark schema.

    Replaces 'query_text' with 'text', transforms 'intent_classification_label'
    into an integer 'label' (0 for KNOWN_EVENT_UNKNOWN_TIME, 1 for KNOWN_TIME_UNKNOWN_EVENT),
    and preserves metadata for partitioning and cross-validation.

    Parameters:
        canonical_record_dictionary: The canonical domain record dictionary.

    Returns:
        A dictionary with standardized 'text' and 'label' keys.

    Raises:
        ConfigurationParameterError: If an unrecognized classification label is encountered.
    """
    raw_intent_label: object = canonical_record_dictionary.get("intent_classification_label")
    if not isinstance(raw_intent_label, str) or raw_intent_label not in STANDARD_FORMAT_INTEGER_LABEL_MAPPING:
        raise ConfigurationParameterError(
            f"Record contains an unrecognized intent classification label: '{raw_intent_label}'."
        )

    integer_class_label: int = STANDARD_FORMAT_INTEGER_LABEL_MAPPING[raw_intent_label]

    return {
        "text": str(canonical_record_dictionary["query_text"]),
        "label": integer_class_label,
        "sample_unique_identifier": canonical_record_dictionary["sample_unique_identifier"],
        "assigned_dataset_partition": canonical_record_dictionary["assigned_dataset_partition"],
        "deterministic_cross_validation_fold_index": canonical_record_dictionary[
            "deterministic_cross_validation_fold_index"
        ],
    }


# =====================================================================
# RECORD BUILDERS
# =====================================================================

def construct_inquiry_record_dictionary(
    query_text: str,
    raw_template_format: str,
    template_complexity_level: int,
    intent_classification_label: str,
    event_name_entity: str | None,
    time_term_entity: str | None,
    time_expression_entity: str | None,
    time_modifier_entity: str | None,
    training_split_ratio: float,
    partition_salt_string: str,
    total_cross_validation_folds_count: int,
    event_modifier_entity: str | None = None,
    base_event_name_entity: str | None = None,
) -> dict[str, object]:
    """
    Instantiate a canonical domain record dictionary with cryptographic identifiers.

    Parameters:
        query_text: The fully rendered natural language query.
        raw_template_format: The template string used to format the query.
        template_complexity_level: Level identifier integer (1 or 2).
        intent_classification_label: The class label string.
        event_name_entity: Extracted fully rendered event name entity or None.
        time_term_entity: Extracted time term modifier or None.
        time_expression_entity: Extracted time expression entity or None.
        time_modifier_entity: Extracted timezone or precision modifier or None.
        training_split_ratio: Proportion of records allocated to the training partition.
        partition_salt_string: Salt for deterministic hashing.
        total_cross_validation_folds_count: Total cross-validation fold count.
        event_modifier_entity: Extracted prefix modifier used to synthesize event name or None.
        base_event_name_entity: Extracted base event designation without prefix or None.

    Returns:
        A dictionary containing the canonical domain record.
    """
    assigned_dataset_partition: str = compute_deterministic_dataset_split_assignment(
        unique_record_content_identifier=query_text,
        training_split_ratio=training_split_ratio,
        partition_salt_string=partition_salt_string,
    )
    assigned_cross_validation_fold: int = compute_deterministic_cross_validation_fold(
        unique_record_content_identifier=query_text,
        total_cross_validation_folds_count=total_cross_validation_folds_count,
        partition_salt_string=partition_salt_string,
    )
    content_hash_digest: str = hashlib.sha256(query_text.encode(encoding="utf-8")).hexdigest()

    return {
        "sample_unique_identifier": content_hash_digest,
        "query_text": query_text,
        "intent_classification_label": intent_classification_label,
        "template_complexity_level": template_complexity_level,
        "raw_template_format": raw_template_format,
        "assigned_dataset_partition": assigned_dataset_partition,
        "deterministic_cross_validation_fold_index": assigned_cross_validation_fold,
        "extracted_entities": {
            "event_name": event_name_entity,
            "event_modifier": event_modifier_entity,
            "base_event_name": base_event_name_entity,
            "time_term": time_term_entity,
            "time_expression": time_expression_entity,
            "time_modifier": time_modifier_entity,
        },
    }


def generate_known_event_unknown_time_records(
    event_names: Sequence[str],
    time_terms: Sequence[str],
    level_one_templates: Sequence[str],
    level_two_templates: Sequence[str],
    training_split_ratio: float,
    partition_salt_string: str,
    total_cross_validation_folds_count: int,
    event_modifiers: Sequence[str] | None = None,
) -> list[dict[str, object]]:
    """
    Generate all inquiry records for Class 0 (KNOWN_EVENT_UNKNOWN_TIME).

    When event_modifiers are supplied, each base event name is combined
    recombinantly with every modifier to produce exhaustive variants.

    Parameters:
        event_names: Sequence of candidate base event names.
        time_terms: Sequence of candidate temporal expressions.
        level_one_templates: 1-level templates accepting '{event_name}'.
        level_two_templates: 2-level templates accepting '{event_name}' and '{time_term}'.
        training_split_ratio: Proportion of records allocated to the training partition.
        partition_salt_string: Salt for deterministic hashing.
        total_cross_validation_folds_count: Total cross-validation fold count.
        event_modifiers: Sequence of candidate prefix modifiers, or None for direct generation.

    Returns:
        List of generated canonical record dictionaries.
    """
    accumulated_records: list[dict[str, object]] = []

    # Assemble tuples of (event_modifier_entity, base_event_entity, rendered_event_name)
    recombinant_event_tuples: list[tuple[str | None, str, str]] = []
    if event_modifiers is not None:
        for modifier_candidate in sorted(set(event_modifiers)):
            for base_event_candidate in sorted(set(event_names)):
                rendered_name: str = format_modular_recombinant_event_name(
                    event_modifier_string=modifier_candidate,
                    base_event_name_string=base_event_candidate,
                )
                modifier_entity: str | None = modifier_candidate.strip() if modifier_candidate.strip() else None
                recombinant_event_tuples.append(
                    (modifier_entity, base_event_candidate.strip(), rendered_name)
                )
    else:
        for event_name_candidate in sorted(set(event_names)):
            cleaned_event: str = event_name_candidate.strip()
            recombinant_event_tuples.append((None, cleaned_event, cleaned_event))

    # Level 1 templates: direct inquiry regarding event time
    for template_str in sorted(set(level_one_templates)):
        validate_template_contains_placeholder_tokens(template_str, ("event_name",))
        for modifier_entity, base_event_entity, rendered_event_str in recombinant_event_tuples:
            rendered_text: str = template_str.format(event_name=rendered_event_str)
            record: dict[str, object] = construct_inquiry_record_dictionary(
                query_text=rendered_text,
                raw_template_format=template_str,
                template_complexity_level=1,
                intent_classification_label=LABEL_KNOWN_EVENT_UNKNOWN_TIME,
                event_name_entity=rendered_event_str,
                time_term_entity=None,
                time_expression_entity=None,
                time_modifier_entity=None,
                training_split_ratio=training_split_ratio,
                partition_salt_string=partition_salt_string,
                total_cross_validation_folds_count=total_cross_validation_folds_count,
                event_modifier_entity=modifier_entity,
                base_event_name_entity=base_event_entity,
            )
            accumulated_records.append(record)

    # Level 2 templates: event query contextualized by broad time term
    for template_str in sorted(set(level_two_templates)):
        validate_template_contains_placeholder_tokens(template_str, ("event_name", "time_term"))
        for modifier_entity, base_event_entity, rendered_event_str in recombinant_event_tuples:
            for time_term_str in sorted(set(time_terms)):
                rendered_text = template_str.format(
                    event_name=rendered_event_str,
                    time_term=time_term_str,
                )
                record = construct_inquiry_record_dictionary(
                    query_text=rendered_text,
                    raw_template_format=template_str,
                    template_complexity_level=2,
                    intent_classification_label=LABEL_KNOWN_EVENT_UNKNOWN_TIME,
                    event_name_entity=rendered_event_str,
                    time_term_entity=time_term_str,
                    time_expression_entity=None,
                    time_modifier_entity=None,
                    training_split_ratio=training_split_ratio,
                    partition_salt_string=partition_salt_string,
                    total_cross_validation_folds_count=total_cross_validation_folds_count,
                    event_modifier_entity=modifier_entity,
                    base_event_name_entity=base_event_entity,
                )
                accumulated_records.append(record)

    return accumulated_records


def generate_known_time_unknown_event_records(
    time_expressions: Sequence[str],
    time_modifiers: Sequence[str],
    level_one_templates: Sequence[str],
    level_two_templates: Sequence[str],
    training_split_ratio: float,
    partition_salt_string: str,
    total_cross_validation_folds_count: int,
) -> list[dict[str, object]]:
    """
    Generate all inquiry records for Class 1 (KNOWN_TIME_UNKNOWN_EVENT).

    Parameters:
        time_expressions: Sequence of candidate base time expressions.
        time_modifiers: Sequence of candidate time modifiers (e.g., timezones).
        level_one_templates: 1-level templates accepting '{time_expression}'.
        level_two_templates: 2-level templates accepting '{time_expression}' and '{time_modifier}'.
        training_split_ratio: Proportion of records allocated to the training partition.
        partition_salt_string: Salt for deterministic hashing.
        total_cross_validation_folds_count: Total cross-validation fold count.

    Returns:
        List of generated canonical record dictionaries.
    """
    accumulated_records: list[dict[str, object]] = []

    # Level 1 templates: direct inquiry regarding events scheduled at a given time expression
    for template_str in sorted(set(level_one_templates)):
        validate_template_contains_placeholder_tokens(template_str, ("time_expression",))
        for time_expr_str in sorted(set(time_expressions)):
            rendered_text: str = template_str.format(time_expression=time_expr_str)
            record: dict[str, object] = construct_inquiry_record_dictionary(
                query_text=rendered_text,
                raw_template_format=template_str,
                template_complexity_level=1,
                intent_classification_label=LABEL_KNOWN_TIME_UNKNOWN_EVENT,
                event_name_entity=None,
                time_term_entity=None,
                time_expression_entity=time_expr_str,
                time_modifier_entity=None,
                training_split_ratio=training_split_ratio,
                partition_salt_string=partition_salt_string,
                total_cross_validation_folds_count=total_cross_validation_folds_count,
                event_modifier_entity=None,
                base_event_name_entity=None,
            )
            accumulated_records.append(record)

    # Level 2 templates: inquiry contextualized with precision or timezone modifiers
    for template_str in sorted(set(level_two_templates)):
        validate_template_contains_placeholder_tokens(
            template_str, ("time_expression", "time_modifier")
        )
        for time_expr_str in sorted(set(time_expressions)):
            for time_modifier_str in sorted(set(time_modifiers)):
                rendered_text = template_str.format(
                    time_expression=time_expr_str,
                    time_modifier=time_modifier_str,
                )
                record = construct_inquiry_record_dictionary(
                    query_text=rendered_text,
                    raw_template_format=template_str,
                    template_complexity_level=2,
                    intent_classification_label=LABEL_KNOWN_TIME_UNKNOWN_EVENT,
                    event_name_entity=None,
                    time_term_entity=None,
                    time_expression_entity=time_expr_str,
                    time_modifier_entity=time_modifier_str,
                    training_split_ratio=training_split_ratio,
                    partition_salt_string=partition_salt_string,
                    total_cross_validation_folds_count=total_cross_validation_folds_count,
                    event_modifier_entity=None,
                    base_event_name_entity=None,
                )
                accumulated_records.append(record)

    return accumulated_records


def balance_binary_class_records(
    class_a_records: list[dict[str, object]],
    class_b_records: list[dict[str, object]],
    balance_salt_string: str,
) -> list[dict[str, object]]:
    """
    Sub-sample majority class deterministically to produce an exact 50/50 balance.

    Parameters:
        class_a_records: Records generated for Class A.
        class_b_records: Records generated for Class B.
        balance_salt_string: Salt for deterministic sorting.

    Returns:
        A combined and deterministically balanced list of canonical records.
    """
    minimum_cardinality: int = min(len(class_a_records), len(class_b_records))

    def _compute_deterministic_ranking_key(canonical_record: dict[str, object]) -> str:
        record_identifier_string: str = str(canonical_record["sample_unique_identifier"])
        salted_key_input: str = f"{balance_salt_string}::balance::{record_identifier_string}"
        return hashlib.sha256(salted_key_input.encode(encoding="utf-8")).hexdigest()

    sorted_class_a_records: list[dict[str, object]] = sorted(
        class_a_records,
        key=_compute_deterministic_ranking_key,
    )
    sorted_class_b_records: list[dict[str, object]] = sorted(
        class_b_records,
        key=_compute_deterministic_ranking_key,
    )

    combined_records_list: list[dict[str, object]] = (
        sorted_class_a_records[:minimum_cardinality] + sorted_class_b_records[:minimum_cardinality]
    )

    return sorted(
        combined_records_list,
        key=lambda canonical_record: str(canonical_record["sample_unique_identifier"]),
    )


# =====================================================================
# PERSISTENCE
# =====================================================================

def write_records_to_jsonl_file(
    record_dictionary_sequence: Sequence[Mapping[str, object]],
    target_destination_file_path: Path,
) -> int:
    """
    Persist record dictionaries to disk in UTF-8 JSON Lines format.

    Parameters:
        record_dictionary_sequence: Sequence of dictionaries to serialize.
        target_destination_file_path: Destination path for the output JSONL file.

    Returns:
        Total count of written lines.

    Raises:
        FilePersistenceError: If writing to the destination path fails.
    """
    try:
        target_destination_file_path.parent.mkdir(parents=True, exist_ok=True)
        written_line_counter: int = 0
        with open(target_destination_file_path, mode="w", encoding="utf-8") as file_writer:
            for record_dictionary in record_dictionary_sequence:
                serialized_line: str = json.dumps(record_dictionary, ensure_ascii=False, sort_keys=True)
                file_writer.write(serialized_line + "\n")
                written_line_counter += 1
        return written_line_counter

    except Exception as io_exception:
        traceback_string: str = traceback.format_exc()
        diagnostic_message: str = (
            f"Failed to persist records to JSONL file at '{target_destination_file_path}': "
            f"{io_exception}\n{traceback_string}"
        )
        raise FilePersistenceError(diagnostic_message) from io_exception


# =====================================================================
# PIPELINE EXECUTION
# =====================================================================

def execute_question_type_dataset_pipeline(
    output_directory_path: Path,
    dataset_organization_mode: str = "split",
    balanced_mode_enabled: bool = True,
    standard_format_enabled: bool = False,
    training_split_ratio: float = 0.80,
    total_cross_validation_folds_count: int = 5,
    partition_salt_string: str = "question_type_classification_salt_2026",
    event_modifiers: Sequence[str] = EVENT_MODIFIER,
) -> dict[str, object]:
    """
    Orchestrate dataset generation, balancing, format selection, and disk persistence.

    Parameters:
        output_directory_path: Directory path where output files will be written.
        dataset_organization_mode: Either 'split' or 'unified'.
        balanced_mode_enabled: Whether to balance both classes to equal cardinality.
        standard_format_enabled: Whether to emit 'text' and 'label' (int) keys.
        training_split_ratio: Proportion of records allocated to the training partition.
        total_cross_validation_folds_count: Total cross-validation fold count.
        partition_salt_string: Salt for deterministic hashing.
        event_modifiers: Sequence of prefix modifiers for modular-recombinant event construction.

    Returns:
        A dictionary containing summary metrics of the pipeline execution.

    Raises:
        ConfigurationParameterError: If invalid operational modes or settings are given.
        FilePersistenceError: If serialization fails.
        DatasetGenerationError: If record generation fails.
    """
    normalized_mode_string: str = dataset_organization_mode.strip().lower()
    if normalized_mode_string not in ("split", "unified"):
        raise ConfigurationParameterError(
            f"Unsupported dataset organization mode: '{dataset_organization_mode}'. Must be 'split' or 'unified'."
        )

    print("[PIPELINE STATUS] Generating Class A records (KNOWN_EVENT_UNKNOWN_TIME) with prefix modifiers...")
    class_a_canonical_records: list[dict[str, object]] = generate_known_event_unknown_time_records(
        event_names=DEFAULT_EVENT_NAME_COLLECTION,
        time_terms=DEFAULT_EVENT_TIME_TERM_COLLECTION,
        level_one_templates=TEMPLATES_KNOWN_EVENT_LEVEL_ONE,
        level_two_templates=TEMPLATES_KNOWN_EVENT_LEVEL_TWO,
        training_split_ratio=training_split_ratio,
        partition_salt_string=partition_salt_string,
        total_cross_validation_folds_count=total_cross_validation_folds_count,
        event_modifiers=event_modifiers,
    )
    print(f"[PIPELINE STATUS] Generated {len(class_a_canonical_records)} records for Class A.")

    print("[PIPELINE STATUS] Generating Class B records (KNOWN_TIME_UNKNOWN_EVENT) with systematic numbers...")
    class_b_canonical_records: list[dict[str, object]] = generate_known_time_unknown_event_records(
        time_expressions=DEFAULT_TIME_EXPRESSION_COLLECTION,
        time_modifiers=DEFAULT_TIME_MODIFIER_COLLECTION,
        level_one_templates=TEMPLATES_KNOWN_TIME_LEVEL_ONE,
        level_two_templates=TEMPLATES_KNOWN_TIME_LEVEL_TWO,
        training_split_ratio=training_split_ratio,
        partition_salt_string=partition_salt_string,
        total_cross_validation_folds_count=total_cross_validation_folds_count,
    )
    print(f"[PIPELINE STATUS] Generated {len(class_b_canonical_records)} records for Class B.")

    canonical_records_pool: list[dict[str, object]]
    if balanced_mode_enabled:
        print("[PIPELINE STATUS] Balancing classes to identical record cardinality...")
        canonical_records_pool = balance_binary_class_records(
            class_a_records=class_a_canonical_records,
            class_b_records=class_b_canonical_records,
            balance_salt_string=partition_salt_string,
        )
        print(f"[PIPELINE STATUS] Balanced dataset total records: {len(canonical_records_pool)}")
    else:
        print("[PIPELINE STATUS] Retaining full combinatorial imbalance...")
        canonical_records_pool = sorted(
            class_a_canonical_records + class_b_canonical_records,
            key=lambda canonical_record: str(canonical_record["sample_unique_identifier"]),
        )

    pipeline_summary: dict[str, object] = {
        "execution_mode": normalized_mode_string,
        "standard_nlp_format_enabled": standard_format_enabled,
        "balanced_subsampling_active": balanced_mode_enabled,
        "total_records_processed": len(canonical_records_pool),
    }

    if normalized_mode_string == "split":
        training_canonical_records: list[dict[str, object]] = [
            record for record in canonical_records_pool if record["assigned_dataset_partition"] == "train"
        ]
        evaluation_canonical_records: list[dict[str, object]] = [
            record for record in canonical_records_pool if record["assigned_dataset_partition"] == "test"
        ]

        # Count per-class distribution within each partition
        train_class_a_count: int = sum(
            1 for item in training_canonical_records
            if item["intent_classification_label"] == LABEL_KNOWN_EVENT_UNKNOWN_TIME
        )
        train_class_b_count: int = sum(
            1 for item in training_canonical_records
            if item["intent_classification_label"] == LABEL_KNOWN_TIME_UNKNOWN_EVENT
        )
        test_class_a_count: int = sum(
            1 for item in evaluation_canonical_records
            if item["intent_classification_label"] == LABEL_KNOWN_EVENT_UNKNOWN_TIME
        )
        test_class_b_count: int = sum(
            1 for item in evaluation_canonical_records
            if item["intent_classification_label"] == LABEL_KNOWN_TIME_UNKNOWN_EVENT
        )

        training_records_to_persist: Sequence[Mapping[str, object]]
        evaluation_records_to_persist: Sequence[Mapping[str, object]]

        if standard_format_enabled:
            print("[PIPELINE STATUS] Projecting records to standard format schema ('text', 'label')...")
            training_records_to_persist = [
                transform_canonical_record_to_standard_nlp_schema(record)
                for record in training_canonical_records
            ]
            evaluation_records_to_persist = [
                transform_canonical_record_to_standard_nlp_schema(record)
                for record in evaluation_canonical_records
            ]
        else:
            training_records_to_persist = training_canonical_records
            evaluation_records_to_persist = evaluation_canonical_records

        train_file_path: Path = output_directory_path / "train_question_type_dataset.jsonl"
        test_file_path: Path = output_directory_path / "test_question_type_dataset.jsonl"

        written_train_count: int = write_records_to_jsonl_file(
            record_dictionary_sequence=training_records_to_persist,
            target_destination_file_path=train_file_path,
        )
        written_test_count: int = write_records_to_jsonl_file(
            record_dictionary_sequence=evaluation_records_to_persist,
            target_destination_file_path=test_file_path,
        )

        print(
            f"[SUCCESS] Wrote Training Set: {train_file_path}\n"
            f"          Total: {written_train_count} | Class A: {train_class_a_count} | Class B: {train_class_b_count}"
        )
        print(
            f"[SUCCESS] Wrote Testing Set: {test_file_path}\n"
            f"          Total: {written_test_count} | Class A: {test_class_a_count} | Class B: {test_class_b_count}"
        )

        pipeline_summary["train_file_path"] = str(train_file_path)
        pipeline_summary["train_records_count"] = written_train_count
        pipeline_summary["test_file_path"] = str(test_file_path)
        pipeline_summary["test_records_count"] = written_test_count

    elif normalized_mode_string == "unified":
        total_class_a_count: int = sum(
            1 for item in canonical_records_pool
            if item["intent_classification_label"] == LABEL_KNOWN_EVENT_UNKNOWN_TIME
        )
        total_class_b_count: int = sum(
            1 for item in canonical_records_pool
            if item["intent_classification_label"] == LABEL_KNOWN_TIME_UNKNOWN_EVENT
        )

        unified_records_to_persist: Sequence[Mapping[str, object]]
        if standard_format_enabled:
            print("[PIPELINE STATUS] Projecting records to standard format schema ('text', 'label')...")
            unified_records_to_persist = [
                transform_canonical_record_to_standard_nlp_schema(record)
                for record in canonical_records_pool
            ]
        else:
            unified_records_to_persist = canonical_records_pool

        unified_file_path: Path = output_directory_path / "unified_question_type_dataset.jsonl"
        written_unified_count: int = write_records_to_jsonl_file(
            record_dictionary_sequence=unified_records_to_persist,
            target_destination_file_path=unified_file_path,
        )

        print(
            f"[SUCCESS] Wrote Unified Set: {unified_file_path}\n"
            f"          Total: {written_unified_count} | Class A: {total_class_a_count} | Class B: {total_class_b_count}"
        )
        print(f"[PIPELINE STATUS] K-Fold CV Folds Assigned: 0 through {total_cross_validation_folds_count - 1}")

        pipeline_summary["unified_file_path"] = str(unified_file_path)
        pipeline_summary["unified_records_count"] = written_unified_count
        pipeline_summary["cross_validation_folds_count"] = total_cross_validation_folds_count

    return pipeline_summary


# =====================================================================
# CLI PARSER AND ENTRYPOINT
# =====================================================================

def construct_command_line_argument_parser() -> argparse.ArgumentParser:
    """
    Construct and configure the command-line argument parser.

    Returns:
        Configured argparse.ArgumentParser instance.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Deterministic 2-Class Question-Type Dataset Generator (JSONL).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-m",
        "--mode",
        "--dataset-organization-mode",
        dest="dataset_organization_mode",
        type=str,
        choices=["split", "unified"],
        default="split",
        help="Organization mode: 'split' outputs train/test files; 'unified' outputs a single file with CV folds.",
    )
    parser.add_argument(
        "-o",
        "--output-directory",
        dest="output_directory_path",
        type=str,
        default="./question_type_dataset_output",
        help="Filesystem directory where output files will be written.",
    )
    parser.add_argument(
        "-r",
        "--train-split-ratio",
        dest="training_split_ratio",
        type=float,
        default=0.80,
        help="Fraction of records allocated to the training split (used in 'split' mode).",
    )
    parser.add_argument(
        "-k",
        "--cross-validation-folds-count",
        dest="cross_validation_folds_count",
        type=int,
        default=5,
        help="Number of deterministic cross-validation folds (used in 'unified' mode).",
    )
    parser.add_argument(
        "-s",
        "--partition-salt",
        dest="partition_salt_string",
        type=str,
        default="binary_question_classification_seed_v2026",
        help="Cryptographic salt string for hashing.",
    )
    parser.add_argument(
        "--standard-format",
        dest="standard_format_enabled",
        action="store_true",
        default=False,
        help=(
            "Emit standard NLP benchmark format: maps queries to key 'text' and "
            "classification targets to key 'label' (0 for KNOWN_EVENT_UNKNOWN_TIME, "
            "1 for KNOWN_TIME_UNKNOWN_EVENT)."
        ),
    )
    parser.add_argument(
        "--balanced",
        dest="balanced_mode_enabled",
        action="store_true",
        default=True,
        help="Subsample majority class to maintain an exact 1:1 binary class distribution.",
    )
    parser.add_argument(
        "--no-balanced",
        dest="balanced_mode_enabled",
        action="store_false",
        help="Disable balancing and output the full combinatorial Cartesian product for both classes.",
    )
    return parser


def main() -> None:
    """Primary execution entrypoint parsing CLI options and invoking the pipeline."""
    argument_parser: argparse.ArgumentParser = construct_command_line_argument_parser()
    parsed_cli_arguments: argparse.Namespace = argument_parser.parse_args()

    try:
        output_directory_path: Path = Path(parsed_cli_arguments.output_directory_path)

        execute_question_type_dataset_pipeline(
            output_directory_path=output_directory_path,
            dataset_organization_mode=parsed_cli_arguments.dataset_organization_mode,
            balanced_mode_enabled=parsed_cli_arguments.balanced_mode_enabled,
            standard_format_enabled=parsed_cli_arguments.standard_format_enabled,
            training_split_ratio=parsed_cli_arguments.training_split_ratio,
            total_cross_validation_folds_count=parsed_cli_arguments.cross_validation_folds_count,
            partition_salt_string=parsed_cli_arguments.partition_salt_string,
            event_modifiers=EVENT_MODIFIER,
        )
        sys.exit(0)

    except Exception as unhandled_execution_exception:
        detailed_traceback_string: str = traceback.format_exc()
        sys.stderr.write(
            f"\n[FATAL PIPELINE FAILURE] Execution terminated unexpectedly:\n"
            f"Exception: {unhandled_execution_exception}\n\n"
            f"Execution Traceback:\n{detailed_traceback_string}\n"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
