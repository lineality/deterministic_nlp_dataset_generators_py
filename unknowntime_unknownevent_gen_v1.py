"""
Binary Intent Classification Dataset Generation Module.

This module deterministically generates synthetic labeled training, testing,
and cross-validation datasets in JSON Lines (JSONL) format for a binary
question-type classification task:

    Class 0: 'KNOWN_EVENT_UNKNOWN_TIME'
        The user specifies an event name and asks for the temporal coordinate.
        (e.g., "What time does the board meeting start this evening?")

    Class 1: 'KNOWN_TIME_UNKNOWN_EVENT'
        The user specifies a temporal coordinate and asks for the scheduled event.
        (e.g., "What is scheduled at 20:00 UTC?")

Supported Dataset Output Modes:
    1. 'split': Emits separate training and evaluation JSONL files.
    2. 'unified': Emits a single consolidated JSONL dataset annotated with
       deterministic K-Fold cross-validation indices.
"""



"""
Binary Split (Default Balanced 80/20 Train/Test)
python generate_question_type_dataset.py \
    --mode split \
    --train-split-ratio 0.80 \
    --output-directory ./data_split


Unified 10-Fold Cross-Validation Mode
```bash
python generate_question_type_dataset.py \
    --mode unified \
    --cross-validation-folds-count 10 \
    --output-directory ./data_unified
```

Unbalanced Mode (Preserves Exhaustive Cartesian Combinations)

```bash
python generate_question_type_dataset.py \
    --mode split \
    --no-balanced \
    --output-directory ./data_exhaustive
```

"""

import argparse
import hashlib
import json
import sys
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path

# =====================================================================
# SYSTEM CONSTANTS: QUESTION INTENT CLASS LABELS
# =====================================================================

LABEL_KNOWN_EVENT_UNKNOWN_TIME: str = "KNOWN_EVENT_UNKNOWN_TIME"
LABEL_KNOWN_TIME_UNKNOWN_EVENT: str = "KNOWN_TIME_UNKNOWN_EVENT"


# =====================================================================
# VOCABULARY AND TEMPLATE COLLECTIONS
# =====================================================================

# ---------------------------------------------------------------------
# Entity Lexicons
# ---------------------------------------------------------------------

DEFAULT_EVENT_NAME_COLLECTION: tuple[str, ...] = (
    "the team standup",
    "the marketing keynote",
    "the quarterly review",
    "the product launch",
    "the security briefing",
    "the sprint retrospective",
    "the press conference",
    "the board meeting",
    "the system maintenance window",
    "the project kickoff",
    "the all-hands meeting",
    "the architecture review",
    "the client demo",
    "the financial audit",
    "the executive sync",
)

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

# TODO: not used yet
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
    "12",
    "twelve",
)

"""
TODO
This should be modular, with numbers from another set of options
"""
DEFAULT_TIME_EXPRESSION_COLLECTION: tuple[str, ...] = (
    "at eight",
    "at 8",
    "at 8 o'clock",
    "at 8:00",
    "at 8 am",
    "at 8:00 am",
    "at 8 pm",
    "at 8:00 pm",
    "at 20:00",
    "at noon",
    "at midday",
    "at midnight",
    "at half past eight",
    "at quarter past eight",
    "at quarter to nine",
    "at 9:30",
    "at 9:30 am",
    "at 14:00",
    "at 15:30",
    "at 17:00",
    "at midnight",
)

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
# EXCEPTION DEFINITIONS
# =====================================================================

class DatasetGenerationError(Exception):
    """Base exception for all failures occurring within the dataset pipeline."""


class TemplateValidationError(DatasetGenerationError):
    """Raised when a formatting template string is missing mandatory placeholder tokens."""


class ConfigurationParameterError(DatasetGenerationError):
    """Raised when an invalid hyperparameter, ratio, or execution mode is supplied."""


class FilePersistenceError(DatasetGenerationError):
    """Raised when filesystem serialization or file writing fails."""


# =====================================================================
# VALIDATION AND DETERMINISTIC HASHING FUNCTIONS
# =====================================================================

def validate_template_contains_placeholder_tokens(
    template_string: str,
    mandatory_placeholder_tokens: Sequence[str],
) -> None:
    """
    Validate that an input template contains all required curly-brace tokens.

    Parameters:
        template_string: The template string to validate.
        mandatory_placeholder_tokens: Sequence of token identifiers without braces.

    Returns:
        None.

    Raises:
        TemplateValidationError: If any mandatory token is absent.
    """
    missing_tokens: list[str] = [
        f"{{{token}}}" for token in mandatory_placeholder_tokens if f"{{{token}}}" not in template_string
    ]
    if missing_tokens:
        diagnostic_message: str = (
            f"Template validation failed. The template '{template_string}' is missing "
            f"mandatory tokens: {missing_tokens}"
        )
        raise TemplateValidationError(diagnostic_message)


def compute_deterministic_dataset_split_assignment(
    unique_record_content_identifier: str,
    training_split_ratio: float,
    partition_salt_string: str,
) -> str:
    """
    Assign a record to 'train' or 'test' via salted SHA-256 hashing.

    Parameters:
        unique_record_content_identifier: Unique string identifier for the record.
        training_split_ratio: Fraction of records allocated to training (0.0 < ratio < 1.0).
        partition_salt_string: Salt to ensure partition separation.

    Returns:
        'train' or 'test'.

    Raises:
        ConfigurationParameterError: If training_split_ratio is out of range.
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
    Assign a record to a zero-indexed CV fold using salted SHA-256 modulo operations.

    Parameters:
        unique_record_content_identifier: Unique string identifier for the record.
        total_cross_validation_folds_count: Number of folds (>= 2).
        partition_salt_string: Salt for hash isolation.

    Returns:
        Zero-indexed fold identifier integer in [0, total_cross_validation_folds_count - 1].

    Raises:
        ConfigurationParameterError: If folds count is less than 2.
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
            f"Failed to calculate CV fold for identifier '{unique_record_content_identifier}': "
            f"{runtime_exception}\n{detailed_traceback}"
        )
        raise DatasetGenerationError(diagnostic_message) from runtime_exception


def compute_deterministic_record_sort_key(
    record_content_identifier: str,
    balance_salt_string: str,
) -> str:
    """
    Compute a consistent hexadecimal sorting key for balanced sub-sampling.

    Parameters:
        record_content_identifier: Unique content identifier string.
        balance_salt_string: Salt string for sorting hash computation.

    Returns:
        A SHA-256 hexadecimal string representation.
    """
    sort_hash_input: str = f"{balance_salt_string}::subsample_rank::{record_content_identifier}"
    return hashlib.sha256(sort_hash_input.encode(encoding="utf-8")).hexdigest()


# =====================================================================
# RECORD GENERATION FACTORY FUNCTIONS
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
) -> dict[str, object]:
    """
    Build and populate a standardized classification record dictionary.

    Parameters:
        query_text: Rendered natural language query string.
        raw_template_format: Unformatted template string.
        template_complexity_level: Complexity level integer (1 or 2).
        intent_classification_label: Binary target classification label.
        event_name_entity: Extracted event name entity or None.
        time_term_entity: Extracted event time term modifier or None.
        time_expression_entity: Extracted base time expression or None.
        time_modifier_entity: Extracted timezone/precision modifier or None.
        training_split_ratio: Train partition allocation proportion.
        partition_salt_string: Cryptographic hash salt.
        total_cross_validation_folds_count: Total cross-validation folds count.

    Returns:
        A fully structured record dictionary.
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
            "time_term": time_term_entity,
            "time_expression": time_expression_entity,
            "time_modifier": time_modifier_entity,
        },
    }


# =====================================================================
# CLASS-SPECIFIC DATASET GENERATION
# =====================================================================

def generate_known_event_unknown_time_records(
    event_names: Sequence[str],
    time_terms: Sequence[str],
    level_one_templates: Sequence[str],
    level_two_templates: Sequence[str],
    training_split_ratio: float,
    partition_salt_string: str,
    total_cross_validation_folds_count: int,
) -> list[dict[str, object]]:
    """
    Generate all inquiry records for Class A (KNOWN_EVENT_UNKNOWN_TIME).

    Parameters:
        event_names: Sequence of candidate event names.
        time_terms: Sequence of candidate temporal expressions.
        level_one_templates: 1-level templates accepting '{event_name}'.
        level_two_templates: 2-level templates accepting '{event_name}' and '{time_term}'.
        training_split_ratio: Ratio for training split assignment.
        partition_salt_string: Deterministic salt string.
        total_cross_validation_folds_count: Number of CV folds.

    Returns:
        List of generated record dictionaries.
    """
    accumulated_records: list[dict[str, object]] = []

    # Level 1 Generation: {event_name}
    for template_str in sorted(set(level_one_templates)):
        validate_template_contains_placeholder_tokens(template_str, ("event_name",))
        for event_name_str in sorted(set(event_names)):
            rendered_text: str = template_str.format(event_name=event_name_str)
            record: dict[str, object] = construct_inquiry_record_dictionary(
                query_text=rendered_text,
                raw_template_format=template_str,
                template_complexity_level=1,
                intent_classification_label=LABEL_KNOWN_EVENT_UNKNOWN_TIME,
                event_name_entity=event_name_str,
                time_term_entity=None,
                time_expression_entity=None,
                time_modifier_entity=None,
                training_split_ratio=training_split_ratio,
                partition_salt_string=partition_salt_string,
                total_cross_validation_folds_count=total_cross_validation_folds_count,
            )
            accumulated_records.append(record)

    # Level 2 Generation: {event_name} + {time_term}
    for template_str in sorted(set(level_two_templates)):
        validate_template_contains_placeholder_tokens(template_str, ("event_name", "time_term"))
        for event_name_str in sorted(set(event_names)):
            for time_term_str in sorted(set(time_terms)):
                rendered_text = template_str.format(
                    event_name=event_name_str,
                    time_term=time_term_str,
                )
                record = construct_inquiry_record_dictionary(
                    query_text=rendered_text,
                    raw_template_format=template_str,
                    template_complexity_level=2,
                    intent_classification_label=LABEL_KNOWN_EVENT_UNKNOWN_TIME,
                    event_name_entity=event_name_str,
                    time_term_entity=time_term_str,
                    time_expression_entity=None,
                    time_modifier_entity=None,
                    training_split_ratio=training_split_ratio,
                    partition_salt_string=partition_salt_string,
                    total_cross_validation_folds_count=total_cross_validation_folds_count,
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
    Generate all inquiry records for Class B (KNOWN_TIME_UNKNOWN_EVENT).

    Parameters:
        time_expressions: Sequence of candidate base time expressions.
        time_modifiers: Sequence of candidate time modifiers (e.g., timezones).
        level_one_templates: 1-level templates accepting '{time_expression}'.
        level_two_templates: 2-level templates accepting '{time_expression}' and '{time_modifier}'.
        training_split_ratio: Ratio for training split assignment.
        partition_salt_string: Deterministic salt string.
        total_cross_validation_folds_count: Number of CV folds.

    Returns:
        List of generated record dictionaries.
    """
    accumulated_records: list[dict[str, object]] = []

    # Level 1 Generation: {time_expression}
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
            )
            accumulated_records.append(record)

    # Level 2 Generation: {time_expression} + {time_modifier}
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
                )
                accumulated_records.append(record)

    return accumulated_records


def balance_binary_class_records(
    class_a_records: list[dict[str, object]],
    class_b_records: list[dict[str, object]],
    balance_salt_string: str,
) -> list[dict[str, object]]:
    """
    Sub-sample the majority class deterministically to achieve an exact 1:1 balance.

    Parameters:
        class_a_records: Generated records for Class A.
        class_b_records: Generated records for Class B.
        balance_salt_string: Cryptographic salt string for sorting and ranking records.

    Returns:
        A combined and deterministically balanced list of record dictionaries.
    """
    minimum_cardinality: int = min(len(class_a_records), len(class_b_records))

    def _sort_key_extractor(record: dict[str, object]) -> str:
        record_id: str = str(record["sample_unique_identifier"])
        return compute_deterministic_record_sort_key(record_id, balance_salt_string)

    sorted_class_a: list[dict[str, object]] = sorted(class_a_records, key=_sort_key_extractor)
    sorted_class_b: list[dict[str, object]] = sorted(class_b_records, key=_sort_key_extractor)

    balanced_records_list: list[dict[str, object]] = (
        sorted_class_a[:minimum_cardinality] + sorted_class_b[:minimum_cardinality]
    )

    # Sort final combined collection for output determinism
    return sorted(balanced_records_list, key=lambda record: str(record["sample_unique_identifier"]))


# =====================================================================
# PERSISTENCE FUNCTIONS
# =====================================================================

def write_records_to_jsonl(
    records_sequence: Sequence[Mapping[str, object]],
    output_file_path: Path,
) -> int:
    """
    Write a sequence of records to a UTF-8 JSON Lines file.

    Parameters:
        records_sequence: Sequence of dictionary records.
        output_file_path: Filesystem path to target JSONL file.

    Returns:
        Integer count of written lines.

    Raises:
        FilePersistenceError: If file writing or directory creation encounters an error.
    """
    try:
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        persisted_lines_count: int = 0
        with open(output_file_path, mode="w", encoding="utf-8") as file_writer:
            for record in records_sequence:
                line_string: str = json.dumps(record, ensure_ascii=False, sort_keys=True)
                file_writer.write(line_string + "\n")
                persisted_lines_count += 1
        return persisted_lines_count

    except Exception as io_exception:
        traceback_string: str = traceback.format_exc()
        diagnostic_message: str = (
            f"Failed to write JSONL file at '{output_file_path}': {io_exception}\n{traceback_string}"
        )
        raise FilePersistenceError(diagnostic_message) from io_exception


# =====================================================================
# PIPELINE ORCHESTRATION
# =====================================================================

def execute_question_type_dataset_pipeline(
    output_directory_path: Path,
    dataset_organization_mode: str = "split",
    balanced_mode_enabled: bool = True,
    training_split_ratio: float = 0.80,
    total_cross_validation_folds_count: int = 5,
    partition_salt_string: str = "question_type_classification_salt_2026",
) -> dict[str, object]:
    """
    Orchestrate generation, balancing, partitioning, and writing for the 2-class dataset.

    Parameters:
        output_directory_path: Target directory path for file outputs.
        dataset_organization_mode: Either 'split' or 'unified'.
        balanced_mode_enabled: When True, balances both classes to equal cardinality.
        training_split_ratio: Ratio for the training partition.
        total_cross_validation_folds_count: Number of CV folds.
        partition_salt_string: Deterministic salt for hashing.

    Returns:
        Diagnostic dictionary detailing pipeline execution statistics.

    Raises:
        ConfigurationParameterError: If configuration arguments are invalid.
    """
    mode_normalized: str = dataset_organization_mode.strip().lower()
    if mode_normalized not in ("split", "unified"):
        raise ConfigurationParameterError(
            f"Unsupported dataset organization mode: '{dataset_organization_mode}'. Must be 'split' or 'unified'."
        )

    print("[PIPELINE STATUS] Generating Class A: KNOWN_EVENT_UNKNOWN_TIME...")
    class_a_records: list[dict[str, object]] = generate_known_event_unknown_time_records(
        event_names=DEFAULT_EVENT_NAME_COLLECTION,
        time_terms=DEFAULT_EVENT_TIME_TERM_COLLECTION,
        level_one_templates=TEMPLATES_KNOWN_EVENT_LEVEL_ONE,
        level_two_templates=TEMPLATES_KNOWN_EVENT_LEVEL_TWO,
        training_split_ratio=training_split_ratio,
        partition_salt_string=partition_salt_string,
        total_cross_validation_folds_count=total_cross_validation_folds_count,
    )
    print(f"[PIPELINE STATUS] Generated {len(class_a_records)} records for Class A.")

    print("[PIPELINE STATUS] Generating Class B: KNOWN_TIME_UNKNOWN_EVENT...")
    class_b_records: list[dict[str, object]] = generate_known_time_unknown_event_records(
        time_expressions=DEFAULT_TIME_EXPRESSION_COLLECTION,
        time_modifiers=DEFAULT_TIME_MODIFIER_COLLECTION,
        level_one_templates=TEMPLATES_KNOWN_TIME_LEVEL_ONE,
        level_two_templates=TEMPLATES_KNOWN_TIME_LEVEL_TWO,
        training_split_ratio=training_split_ratio,
        partition_salt_string=partition_salt_string,
        total_cross_validation_folds_count=total_cross_validation_folds_count,
    )
    print(f"[PIPELINE STATUS] Generated {len(class_b_records)} records for Class B.")

    final_records: list[dict[str, object]]
    if balanced_mode_enabled:
        print("[PIPELINE STATUS] Balancing classes to identical record cardinality...")
        final_records = balance_binary_class_records(
            class_a_records=class_a_records,
            class_b_records=class_b_records,
            balance_salt_string=partition_salt_string,
        )
        print(f"[PIPELINE STATUS] Balanced dataset total records: {len(final_records)}")
    else:
        print("[PIPELINE STATUS] Retaining full combinatorial imbalance...")
        final_records = sorted(
            class_a_records + class_b_records,
            key=lambda item: str(item["sample_unique_identifier"]),
        )

    execution_summary: dict[str, object] = {
        "execution_mode": mode_normalized,
        "balanced_subsampling_active": balanced_mode_enabled,
        "total_records_generated": len(final_records),
    }

    if mode_normalized == "split":
        train_file_path: Path = output_directory_path / "train_question_type_dataset.jsonl"
        test_file_path: Path = output_directory_path / "test_question_type_dataset.jsonl"

        train_records = [r for r in final_records if r["assigned_dataset_partition"] == "train"]
        test_records = [r for r in final_records if r["assigned_dataset_partition"] == "test"]

        train_persisted: int = write_records_to_jsonl(train_records, train_file_path)
        test_persisted: int = write_records_to_jsonl(test_records, test_file_path)

        train_class_a: int = sum(1 for r in train_records if r["intent_classification_label"] == LABEL_KNOWN_EVENT_UNKNOWN_TIME)
        train_class_b: int = sum(1 for r in train_records if r["intent_classification_label"] == LABEL_KNOWN_TIME_UNKNOWN_EVENT)
        test_class_a: int = sum(1 for r in test_records if r["intent_classification_label"] == LABEL_KNOWN_EVENT_UNKNOWN_TIME)
        test_class_b: int = sum(1 for r in test_records if r["intent_classification_label"] == LABEL_KNOWN_TIME_UNKNOWN_EVENT)

        print(f"[SUCCESS] Wrote Training Set: {train_file_path} (Total: {train_persisted} | Class A: {train_class_a} | Class B: {train_class_b})")
        print(f"[SUCCESS] Wrote Testing Set: {test_file_path} (Total: {test_persisted} | Class A: {test_class_a} | Class B: {test_class_b})")

        execution_summary["train_file_path"] = str(train_file_path)
        execution_summary["train_records_count"] = train_persisted
        execution_summary["test_file_path"] = str(test_file_path)
        execution_summary["test_records_count"] = test_persisted

    elif mode_normalized == "unified":
        unified_file_path: Path = output_directory_path / "unified_question_type_dataset.jsonl"
        unified_persisted: int = write_records_to_jsonl(final_records, unified_file_path)

        total_class_a: int = sum(1 for r in final_records if r["intent_classification_label"] == LABEL_KNOWN_EVENT_UNKNOWN_TIME)
        total_class_b: int = sum(1 for r in final_records if r["intent_classification_label"] == LABEL_KNOWN_TIME_UNKNOWN_EVENT)

        print(f"[SUCCESS] Wrote Unified Set: {unified_file_path} (Total: {unified_persisted} | Class A: {total_class_a} | Class B: {total_class_b})")
        print(f"[PIPELINE STATUS] K-Fold CV Folds Assigned: 0 to {total_cross_validation_folds_count - 1}")

        execution_summary["unified_file_path"] = str(unified_file_path)
        execution_summary["unified_records_count"] = unified_persisted
        execution_summary["cross_validation_folds_count"] = total_cross_validation_folds_count

    return execution_summary


# =====================================================================
# COMMAND-LINE PARSER & ENTRYPOINT
# =====================================================================

def construct_cli_argument_parser() -> argparse.ArgumentParser:
    """
    Construct the CLI argument parser with explicit long and short flags.

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
        help="Organization mode: 'split' outputs separate train/test files; 'unified' outputs a single file with CV folds.",
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
    """Main CLI entrypoint with traceback handling and explicit exit statuses."""
    parser: argparse.ArgumentParser = construct_cli_argument_parser()
    args: argparse.Namespace = parser.parse_args()

    try:
        output_dir: Path = Path(args.output_directory_path)
        execute_question_type_dataset_pipeline(
            output_directory_path=output_dir,
            dataset_organization_mode=args.dataset_organization_mode,
            balanced_mode_enabled=args.balanced_mode_enabled,
            training_split_ratio=args.training_split_ratio,
            total_cross_validation_folds_count=args.cross_validation_folds_count,
            partition_salt_string=args.partition_salt_string,
        )
        sys.exit(0)

    except Exception as fatal_exception:
        traceback_details: str = traceback.format_exc()
        sys.stderr.write(
            f"\n[FATAL PIPELINE FAILURE] Execution terminated unexpectedly:\n"
            f"Exception: {fatal_exception}\n\n"
            f"Execution Traceback:\n{traceback_details}\n"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
