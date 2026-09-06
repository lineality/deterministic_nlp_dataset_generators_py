"""
Dataset Generation Module for Temporal Event Inquiries.

This module provides a deterministic, functional pipeline for generating synthetic
training and evaluation datasets in JSON Lines (JSONL) format. It systematically
combines event nomenclature and temporal modifiers into natural language question
templates spanning multiple syntactic structures (direct wh-questions, inverted
declarative queries, polite indirect requests, and compact search phrases).

Data partitioning into training and evaluation sets is achieved deterministically
using salted cryptographic hashing (SHA-256) rather than pseudo-random permutations,
ensuring invariant splits across all execution platforms.
"""
import hashlib
import json
import sys
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path

# =====================================================================
# LINGUISTIC TEMPLATE DEFINITIONS
# =====================================================================

# Level-1 Variations: Queries containing exclusively the {event_name} placeholder.
# These encompass diverse communicative registers: direct wh-questions,
# inverted subject-first questions, nominalized inquiries, indirect polite requests,
# and compact search formulations.
LEVEL_ONE_EVENT_NAME_TEMPLATES: tuple[str, ...] = (
    # Direct Wh-Questions (Standard Present and Simple Future)
    "When is {event_name}?",
    "When is {event_name} taking place?",
    "When is {event_name} scheduled?",
    "When is {event_name} happening?",
    "When will {event_name} take place?",
    "When will {event_name} occur?",
    "When will {event_name} happen?",
    "When will {event_name} be held?",
    "When does {event_name} start?",
    "When does {event_name} begin?",
    "When does {event_name} commence?",
    "When does {event_name} kick off?",
    "When does {event_name} get underway?",
    "When will {event_name} start?",
    "When will {event_name} begin?",
    "What time is {event_name}?",
    "What time is {event_name} set for?",
    "What time is {event_name} scheduled for?",
    "What time does {event_name} start?",
    "What time does {event_name} begin?",
    "What time does {event_name} commence?",
    "What time does {event_name} kick off?",
    "What time does {event_name} open?",
    "What time will {event_name} start?",
    "What time will {event_name} begin?",
    "What time will {event_name} be held?",
    "What time will {event_name} take place?",
    "At what time is {event_name}?",
    "At what time does {event_name} start?",
    "At what time does {event_name} begin?",
    "At what time does {event_name} commence?",
    "At what time does {event_name} kick off?",
    "At what time will {event_name} start?",
    "At what time will {event_name} begin?",
    "At what time will {event_name} take place?",
    "At what time is {event_name} scheduled to start?",
    "At what time is {event_name} supposed to begin?",
    # Subject-First / Inverted Declarative Queries
    "{event_name} starts at what time?",
    "{event_name} begins at what time?",
    "{event_name} kicks off at what time?",
    "{event_name} commences at what time?",
    "{event_name} is at what time?",
    "{event_name} takes place at what time?",
    "{event_name} happens at what time?",
    "{event_name} is scheduled for what time?",
    "{event_name} is when?",
    "{event_name} starts when?",
    "{event_name} begins when?",
    "{event_name} kicks off when?",
    "{event_name} takes place when?",
    # Nominal and Schedule Inquiries
    "What is the start time of {event_name}?",
    "What is the start time for {event_name}?",
    "What is the starting time of {event_name}?",
    "What is the starting time for {event_name}?",
    "What is the commencement time of {event_name}?",
    "What is the kickoff time of {event_name}?",
    "What is the kickoff time for {event_name}?",
    "What is the scheduled time for {event_name}?",
    "What is the schedule for {event_name}?",
    "What is the timetable for {event_name}?",
    "What time is listed on the schedule for {event_name}?",
    # Polite / Indirect / Conversational Requests
    "Could you tell me when {event_name} starts?",
    "Could you tell me what time {event_name} begins?",
    "Could you please tell me what time {event_name} starts?",
    "Could you check what time {event_name} starts?",
    "Can you tell me what time {event_name} starts?",
    "Can you tell me when {event_name} takes place?",
    "Can you verify the start time for {event_name}?",
    "Do you know what time {event_name} starts?",
    "Do you know what time {event_name} begins?",
    "Do you know when {event_name} is happening?",
    "Do you know the start time for {event_name}?",
    "Please tell me the start time of {event_name}.",
    "Please tell me what time {event_name} starts.",
    "Please let me know when {event_name} begins.",
    "Please let me know what time {event_name} is scheduled.",
    "I need to know what time {event_name} starts.",
    "I need to know when {event_name} begins.",
    "I need to know when {event_name} is scheduled.",
    "I would like to know what time {event_name} begins.",
    "I'd like to know what time {event_name} starts.",
    "Let me know what time {event_name} starts.",
    # Concise / Search Engine / Telegraphic Formulations
    "{event_name} time",
    "{event_name} start time",
    "{event_name} starting time",
    "{event_name} kickoff time",
    "{event_name} schedule",
    "time of {event_name}",
    "start time of {event_name}",
    "timing for {event_name}",
)

# Level-2 Modular Variations: Queries containing both {event_name} and {time_term}.
LEVEL_TWO_MODULAR_TEMPLATES: tuple[str, ...] = (
    # Postfixed Temporal Modifiers
    "When is {event_name} {time_term}?",
    "When does {event_name} start {time_term}?",
    "When does {event_name} begin {time_term}?",
    "When will {event_name} take place {time_term}?",
    "What time is {event_name} {time_term}?",
    "What time does {event_name} start {time_term}?",
    "What time does {event_name} begin {time_term}?",
    "What time does {event_name} kick off {time_term}?",
    "What time will {event_name} take place {time_term}?",
    "At what time does {event_name} start {time_term}?",
    "At what time is {event_name} scheduled {time_term}?",
    "{event_name} starts at what time {time_term}?",
    "{event_name} begins at what time {time_term}?",
    "{event_name} is at what time {time_term}?",
    "{event_name} takes place at what time {time_term}?",
    "{event_name} is scheduled for what time {time_term}?",
    "What is the start time of {event_name} {time_term}?",
    "What is the scheduled time for {event_name} {time_term}?",
    "Could you tell me what time {event_name} starts {time_term}?",
    "Can you tell me what time {event_name} begins {time_term}?",
    "Do you know what time {event_name} starts {time_term}?",
    "Please tell me what time {event_name} starts {time_term}.",
    # Prefixed Temporal Modifiers
    "{time_term}, what time is {event_name}?",
    "{time_term}, when does {event_name} start?",
    "{time_term}, at what time does {event_name} begin?",
    "{time_term}, what is the start time for {event_name}?",
    "{time_term}, when is {event_name} scheduled to take place?",
    "{time_term}, could you tell me what time {event_name} starts?",
    # Scoped / Prepositional Context
    "For {time_term}, what time does {event_name} start?",
    "For {time_term}, what is the scheduled start time of {event_name}?",
    "Looking at {time_term}, when is {event_name} taking place?",
)

# Standardized baseline data samples for verification and generation
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
    "the show",
    "breakfast",
    "the movie",
    "dinner",
    "the ball",
    "the dance",
    "the comedy act",
    "the game",
    "the play",
    "the reading",
    "the book signing",
    "the tennis match",
    "the race",
    "the fundraiser",
    "the hay ride",
    "the auction",
    "the concert",
    "the dog show",
    "thing singing contest",
    "the tallent show",
    "The Birds",
    "the murder mystery dinner",

)

DEFAULT_TIME_TERM_COLLECTION: tuple[str, ...] = (
    "this morning",
    "this afternoon",
    "this evening",
    "tonight",
    "today",
    "tomorrow",
    "tomorrow morning",
    "tomorrow afternoon",
    "tomorrow evening",
    "tomorrow night",
    "later today",
    "this Friday",
    "next Monday",
)


# =====================================================================
# EXCEPTION DEFINITIONS
# =====================================================================

class DatasetGenerationError(Exception):
    """Base exception for all errors raised during dataset generation operations."""


class TemplateValidationError(DatasetGenerationError):
    """Raised when a template string lacks required placeholder formatting tokens."""


class ConfigurationParameterError(DatasetGenerationError):
    """Raised when invalid configuration parameters or ratios are provided."""


class FilePersistenceError(DatasetGenerationError):
    """Raised when filesystem write or directory creation operations fail."""


# =====================================================================
# VALIDATION AND FORMATTING FUNCTIONS
# =====================================================================

def validate_template_contains_placeholder_tokens(
    template_string: str,
    mandatory_placeholder_token_names: Sequence[str],
) -> None:
    """
    Validate that a formatting template contains all required placeholder tokens.

    Parameters:
        template_string: The string template to inspect for curly-brace tokens.
        mandatory_placeholder_token_names: A sequence of expected token names
            (e.g., ['event_name', 'time_term']) that must appear inside the template.

    Returns:
        None.

    Raises:
        TemplateValidationError: If any mandatory placeholder token is absent from the template.
    """
    missing_placeholder_token_list: list[str] = []

    for expected_token_name in mandatory_placeholder_token_names:
        formatting_token_marker: str = f"{{{expected_token_name}}}"
        if formatting_token_marker not in template_string:
            missing_placeholder_token_list.append(formatting_token_marker)

    if missing_placeholder_token_list:
        error_diagnostic_message: str = (
            f"Template validation failed. The template string '{template_string}' is missing "
            f"the required placeholder tokens: {missing_placeholder_token_list}."
        )
        raise TemplateValidationError(error_diagnostic_message)


def compute_deterministic_dataset_split_assignment(
    unique_record_content_identifier: str,
    training_split_ratio_floating_point: float,
    partition_salt_string: str = "event_time_dataset_salt_v1",
) -> str:
    """
    Assign a sample record deterministically to either 'train' or 'test' using SHA-256 hashing.

    The combination of the partition salt and the record identifier is hashed. The first
    8 bytes of the digest are converted into an unsigned integer, normalized to a float
    in the range [0.0, 1.0), and compared against the training split ratio threshold.

    Parameters:
        unique_record_content_identifier: A string uniquely identifying the generated sample
            (such as the fully rendered query text).
        training_split_ratio_floating_point: The fraction of records allocated to training
            (must be strictly between 0.0 and 1.0).
        partition_salt_string: A cryptographic salt string ensuring partition separation.

    Returns:
        A string literal: either 'train' or 'test'.

    Raises:
        ConfigurationParameterError: If training_split_ratio_floating_point is not in (0.0, 1.0).
    """
    if not (0.0 < training_split_ratio_floating_point < 1.0):
        error_message: str = (
            f"Invalid training_split_ratio_floating_point: {training_split_ratio_floating_point}. "
            "The ratio must be strictly greater than 0.0 and strictly less than 1.0."
        )
        raise ConfigurationParameterError(error_message)

    try:
        hash_input_string: str = f"{partition_salt_string}::{unique_record_content_identifier}"
        digest_bytes: bytes = hashlib.sha256(hash_input_string.encode(encoding="utf-8")).digest()

        # Extract the first 8 bytes as an unsigned 64-bit big-endian integer
        extracted_integer_value: int = int.from_bytes(digest_bytes[:8], byteorder="big", signed=False)
        maximum_possible_integer_value: int = 0xFFFFFFFFFFFFFFFF  # 2^64 - 1
        normalized_ratio_value: float = extracted_integer_value / maximum_possible_integer_value

        if normalized_ratio_value < training_split_ratio_floating_point:
            return "train"
        return "test"

    except Exception as runtime_exception:
        detailed_traceback: str = traceback.format_exc()
        diagnostic_message: str = (
            f"Encountered unexpected failure calculating partition split for identifier "
            f"'{unique_record_content_identifier}': {runtime_exception}\n{detailed_traceback}"
        )
        raise DatasetGenerationError(diagnostic_message) from runtime_exception


# =====================================================================
# DATASET RECORD GENERATION FUNCTIONS
# =====================================================================

def construct_level_one_record_dictionary(
    event_name_string: str,
    template_format_string: str,
    training_split_ratio_floating_point: float,
    partition_salt_string: str,
    target_intent_label_string: str,
) -> dict[str, object]:
    """
    Render a single Level-1 (event name only) inquiry record.

    Parameters:
        event_name_string: The target event name (e.g., 'the team standup').
        template_format_string: Template containing the '{event_name}' placeholder.
        training_split_ratio_floating_point: Proportion for the training dataset partition.
        partition_salt_string: Salt for deterministic hashing.
        target_intent_label_string: The classification label assigned to this inquiry.

    Returns:
        A dictionary adhering to the standardized JSONL schema.
    """
    validate_template_contains_placeholder_tokens(
        template_string=template_format_string,
        mandatory_placeholder_token_names=("event_name",),
    )

    rendered_query_text: str = template_format_string.format(event_name=event_name_string)
    assigned_dataset_split: str = compute_deterministic_dataset_split_assignment(
        unique_record_content_identifier=rendered_query_text,
        training_split_ratio_floating_point=training_split_ratio_floating_point,
        partition_salt_string=partition_salt_string,
    )

    sha256_content_hash: str = hashlib.sha256(rendered_query_text.encode(encoding="utf-8")).hexdigest()

    return {
        "sample_unique_identifier": sha256_content_hash,
        "query_text": rendered_query_text,
        "event_name_entity": event_name_string,
        "time_term_entity": None,
        "template_complexity_level": 1,
        "raw_template_format": template_format_string,
        "intent_classification_label": target_intent_label_string,
        "assigned_dataset_partition": assigned_dataset_split,
    }


def construct_level_two_modular_record_dictionary(
    event_name_string: str,
    time_term_string: str,
    template_format_string: str,
    training_split_ratio_floating_point: float,
    partition_salt_string: str,
    target_intent_label_string: str,
) -> dict[str, object]:
    """
    Render a single Level-2 (event name and time term) modular inquiry record.

    Parameters:
        event_name_string: The target event name (e.g., 'the quarterly review').
        time_term_string: The temporal term (e.g., 'tomorrow morning').
        template_format_string: Template containing '{event_name}' and '{time_term}'.
        training_split_ratio_floating_point: Proportion for the training dataset partition.
        partition_salt_string: Salt for deterministic hashing.
        target_intent_label_string: The classification label assigned to this inquiry.

    Returns:
        A dictionary adhering to the standardized JSONL schema.
    """
    validate_template_contains_placeholder_tokens(
        template_string=template_format_string,
        mandatory_placeholder_token_names=("event_name", "time_term"),
    )

    rendered_query_text: str = template_format_string.format(
        event_name=event_name_string,
        time_term=time_term_string,
    )
    assigned_dataset_split: str = compute_deterministic_dataset_split_assignment(
        unique_record_content_identifier=rendered_query_text,
        training_split_ratio_floating_point=training_split_ratio_floating_point,
        partition_salt_string=partition_salt_string,
    )

    sha256_content_hash: str = hashlib.sha256(rendered_query_text.encode(encoding="utf-8")).hexdigest()

    return {
        "sample_unique_identifier": sha256_content_hash,
        "query_text": rendered_query_text,
        "event_name_entity": event_name_string,
        "time_term_entity": time_term_string,
        "template_complexity_level": 2,
        "raw_template_format": template_format_string,
        "intent_classification_label": target_intent_label_string,
        "assigned_dataset_partition": assigned_dataset_split,
    }


def generate_all_dataset_records(
    event_name_sequence: Sequence[str],
    time_term_sequence: Sequence[str],
    level_one_template_sequence: Sequence[str],
    level_two_modular_template_sequence: Sequence[str],
    training_split_ratio_floating_point: float,
    partition_salt_string: str,
    target_intent_label_string: str = "query_event_start_time",
) -> list[dict[str, object]]:
    """
    Generate all Level-1 and Level-2 records across the cartesian product of inputs.

    Inputs are sorted prior to iteration to guarantee identical processing order
    regardless of input collection variation.

    Parameters:
        event_name_sequence: Sequence of candidate event names.
        time_term_sequence: Sequence of candidate temporal expressions.
        level_one_template_sequence: Sequence of 1-level templates.
        level_two_modular_template_sequence: Sequence of 2-level modular templates.
        training_split_ratio_floating_point: Fraction of records assigned to 'train'.
        partition_salt_string: Deterministic partition salt string.
        target_intent_label_string: Intent label applied across all generated items.

    Returns:
        A list of generated record dictionaries.

    Raises:
        ConfigurationParameterError: If any input sequence is empty.
    """
    if not event_name_sequence:
        raise ConfigurationParameterError("The event_name_sequence parameter must not be empty.")
    if not time_term_sequence:
        raise ConfigurationParameterError("The time_term_sequence parameter must not be empty.")
    if not level_one_template_sequence and not level_two_modular_template_sequence:
        raise ConfigurationParameterError("At least one template sequence must be provided.")

    # Canonical sorting ensures input order independence across executions
    sorted_event_names: list[str] = sorted(set(event_name_sequence))
    sorted_time_terms: list[str] = sorted(set(time_term_sequence))
    sorted_level_one_templates: list[str] = sorted(set(level_one_template_sequence))
    sorted_level_two_templates: list[str] = sorted(set(level_two_modular_template_sequence))

    accumulated_generated_records: list[dict[str, object]] = []

    # Step 1: Render Level-1 variations
    for template_format_string in sorted_level_one_templates:
        for event_name_string in sorted_event_names:
            level_one_record: dict[str, object] = construct_level_one_record_dictionary(
                event_name_string=event_name_string,
                template_format_string=template_format_string,
                training_split_ratio_floating_point=training_split_ratio_floating_point,
                partition_salt_string=partition_salt_string,
                target_intent_label_string=target_intent_label_string,
            )
            accumulated_generated_records.append(level_one_record)

    # Step 2: Render Level-2 modular variations
    for template_format_string in sorted_level_two_templates:
        for event_name_string in sorted_event_names:
            for time_term_string in sorted_time_terms:
                level_two_record: dict[str, object] = construct_level_two_modular_record_dictionary(
                    event_name_string=event_name_string,
                    time_term_string=time_term_string,
                    template_format_string=template_format_string,
                    training_split_ratio_floating_point=training_split_ratio_floating_point,
                    partition_salt_string=partition_salt_string,
                    target_intent_label_string=target_intent_label_string,
                )
                accumulated_generated_records.append(level_two_record)

    return accumulated_generated_records


# =====================================================================
# PERSISTENCE / FILE SYSTEM WRITER FUNCTIONS
# =====================================================================

def write_records_to_jsonl_file_path(
    record_dictionary_sequence: Sequence[Mapping[str, object]],
    target_destination_file_path: Path,
) -> int:
    """
    Write a sequence of record dictionaries to a JSON Lines file in UTF-8 encoding.

    Parent directories are created if they do not exist. Operations are wrapped in
    comprehensive error handling with traceback capture.

    Parameters:
        record_dictionary_sequence: Sequence of dictionary mappings to persist.
        target_destination_file_path: Filesystem path of the destination JSONL file.

    Returns:
        The total count of persisted JSON lines.

    Raises:
        FilePersistenceError: If filesystem access or writing fails.
    """
    try:
        # Guarantee parent directory presence
        target_destination_file_path.parent.mkdir(parents=True, exist_ok=True)

        lines_persisted_counter: int = 0
        with open(target_destination_file_path, mode="w", encoding="utf-8") as file_stream_writer:
            for single_record_dictionary in record_dictionary_sequence:
                serialized_json_line_string: str = json.dumps(
                    single_record_dictionary,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                file_stream_writer.write(serialized_json_line_string + "\n")
                lines_persisted_counter += 1

        return lines_persisted_counter

    except Exception as runtime_io_error:
        detailed_traceback_string: str = traceback.format_exc()
        diagnostic_message: str = (
            f"Failed to persist {len(record_dictionary_sequence)} records into destination file "
            f"'{target_destination_file_path}': {runtime_io_error}\n{detailed_traceback_string}"
        )
        raise FilePersistenceError(diagnostic_message) from runtime_io_error


def partition_and_save_jsonl_datasets(
    all_generated_records_sequence: Sequence[dict[str, object]],
    output_training_jsonl_file_path: Path,
    output_evaluation_jsonl_file_path: Path,
) -> tuple[int, int]:
    """
    Separate dataset records by their assigned split and write them to distinct files.

    Parameters:
        all_generated_records_sequence: Full sequence of generated records.
        output_training_jsonl_file_path: Destination path for the training JSONL file.
        output_evaluation_jsonl_file_path: Destination path for the evaluation JSONL file.

    Returns:
        A tuple of integers: (training_records_count, evaluation_records_count).
    """
    training_records_collection: list[dict[str, object]] = []
    evaluation_records_collection: list[dict[str, object]] = []

    for individual_record in all_generated_records_sequence:
        assigned_partition_label: object = individual_record.get("assigned_dataset_partition")
        if assigned_partition_label == "train":
            training_records_collection.append(individual_record)
        elif assigned_partition_label == "test":
            evaluation_records_collection.append(individual_record)
        else:
            invalid_partition_message: str = (
                f"Record with identifier '{individual_record.get('sample_unique_identifier')}' "
                f"contains unrecognized partition label '{assigned_partition_label}'."
            )
            raise DatasetGenerationError(invalid_partition_message)

    persisted_train_count: int = write_records_to_jsonl_file_path(
        record_dictionary_sequence=training_records_collection,
        target_destination_file_path=output_training_jsonl_file_path,
    )
    persisted_test_count: int = write_records_to_jsonl_file_path(
        record_dictionary_sequence=evaluation_records_collection,
        target_destination_file_path=output_evaluation_jsonl_file_path,
    )

    return (persisted_train_count, persisted_test_count)


# =====================================================================
# PIPELINE ORCHESTRATION FUNCTION
# =====================================================================

def execute_deterministic_dataset_generation_pipeline(
    output_directory_path: Path,
    training_split_ratio_floating_point: float = 0.80,
    partition_salt_string: str = "production_seed_salt_value_2026",
    candidate_event_names: Sequence[str] = DEFAULT_EVENT_NAME_COLLECTION,
    candidate_time_terms: Sequence[str] = DEFAULT_TIME_TERM_COLLECTION,
    level_one_templates: Sequence[str] = LEVEL_ONE_EVENT_NAME_TEMPLATES,
    level_two_modular_templates: Sequence[str] = LEVEL_TWO_MODULAR_TEMPLATES,
) -> tuple[Path, Path, int, int]:
    """
    Orchestrate the end-to-end generation and persistence of the train and test JSONL files.

    Parameters:
        output_directory_path: Base directory where output JSONL files will be created.
        training_split_ratio_floating_point: Ratio allocated to training (default: 0.80).
        partition_salt_string: Deterministic salt for hash-based dataset splitting.
        candidate_event_names: Sequence of event names to format into queries.
        candidate_time_terms: Sequence of temporal terms to format into queries.
        level_one_templates: Sequence of 1-level templates.
        level_two_modular_templates: Sequence of 2-level modular templates.

    Returns:
        A 4-tuple consisting of:
            (training_jsonl_path, evaluation_jsonl_path, train_count, test_count)
    """
    print("[PIPELINE NOTICE] Starting deterministic dataset generation...")
    print(f"[PIPELINE NOTICE] Level-1 Templates Count: {len(level_one_templates)}")
    print(f"[PIPELINE NOTICE] Level-2 Templates Count: {len(level_two_modular_templates)}")
    print(f"[PIPELINE NOTICE] Event Names Count: {len(candidate_event_names)}")
    print(f"[PIPELINE NOTICE] Time Terms Count: {len(candidate_time_terms)}")
    print(f"[PIPELINE NOTICE] Configured Training Split Ratio: {training_split_ratio_floating_point:.2%}")
    print(f"[PIPELINE NOTICE] Cryptographic Partition Salt: '{partition_salt_string}'")

    generated_records: list[dict[str, object]] = generate_all_dataset_records(
        event_name_sequence=candidate_event_names,
        time_term_sequence=candidate_time_terms,
        level_one_template_sequence=level_one_templates,
        level_two_modular_template_sequence=level_two_modular_templates,
        training_split_ratio_floating_point=training_split_ratio_floating_point,
        partition_salt_string=partition_salt_string,
    )

    print(f"[PIPELINE NOTICE] Total Candidate Records Generated: {len(generated_records)}")

    destination_train_jsonl_file_path: Path = output_directory_path / "train_event_time_queries.jsonl"
    destination_test_jsonl_file_path: Path = output_directory_path / "test_event_time_queries.jsonl"

    train_count, test_count = partition_and_save_jsonl_datasets(
        all_generated_records_sequence=generated_records,
        output_training_jsonl_file_path=destination_train_jsonl_file_path,
        output_evaluation_jsonl_file_path=destination_test_jsonl_file_path,
    )

    observed_train_ratio: float = (train_count / len(generated_records)) if generated_records else 0.0
    print(f"[PIPELINE SUCCESS] Successfully persisted training dataset: {destination_train_jsonl_file_path}")
    print(f"[PIPELINE SUCCESS] Training Record Count: {train_count} ({observed_train_ratio:.2%})")
    print(f"[PIPELINE SUCCESS] Successfully persisted test dataset: {destination_test_jsonl_file_path}")
    print(f"[PIPELINE SUCCESS] Evaluation Record Count: {test_count} ({1.0 - observed_train_ratio:.2%})")

    return (
        destination_train_jsonl_file_path,
        destination_test_jsonl_file_path,
        train_count,
        test_count,
    )


# =====================================================================
# SCRIPT ENTRYPOINT
# =====================================================================

if __name__ == "__main__":
    try:
        # Default destination directory relative to script invocation path
        destination_output_directory: Path = Path("./generated_dataset_output")

        execute_deterministic_dataset_generation_pipeline(
            output_directory_path=destination_output_directory,
            training_split_ratio_floating_point=0.80,
            partition_salt_string="production_dataset_partition_salt_v2026",
        )
        sys.exit(0)

    except Exception as unhandled_execution_error:
        formatted_error_traceback: str = traceback.format_exc()
        sys.stderr.write(
            f"\n[FATAL PIPELINE ERROR] An unhandled exception terminated the generation process:\n"
            f"{unhandled_execution_error}\n\n"
            f"Detailed Execution Traceback:\n{formatted_error_traceback}\n"
        )
        sys.exit(1)
