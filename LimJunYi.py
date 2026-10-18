from dataclasses import dataclass
from functools import lru_cache
import gc
from math import ceil
from statistics import fmean, median, stdev
from time import perf_counter
import tracemalloc
from typing import Dict, Iterator, List, Optional, Set, Tuple


@dataclass(frozen=True)
class State:
    """Missionaries/cannibals on the left bank and the boat position."""

    missionaries_left: int
    cannibals_left: int
    boat_side: str


@dataclass(frozen=True)
class Action:
    """Passengers and direction for one legal boat crossing."""

    missionaries: int
    cannibals: int
    direction: str


@dataclass(frozen=True)
class TestCase:
    """Complete configuration for one search experiment."""

    number: int
    total_missionaries: int
    total_cannibals: int
    boat_capacity: int
    initial_state: State
    goal_state: State


@dataclass
class SearchCounters:
    """Mutable counters collected while Hill Climbing examines moves."""

    moves_attempted: int = 0
    unsafe_moves_rejected: int = 0
    unavailable_passengers_rejected: int = 0
    duplicate_states_rejected: int = 0
    states_enqueued: int = 0


@dataclass
class HillClimbingResult:
    """A Hill Climbing outcome and its search counters."""

    solution_found: bool
    path: List[State]
    actions: List[Action]
    states_expanded: int
    states_generated: int
    unique_states_visited: int
    peak_frontier_size: int
    moves_attempted: int
    unsafe_moves_rejected: int
    unavailable_passengers_rejected: int
    duplicate_states_rejected: int
    states_enqueued: int
    reason: str = ""

    @property
    def solution_depth(self) -> Optional[int]:
        """Return the number of crossings, or None when no solution exists."""

        return len(self.actions) if self.solution_found else None


@dataclass
class BenchmarkResult:
    """Hill Climbing result plus measurements collected outside printing."""

    test_case: TestCase
    search: HillClimbingResult
    average_time_ms: float
    median_time_ms: float
    standard_deviation_ms: float
    minimum_time_ms: float
    maximum_time_ms: float
    peak_memory_bytes: int
    timing_repeats: int
    timing_batches: int
    memory_repeats: int


# The rows below come from Data test (1).xlsx. Every row describes the same
# total population (5M and 5C) but a different start state. The workbook does
# not include boat capacity, so capacity 3 is used: it is the smallest capacity
# that makes all 10 revised cases solvable under the bank-and-boat safety rule.
DEFAULT_GOAL = State(0, 0, "R")

TEST_CASES: List[TestCase] = [
    TestCase(1, 5, 5, 3, State(5, 4, "R"), DEFAULT_GOAL),
    TestCase(2, 5, 5, 3, State(5, 5, "L"), DEFAULT_GOAL),
    TestCase(3, 5, 5, 3, State(4, 4, "L"), DEFAULT_GOAL),
    TestCase(4, 5, 5, 3, State(5, 3, "R"), DEFAULT_GOAL),
    TestCase(5, 5, 5, 3, State(5, 2, "R"), DEFAULT_GOAL),
    TestCase(6, 5, 5, 3, State(4, 4, "R"), DEFAULT_GOAL),
    TestCase(7, 5, 5, 3, State(3, 3, "R"), DEFAULT_GOAL),
    TestCase(8, 5, 5, 3, State(5, 4, "L"), DEFAULT_GOAL),
    TestCase(9, 5, 5, 3, State(5, 3, "L"), DEFAULT_GOAL),
    TestCase(10, 5, 5, 3, State(5, 1, "L"), DEFAULT_GOAL),
]

# Repeating very short searches makes the average time more stable.  Memory is
# measured separately because tracemalloc would otherwise distort timing.
INITIAL_TIMING_REPEATS = 100
TIMING_BATCHES = 5
MINIMUM_BATCH_SECONDS = 0.01
MAXIMUM_TIMING_REPEATS = 1_000_000
MEMORY_REPEATS = 5
WARMUP_REPEATS = 5


def is_safe_group(missionaries: int, cannibals: int) -> bool:
    """Return True when no missionaries in one location can be eaten."""

    return missionaries == 0 or missionaries >= cannibals


def is_valid_state(
    state: State, total_missionaries: int, total_cannibals: int
) -> bool:
    """Return True only when counts are in range and both banks are safe."""

    missionaries_left = state.missionaries_left
    cannibals_left = state.cannibals_left
    if state.boat_side not in ("L", "R"):
        return False
    if not 0 <= missionaries_left <= total_missionaries:
        return False
    if not 0 <= cannibals_left <= total_cannibals:
        return False

    missionaries_right = total_missionaries - missionaries_left
    cannibals_right = total_cannibals - cannibals_left

    # Cannibals may outnumber missionaries only when no missionary is present.
    left_is_safe = is_safe_group(missionaries_left, cannibals_left)
    right_is_safe = is_safe_group(missionaries_right, cannibals_right)
    return left_is_safe and right_is_safe


def validation_error(test_case: TestCase) -> str:
    """Return an explanation for an invalid test case, or an empty string."""

    if test_case.total_missionaries < 0 or test_case.total_cannibals < 0:
        return "Population totals cannot be negative."
    if test_case.boat_capacity < 1:
        return "Boat capacity must be at least 1."
    if not is_valid_state(
        test_case.initial_state,
        test_case.total_missionaries,
        test_case.total_cannibals,
    ):
        return "The initial state is outside the totals or is unsafe."
    if not is_valid_state(
        test_case.goal_state,
        test_case.total_missionaries,
        test_case.total_cannibals,
    ):
        return "The goal state is outside the totals or is unsafe."
    return ""


def is_goal(state: State, goal_state: State) -> bool:
    """Return True when the current state matches this case's goal state."""

    return state == goal_state


@lru_cache(maxsize=None)
def passenger_options(boat_capacity: int) -> Tuple[Tuple[int, int], ...]:
    """Precompute all nonempty passenger combinations for one capacity."""

    options: List[Tuple[int, int]] = []
    for passenger_count in range(1, boat_capacity + 1):
        for missionaries in range(passenger_count + 1):
            options.append((missionaries, passenger_count - missionaries))
    return tuple(options)


def generate_successors(
    state: State,
    total_missionaries: int,
    total_cannibals: int,
    boat_capacity: int,
    counters: Optional[SearchCounters] = None,
) -> Iterator[Tuple[State, Action]]:
    """Yield every legal one-way boat move from state.

    Passenger combinations are generated in a deterministic order. Each boat
    carries at least one person and no more than boat_capacity people. The
    passengers in the boat must also satisfy the missionary-safety rule.
    """

    missionaries_left = state.missionaries_left
    cannibals_left = state.cannibals_left

    # A boat cannot carry more people than exist in the complete problem.
    # Capping here prevents enormous, useless passenger-option caches when a
    # user supplies an unusually large nominal capacity.
    effective_capacity = min(
        boat_capacity, total_missionaries + total_cannibals
    )
    for missionaries_in_boat, cannibals_in_boat in passenger_options(
        effective_capacity
    ):
        if counters is not None:
            counters.moves_attempted += 1

        if state.boat_side == "L":
            # The left bank must contain everyone selected for the boat.
            if (
                missionaries_in_boat > missionaries_left
                or cannibals_in_boat > cannibals_left
            ):
                if counters is not None:
                    counters.unavailable_passengers_rejected += 1
                continue
            next_state = State(
                missionaries_left - missionaries_in_boat,
                cannibals_left - cannibals_in_boat,
                "R",
            )
            direction = "L -> R"
        else:
            missionaries_right = total_missionaries - missionaries_left
            cannibals_right = total_cannibals - cannibals_left
            if (
                missionaries_in_boat > missionaries_right
                or cannibals_in_boat > cannibals_right
            ):
                if counters is not None:
                    counters.unavailable_passengers_rejected += 1
                continue
            next_state = State(
                missionaries_left + missionaries_in_boat,
                cannibals_left + cannibals_in_boat,
                "L",
            )
            direction = "R -> L"

        # The boat is a separate location during the crossing. If missionaries
        # are aboard, they must not be outnumbered by cannibals.
        if not is_safe_group(missionaries_in_boat, cannibals_in_boat):
            if counters is not None:
                counters.unsafe_moves_rejected += 1
            continue

        if not is_valid_state(next_state, total_missionaries, total_cannibals):
            if counters is not None:
                counters.unsafe_moves_rejected += 1
            continue

        action = Action(missionaries_in_boat, cannibals_in_boat, direction)
        yield next_state, action


def reconstruct_path(
    goal_state: State,
    parent: Dict[State, Optional[State]],
    parent_action: Dict[State, Action],
) -> Tuple[List[State], List[Action]]:
    """Follow parent links from the goal back to the start, then reverse them."""

    reversed_path: List[State] = []
    reversed_actions: List[Action] = []
    current: Optional[State] = goal_state

    while current is not None:
        reversed_path.append(current)
        previous = parent[current]
        if previous is not None:
            reversed_actions.append(parent_action[current])
        current = previous

    reversed_path.reverse()
    reversed_actions.reverse()
    return reversed_path, reversed_actions


def heuristic(state: State, test_case: TestCase) -> int:
    """Estimate remaining work from only the current state.

    The score counts people on the wrong bank and adds a penalty when the boat
    is already on the goal side while people still need to be collected.
    """

    people_out_of_place = (
        abs(state.missionaries_left - test_case.goal_state.missionaries_left)
        + abs(state.cannibals_left - test_case.goal_state.cannibals_left)
    )
    boat_penalty = int(
        people_out_of_place > 0
        and state.boat_side == test_case.goal_state.boat_side
    )
    return people_out_of_place + boat_penalty


def hill_climbing_search(test_case: TestCase) -> HillClimbingResult:
    """Run deterministic steepest-descent Hill Climbing.

    Only a strictly improving neighbour is accepted. There is no backtracking,
    restart, or exhaustive fallback, so a local optimum is a valid failure.
    """

    error = validation_error(test_case)
    if error:
        return HillClimbingResult(
            solution_found=False,
            path=[],
            actions=[],
            states_expanded=0,
            states_generated=0,
            unique_states_visited=0,
            peak_frontier_size=0,
            moves_attempted=0,
            unsafe_moves_rejected=0,
            unavailable_passengers_rejected=0,
            duplicate_states_rejected=0,
            states_enqueued=0,
            reason=error,
        )

    current = test_case.initial_state
    path: List[State] = [current]
    actions: List[Action] = []
    evaluated_states: Set[State] = {current}
    counters = SearchCounters()
    states_expanded = 0
    states_generated = 0
    peak_frontier_size = 1
    reason = "Goal reached."

    while not is_goal(current, test_case.goal_state):
        states_expanded += 1
        neighbours = list(
            generate_successors(
                current,
                test_case.total_missionaries,
                test_case.total_cannibals,
                test_case.boat_capacity,
                counters,
            )
        )
        states_generated += len(neighbours)
        peak_frontier_size = max(peak_frontier_size, len(neighbours))

        for next_state, _action in neighbours:
            if next_state in evaluated_states:
                counters.duplicate_states_rejected += 1
            evaluated_states.add(next_state)

        if not neighbours:
            reason = "No valid neighbouring state is available."
            break

        best_state, best_action = min(
            neighbours,
            key=lambda item: (
                heuristic(item[0], test_case),
                item[0].missionaries_left,
                item[0].cannibals_left,
                item[0].boat_side,
            ),
        )
        if heuristic(best_state, test_case) >= heuristic(current, test_case):
            reason = "Local optimum reached; no neighbour improves the heuristic."
            break

        current = best_state
        path.append(current)
        actions.append(best_action)
        counters.states_enqueued += 1

    solution_found = is_goal(current, test_case.goal_state)
    return HillClimbingResult(
        solution_found=solution_found,
        path=path,
        actions=actions,
        states_expanded=states_expanded,
        states_generated=states_generated,
        unique_states_visited=len(evaluated_states),
        peak_frontier_size=peak_frontier_size,
        moves_attempted=counters.moves_attempted,
        unsafe_moves_rejected=counters.unsafe_moves_rejected,
        unavailable_passengers_rejected=counters.unavailable_passengers_rejected,
        duplicate_states_rejected=counters.duplicate_states_rejected,
        states_enqueued=counters.states_enqueued,
        reason=reason,
    )


def verify_solution(test_case: TestCase, result: HillClimbingResult) -> None:
    """Raise ValueError if a returned solution path is internally inconsistent."""

    if not result.solution_found:
        return

    if not result.path:
        raise ValueError("A solved result must contain a path.")
    if result.path[0] != test_case.initial_state:
        raise ValueError("Solution path does not begin at the initial state.")
    if not is_goal(result.path[-1], test_case.goal_state):
        raise ValueError("Solution path does not end at the configured goal state.")
    if len(result.path) != len(result.actions) + 1:
        raise ValueError("Solution path and action counts are inconsistent.")

    for old_state, new_state, action in zip(
        result.path, result.path[1:], result.actions
    ):
        legal_moves = dict(
            generate_successors(
                old_state,
                test_case.total_missionaries,
                test_case.total_cannibals,
                test_case.boat_capacity,
            )
        )
        if new_state not in legal_moves:
            raise ValueError("Solution path contains an illegal state transition.")
        if legal_moves[new_state] != action:
            raise ValueError("A recorded action does not match its state transition.")


def calibrated_repeat_count(test_case: TestCase, initial_repeats: int) -> int:
    """Increase repetitions until a calibration batch lasts long enough."""

    if initial_repeats < 1:
        raise ValueError("timing_repeats must be at least 1.")

    repeats = initial_repeats
    while True:
        gc.collect()
        start_time = perf_counter()
        for _ in range(repeats):
            hill_climbing_search(test_case)
        elapsed_seconds = perf_counter() - start_time

        if (
            elapsed_seconds >= MINIMUM_BATCH_SECONDS
            or repeats >= MAXIMUM_TIMING_REPEATS
        ):
            return repeats

        duration = max(elapsed_seconds, 1e-12)
        growth_factor = max(2, ceil(MINIMUM_BATCH_SECONDS / duration))
        repeats = min(MAXIMUM_TIMING_REPEATS, repeats * growth_factor)


def benchmark_test_case(
    test_case: TestCase,
    timing_repeats: int = INITIAL_TIMING_REPEATS,
    timing_batches: int = TIMING_BATCHES,
    memory_repeats: int = MEMORY_REPEATS,
) -> BenchmarkResult:
    """Measure Hill Climbing without including console-output time."""

    if timing_batches < 2:
        raise ValueError("timing_batches must be at least 2 for sample statistics.")
    if memory_repeats < 1:
        raise ValueError("memory_repeats must be at least 1.")

    canonical_result = hill_climbing_search(test_case)
    verify_solution(test_case, canonical_result)

    # Warm-up reduces one-time interpreter effects before timing begins.
    for _ in range(WARMUP_REPEATS):
        hill_climbing_search(test_case)

    actual_repeats = calibrated_repeat_count(test_case, timing_repeats)
    timing_samples: List[float] = []
    for _ in range(timing_batches):
        gc.collect()
        start_time = perf_counter()
        for _ in range(actual_repeats):
            hill_climbing_search(test_case)
        elapsed_seconds = perf_counter() - start_time
        timing_samples.append(elapsed_seconds * 1000.0 / actual_repeats)

    average_time_ms = fmean(timing_samples)
    median_time_ms = median(timing_samples)
    standard_deviation_ms = stdev(timing_samples)

    # Measure memory in separate trials so tracemalloc overhead is not included
    # in average_time_ms.  Report the largest observed peak for this test case.
    memory_peaks: List[int] = []
    for _ in range(memory_repeats):
        # Include passenger-option allocation in every reported memory trial.
        passenger_options.cache_clear()
        gc.collect()
        tracemalloc.start()
        hill_climbing_search(test_case)
        _current_bytes, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory_peaks.append(peak_bytes)

    return BenchmarkResult(
        test_case=test_case,
        search=canonical_result,
        average_time_ms=average_time_ms,
        median_time_ms=median_time_ms,
        standard_deviation_ms=standard_deviation_ms,
        minimum_time_ms=min(timing_samples),
        maximum_time_ms=max(timing_samples),
        peak_memory_bytes=max(memory_peaks),
        timing_repeats=actual_repeats,
        timing_batches=timing_batches,
        memory_repeats=memory_repeats,
    )


def compact_state(state: State) -> str:
    """Return the tuple notation used in the supplied test-data workbook."""

    return (
        f"({state.missionaries_left},"
        f"{state.cannibals_left},{state.boat_side})"
    )


def descriptive_state(state: State, test_case: TestCase) -> str:
    """Show the complete population and boat location on both banks."""

    missionaries_left = state.missionaries_left
    cannibals_left = state.cannibals_left
    missionaries_right = test_case.total_missionaries - missionaries_left
    cannibals_right = test_case.total_cannibals - cannibals_left
    boat = "LEFT" if state.boat_side == "L" else "RIGHT"
    return (
        f"Left[{missionaries_left}M,{cannibals_left}C]  "
        f"Right[{missionaries_right}M,{cannibals_right}C]  Boat={boat}"
    )


def format_action(action: Action) -> str:
    """Return a readable description of one boat crossing."""

    return (
        f"move {action.missionaries}M and {action.cannibals}C "
        f"{action.direction}"
    )


def format_solution_cost(solution_depth: Optional[int]) -> str:
    """Format the crossing count and Hill Climbing optimality limitation."""

    if solution_depth is None:
        return "N/A - failed (optimality not guaranteed)"
    unit = "step" if solution_depth == 1 else "steps"
    return f"{solution_depth} {unit} (optimality not guaranteed)"


def format_memory_size(byte_count: int) -> str:
    """Format an exact byte count as decimal KB with bytes shown."""

    return f"{byte_count / 1000.0:.3f} KB ({byte_count:,} bytes)"


def print_case_evaluation(result: BenchmarkResult) -> None:
    """Print the four required measured evaluation results for one dataset."""

    search = result.search
    completeness = (
        "Successful solve" if search.solution_found else "Failed to solve"
    )
    cost = format_solution_cost(search.solution_depth)
    print("Evaluation criteria for this dataset:")
    print_ascii_table(
        ["Evaluation criterion", "Result"],
        [
            ["Completeness (Hill Climbing result)", completeness],
            ["Cost-Optimality", cost],
            [
                "Time Complexity",
                f"{result.average_time_ms:.6f} ms mean execution time",
            ],
            [
                "Space Complexity",
                f"{format_memory_size(result.peak_memory_bytes)} peak memory usage",
            ],
        ],
    )


def print_test_case(result: BenchmarkResult) -> None:
    """Display the required details and complete path for one test case."""

    case = result.test_case
    search = result.search
    print("\n" + "=" * 78)
    print(f"TEST CASE {case.number}")
    print("=" * 78)
    print(
        "Initial configuration: "
        f"total={case.total_missionaries}M/{case.total_cannibals}C, "
        f"capacity={case.boat_capacity}, start={compact_state(case.initial_state)}, "
        f"goal={compact_state(case.goal_state)}"
    )
    print(f"Solution found: {'YES' if search.solution_found else 'NO'}")

    heading = (
        "Complete solution path:"
        if search.solution_found
        else "Attempted path before failure:"
    )
    print(heading)
    path_rows: List[List[str]] = []
    for step, state in enumerate(search.path):
        missionaries_right = case.total_missionaries - state.missionaries_left
        cannibals_right = case.total_cannibals - state.cannibals_left
        action = (
            "Initial state"
            if step == 0
            else format_action(search.actions[step - 1])
        )
        path_rows.append(
            [
                str(step),
                f"{state.missionaries_left}M, {state.cannibals_left}C",
                f"{missionaries_right}M, {cannibals_right}C",
                "LEFT" if state.boat_side == "L" else "RIGHT",
                action,
            ]
        )
    print_ascii_table(
        ["Step", "Left bank", "Right bank", "Boat", "Action"],
        path_rows,
    )

    if search.solution_found:
        print(f"Number of boat crossings: {search.solution_depth}")
    else:
        print(f"Number of boat crossings before failure: {len(search.actions)}")
        print(f"Failure reason: {search.reason}")

    print_case_evaluation(result)


def print_ascii_table(headers: List[str], rows: List[List[str]]) -> None:
    """Print a dependency-free summary table."""

    widths = [
        max(len(headers[column]), *(len(row[column]) for row in rows))
        for column in range(len(headers))
    ]
    separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"

    def print_row(values: List[str]) -> None:
        cells = [value.ljust(widths[index]) for index, value in enumerate(values)]
        print("| " + " | ".join(cells) + " |")

    print(separator)
    print_row(headers)
    print(separator)
    for row in rows:
        print_row(row)
    print(separator)


def print_summary(
    results: List[BenchmarkResult], title: str, show_averages: bool
) -> None:
    """Summarise every dataset using the four evaluation criteria."""

    summary_rows: List[List[str]] = []
    for result in results:
        case = result.test_case
        search = result.search
        summary_rows.append(
            [
                str(case.number),
                compact_state(case.initial_state),
                "Successful solve" if search.solution_found else "Failed to solve",
                format_solution_cost(search.solution_depth),
                f"{result.average_time_ms:.6f} ms",
                format_memory_size(result.peak_memory_bytes),
            ]
        )

    if show_averages and results:
        solved_depths = [
            result.search.solution_depth
            for result in results
            if result.search.solution_depth is not None
        ]
        average_cost = (
            f"{fmean(solved_depths):.2f} steps "
            "(optimality not guaranteed)"
            if solved_depths
            else "N/A - failures (optimality not guaranteed)"
        )
        successful = sum(result.search.solution_found for result in results)
        summary_rows.append(
            [
                "AVERAGE",
                "-",
                f"{successful}/{len(results)} successful",
                average_cost,
                f"{fmean(result.average_time_ms for result in results):.6f} ms",
                format_memory_size(
                    round(fmean(result.peak_memory_bytes for result in results))
                ),
            ]
        )

    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    print_ascii_table(
        [
            "Dataset",
            "Start state",
            "Completeness (Hill Climbing result)",
            "Cost-Optimality",
            "Time Complexity",
            "Space Complexity",
        ],
        summary_rows,
    )


def main() -> None:
    """Benchmark and print only the 10 required datasets."""

    required_results = [benchmark_test_case(case) for case in TEST_CASES]

    for result in required_results:
        print_test_case(result)
    print_summary(
        required_results,
        "SUMMARY OF THE 10 REQUIRED TEST CASES",
        show_averages=True,
    )


if __name__ == "__main__":
    main()
