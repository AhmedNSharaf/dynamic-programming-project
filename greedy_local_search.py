"""
Greedy Heuristic + Local Search Algorithm for Cloudlet Placement
Fast constructive approach followed by iterative improvement
"""

import numpy as np
import random
import copy
from typing import List, Tuple, Optional
from cloudlet_placement import CloudletPlacementProblem, Solution


class GreedyLocalSearchSolver:
    """
    Greedy Heuristic + Local Search solver for cloudlet placement problem

    Phase 1: Greedy Construction
    - Place cloudlets based on coverage/cost efficiency
    - Assign devices to best available cloudlets

    Phase 2: Local Search
    - Swap cloudlet positions
    - Reassign devices
    - Replace cloudlets with better alternatives
    - Continue until no improvement
    """

    def __init__(self,
                 problem: CloudletPlacementProblem,
                 cost_weight: float = 0.5,
                 latency_weight: float = 0.5,
                 max_iterations: int = 1000,
                 no_improvement_limit: int = 100):

        self.problem = problem
        self.cost_weight = cost_weight
        self.latency_weight = latency_weight
        self.max_iterations = max_iterations
        self.no_improvement_limit = no_improvement_limit

        self.best_solution: Optional[Solution] = None
        self.history = []

    def fitness(self, solution: Solution) -> float:
        """Calculate fitness score (lower is better)"""
        if not solution.is_feasible:
            penalty = 1e6 * solution.violations
            return solution.total_cost + solution.total_latency + penalty

        return (self.cost_weight * solution.total_cost +
                self.latency_weight * solution.total_latency)

    def greedy_construction(self) -> Solution:
        """
        Phase 1: Greedy construction of initial solution

        Strategy:
        1. Calculate device density around each candidate point
        2. Sort cloudlets by capacity/cost ratio
        3. Place cloudlets at high-demand locations
        4. Assign devices to nearest cloudlet within coverage
        """
        print("Phase 1: Greedy Construction")

        solution = Solution(
            len(self.problem.candidate_points),
            len(self.problem.cloudlets),
            len(self.problem.devices)
        )

        # Step 1: Calculate demand coverage for each point-cloudlet combination
        point_scores = []

        for point_id, point in enumerate(self.problem.candidate_points):
            for cloudlet_id, cloudlet in enumerate(self.problem.cloudlets):
                # Count devices within coverage
                devices_covered = []
                total_demand = {'cpu': 0, 'memory': 0, 'storage': 0}

                for device_id, device in enumerate(self.problem.devices):
                    if self.problem.check_coverage(device_id, cloudlet_id, point_id):
                        devices_covered.append(device_id)
                        total_demand['cpu'] += device.cpu_demand
                        total_demand['memory'] += device.memory_demand
                        total_demand['storage'] += device.storage_demand

                # Check if cloudlet can handle the demand
                feasible = (
                    total_demand['cpu'] <= cloudlet.cpu_capacity and
                    total_demand['memory'] <= cloudlet.memory_capacity and
                    total_demand['storage'] <= cloudlet.storage_capacity
                )

                # Calculate score: coverage benefit vs. cost
                coverage_benefit = len(devices_covered)
                placement_cost = cloudlet.base_cost * point.placement_cost_multiplier

                # Efficiency score (devices covered per unit cost)
                if placement_cost > 0:
                    efficiency = coverage_benefit / placement_cost
                else:
                    efficiency = coverage_benefit

                point_scores.append({
                    'point_id': point_id,
                    'cloudlet_id': cloudlet_id,
                    'devices_covered': set(devices_covered),
                    'efficiency': efficiency,
                    'coverage_benefit': coverage_benefit,
                    'cost': placement_cost,
                    'feasible': feasible
                })

        # Step 2: Greedy placement - select best placements iteratively
        used_points = set()
        used_cloudlets = set()
        covered_devices = set()

        # Sort by efficiency (descending)
        point_scores.sort(key=lambda x: x['efficiency'], reverse=True)

        for score in point_scores:
            point_id = score['point_id']
            cloudlet_id = score['cloudlet_id']

            # Skip if point or cloudlet already used
            if point_id in used_points or cloudlet_id in used_cloudlets:
                continue

            # Calculate uncovered devices this placement would add
            new_devices = score['devices_covered'] - covered_devices

            if len(new_devices) > 0:  # Only place if it covers new devices
                solution.cloudlet_placement[cloudlet_id] = point_id
                used_points.add(point_id)
                used_cloudlets.add(cloudlet_id)
                covered_devices.update(new_devices)

        print(f"  Placed {len(used_cloudlets)} cloudlets covering {len(covered_devices)} devices initially")

        # Step 3: If not all devices covered, place additional cloudlets
        uncovered = set(range(len(self.problem.devices))) - covered_devices

        if uncovered:
            print(f"  Attempting to cover {len(uncovered)} remaining devices...")

            # Try to place remaining cloudlets to cover uncovered devices
            for cloudlet_id in range(len(self.problem.cloudlets)):
                if cloudlet_id in used_cloudlets or len(uncovered) == 0:
                    continue

                best_point = None
                best_coverage = 0

                for point_id in range(len(self.problem.candidate_points)):
                    if point_id in used_points:
                        continue

                    # Count uncovered devices within range
                    coverage_count = sum(
                        1 for device_id in uncovered
                        if self.problem.check_coverage(device_id, cloudlet_id, point_id)
                    )

                    if coverage_count > best_coverage:
                        best_coverage = coverage_count
                        best_point = point_id

                if best_point is not None and best_coverage > 0:
                    solution.cloudlet_placement[cloudlet_id] = best_point
                    used_points.add(best_point)
                    used_cloudlets.add(cloudlet_id)

                    # Update uncovered devices
                    newly_covered = [
                        d for d in uncovered
                        if self.problem.check_coverage(d, cloudlet_id, best_point)
                    ]
                    uncovered -= set(newly_covered)

        # Step 4: Assign devices to cloudlets
        self._assign_devices(solution)

        # Evaluate solution
        self.problem.evaluate_solution(solution)

        print(f"  Initial solution: Cost=${solution.total_cost:.2f}, "
              f"Latency={solution.total_latency:.2f}, "
              f"Feasible={solution.is_feasible}, "
              f"Coverage={sum(1 for d in solution.device_assignment if d != -1)}/{len(self.problem.devices)}")

        return solution

    def _assign_devices(self, solution: Solution):
        """Assign each device to the best available cloudlet"""
        # Create mapping: point -> cloudlet
        point_to_cloudlet = {}
        for cloudlet_id, point_id in enumerate(solution.cloudlet_placement):
            if point_id != -1:
                point_to_cloudlet[point_id] = cloudlet_id

        # Assign each device to nearest cloudlet within coverage
        for device_id in range(len(self.problem.devices)):
            best_point = -1
            best_score = float('inf')

            for point_id, cloudlet_id in point_to_cloudlet.items():
                # Check if device is in coverage
                if self.problem.check_coverage(device_id, cloudlet_id, point_id):
                    # Calculate score: distance (lower is better)
                    distance = self.problem.get_distance(device_id, point_id)

                    if distance < best_score:
                        best_score = distance
                        best_point = point_id

            solution.device_assignment[device_id] = best_point

    def _reassign_devices_optimally(self, solution: Solution):
        """Reassign devices to minimize latency while respecting capacity"""
        # Create mapping: point -> cloudlet
        point_to_cloudlet = {}
        for cloudlet_id, point_id in enumerate(solution.cloudlet_placement):
            if point_id != -1:
                point_to_cloudlet[point_id] = cloudlet_id

        # Track resource usage at each point
        resource_usage = {
            point_id: {'cpu': 0, 'memory': 0, 'storage': 0}
            for point_id in point_to_cloudlet.keys()
        }

        # Sort devices by total demand (assign high-demand devices first)
        device_priorities = []
        for device_id, device in enumerate(self.problem.devices):
            total_demand = device.cpu_demand + device.memory_demand + device.storage_demand
            device_priorities.append((device_id, total_demand))

        device_priorities.sort(key=lambda x: x[1], reverse=True)

        # Reset assignments
        solution.device_assignment = [-1] * len(self.problem.devices)

        # Assign devices
        for device_id, _ in device_priorities:
            device = self.problem.devices[device_id]

            # Find best cloudlet within coverage that has capacity
            best_point = -1
            best_distance = float('inf')

            for point_id, cloudlet_id in point_to_cloudlet.items():
                cloudlet = self.problem.cloudlets[cloudlet_id]

                # Check coverage
                if not self.problem.check_coverage(device_id, cloudlet_id, point_id):
                    continue

                # Check capacity
                if (resource_usage[point_id]['cpu'] + device.cpu_demand > cloudlet.cpu_capacity or
                    resource_usage[point_id]['memory'] + device.memory_demand > cloudlet.memory_capacity or
                    resource_usage[point_id]['storage'] + device.storage_demand > cloudlet.storage_capacity):
                    continue

                # Calculate distance
                distance = self.problem.get_distance(device_id, point_id)

                if distance < best_distance:
                    best_distance = distance
                    best_point = point_id

            if best_point != -1:
                solution.device_assignment[device_id] = best_point
                resource_usage[best_point]['cpu'] += device.cpu_demand
                resource_usage[best_point]['memory'] += device.memory_demand
                resource_usage[best_point]['storage'] += device.storage_demand

    def local_search(self, initial_solution: Solution) -> Solution:
        """
        Phase 2: Local Search improvement

        Neighborhood operators:
        1. Swap two cloudlet positions
        2. Move cloudlet to different location
        3. Replace cloudlet with different type
        4. Reassign devices to different cloudlets

        Strategy: First-improvement hill climbing with restarts
        """
        print("\nPhase 2: Local Search Improvement")

        current_solution = copy.deepcopy(initial_solution)
        best_solution = copy.deepcopy(initial_solution)
        best_fitness = self.fitness(best_solution)

        no_improvement_count = 0
        iteration = 0

        while iteration < self.max_iterations and no_improvement_count < self.no_improvement_limit:
            iteration += 1
            improved = False

            # Operator 1: Swap cloudlet positions
            if not improved:
                improved = self._try_swap_positions(current_solution, best_solution, best_fitness)
                if improved:
                    best_fitness = self.fitness(best_solution)

            # Operator 2: Move cloudlet to better location
            if not improved:
                improved = self._try_move_cloudlet(current_solution, best_solution, best_fitness)
                if improved:
                    best_fitness = self.fitness(best_solution)

            # Operator 3: Replace cloudlet with different type
            if not improved:
                improved = self._try_replace_cloudlet(current_solution, best_solution, best_fitness)
                if improved:
                    best_fitness = self.fitness(best_solution)

            # Operator 4: Reassign devices
            if not improved:
                improved = self._try_reassign_devices(current_solution, best_solution, best_fitness)
                if improved:
                    best_fitness = self.fitness(best_solution)

            if improved:
                current_solution = copy.deepcopy(best_solution)
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # Progress reporting
            if iteration % 100 == 0:
                print(f"  Iteration {iteration}: Best Fitness={best_fitness:.2f}, "
                      f"Cost=${best_solution.total_cost:.2f}, "
                      f"Latency={best_solution.total_latency:.2f}, "
                      f"Feasible={best_solution.is_feasible}")

        print(f"\nLocal search completed after {iteration} iterations")
        print(f"  Final: Cost=${best_solution.total_cost:.2f}, "
              f"Latency={best_solution.total_latency:.2f}, "
              f"Feasible={best_solution.is_feasible}")

        return best_solution

    def _try_swap_positions(self, current: Solution, best: Solution, best_fitness: float) -> bool:
        """Try swapping positions of two placed cloudlets"""
        placed_cloudlets = [i for i, p in enumerate(current.cloudlet_placement) if p != -1]

        if len(placed_cloudlets) < 2:
            return False

        for i in range(len(placed_cloudlets)):
            for j in range(i + 1, len(placed_cloudlets)):
                c1, c2 = placed_cloudlets[i], placed_cloudlets[j]

                # Swap positions
                temp_solution = copy.deepcopy(current)
                temp_solution.cloudlet_placement[c1], temp_solution.cloudlet_placement[c2] = \
                    temp_solution.cloudlet_placement[c2], temp_solution.cloudlet_placement[c1]

                # Reassign devices
                self._reassign_devices_optimally(temp_solution)
                self.problem.evaluate_solution(temp_solution)

                # Check if better
                if self.fitness(temp_solution) < best_fitness:
                    best.cloudlet_placement = temp_solution.cloudlet_placement[:]
                    best.device_assignment = temp_solution.device_assignment[:]
                    self.problem.evaluate_solution(best)
                    return True

        return False

    def _try_move_cloudlet(self, current: Solution, best: Solution, best_fitness: float) -> bool:
        """Try moving a cloudlet to a different candidate point"""
        placed_cloudlets = [i for i, p in enumerate(current.cloudlet_placement) if p != -1]
        used_points = set(p for p in current.cloudlet_placement if p != -1)

        for cloudlet_id in placed_cloudlets:
            current_point = current.cloudlet_placement[cloudlet_id]

            # Try all other points
            for new_point in range(len(self.problem.candidate_points)):
                if new_point in used_points:
                    continue

                # Move cloudlet
                temp_solution = copy.deepcopy(current)
                temp_solution.cloudlet_placement[cloudlet_id] = new_point

                # Reassign devices
                self._reassign_devices_optimally(temp_solution)
                self.problem.evaluate_solution(temp_solution)

                # Check if better
                if self.fitness(temp_solution) < best_fitness:
                    best.cloudlet_placement = temp_solution.cloudlet_placement[:]
                    best.device_assignment = temp_solution.device_assignment[:]
                    self.problem.evaluate_solution(best)
                    return True

        return False

    def _try_replace_cloudlet(self, current: Solution, best: Solution, best_fitness: float) -> bool:
        """Try replacing a placed cloudlet with an unplaced one"""
        placed_cloudlets = [i for i, p in enumerate(current.cloudlet_placement) if p != -1]
        unplaced_cloudlets = [i for i, p in enumerate(current.cloudlet_placement) if p == -1]

        if not unplaced_cloudlets:
            return False

        for placed_id in placed_cloudlets:
            point_id = current.cloudlet_placement[placed_id]

            for unplaced_id in unplaced_cloudlets:
                # Replace cloudlet
                temp_solution = copy.deepcopy(current)
                temp_solution.cloudlet_placement[placed_id] = -1
                temp_solution.cloudlet_placement[unplaced_id] = point_id

                # Reassign devices
                self._reassign_devices_optimally(temp_solution)
                self.problem.evaluate_solution(temp_solution)

                # Check if better
                if self.fitness(temp_solution) < best_fitness:
                    best.cloudlet_placement = temp_solution.cloudlet_placement[:]
                    best.device_assignment = temp_solution.device_assignment[:]
                    self.problem.evaluate_solution(best)
                    return True

        return False

    def _try_reassign_devices(self, current: Solution, best: Solution, best_fitness: float) -> bool:
        """Try reassigning devices to reduce latency while maintaining feasibility"""
        temp_solution = copy.deepcopy(current)

        # Reassign with optimal strategy
        self._reassign_devices_optimally(temp_solution)
        self.problem.evaluate_solution(temp_solution)

        # Check if better
        if self.fitness(temp_solution) < best_fitness:
            best.cloudlet_placement = temp_solution.cloudlet_placement[:]
            best.device_assignment = temp_solution.device_assignment[:]
            self.problem.evaluate_solution(best)
            return True

        return False

    def solve(self) -> Solution:
        """Run complete Greedy + Local Search algorithm"""
        print("="*80)
        print("GREEDY HEURISTIC + LOCAL SEARCH ALGORITHM")
        print("="*80)

        # Phase 1: Greedy construction
        initial_solution = self.greedy_construction()

        self.history.append({
            'iteration': 0,
            'phase': 'greedy',
            'fitness': self.fitness(initial_solution),
            'cost': initial_solution.total_cost,
            'latency': initial_solution.total_latency,
            'is_feasible': initial_solution.is_feasible
        })

        # Phase 2: Local search
        final_solution = self.local_search(initial_solution)

        self.history.append({
            'iteration': 1,
            'phase': 'local_search',
            'fitness': self.fitness(final_solution),
            'cost': final_solution.total_cost,
            'latency': final_solution.total_latency,
            'is_feasible': final_solution.is_feasible
        })

        self.best_solution = final_solution

        print("="*80)
        return final_solution
