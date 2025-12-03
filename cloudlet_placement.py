"""
Cloudlet Placement in Edge Computing using Genetic Algorithm
Multi-objective optimization: minimize cost and latency while satisfying all constraints
"""

import numpy as np
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import json
import copy


@dataclass
class CandidatePoint:
    """Represents a candidate location where a cloudlet can be placed"""
    id: int
    x: float
    y: float
    placement_cost_multiplier: float = 1.0  # Location-specific cost multiplier


@dataclass
class Cloudlet:
    """Represents a cloudlet with heterogeneous capacities"""
    id: int
    cpu_capacity: float  # GHz
    memory_capacity: float  # GB
    storage_capacity: float  # GB
    coverage_radius: float  # distance units
    base_cost: float  # Base placement cost


@dataclass
class Device:
    """Represents an end-user device with resource demands"""
    id: int
    x: float
    y: float
    cpu_demand: float  # GHz
    memory_demand: float  # GB
    storage_demand: float  # GB


class Solution:
    """Represents a solution to the cloudlet placement problem"""

    def __init__(self, num_points: int, num_cloudlets: int, num_devices: int):
        # cloudlet_placement[i] = j means cloudlet i is placed at point j (-1 if not placed)
        self.cloudlet_placement: List[int] = [-1] * num_cloudlets

        # device_assignment[i] = j means device i is assigned to point j (-1 if not assigned)
        self.device_assignment: List[int] = [-1] * num_devices

        self.total_cost: float = float('inf')
        self.total_latency: float = float('inf')
        self.is_feasible: bool = False
        self.violations: int = 0


class CloudletPlacementProblem:
    """Main problem class with all data and constraint checking"""

    def __init__(self,
                 candidate_points: List[CandidatePoint],
                 cloudlets: List[Cloudlet],
                 devices: List[Device]):
        self.candidate_points = candidate_points
        self.cloudlets = cloudlets
        self.devices = devices

        # Precompute distances
        self.device_to_point_distances = self._compute_device_to_point_distances()

    def _compute_device_to_point_distances(self) -> np.ndarray:
        """Precompute Euclidean distances between all devices and candidate points"""
        distances = np.zeros((len(self.devices), len(self.candidate_points)))
        for i, device in enumerate(self.devices):
            for j, point in enumerate(self.candidate_points):
                distances[i, j] = np.sqrt((device.x - point.x)**2 + (device.y - point.y)**2)
        return distances

    def get_distance(self, device_id: int, point_id: int) -> float:
        """Get precomputed distance between device and point"""
        return self.device_to_point_distances[device_id, point_id]

    def check_coverage(self, device_id: int, cloudlet_id: int, point_id: int) -> bool:
        """Check if device is within coverage radius of cloudlet at given point"""
        distance = self.get_distance(device_id, point_id)
        return distance <= self.cloudlets[cloudlet_id].coverage_radius

    def evaluate_solution(self, solution: Solution) -> Solution:
        """Evaluate solution: calculate cost, latency, and check constraints"""

        # Reset evaluation metrics
        solution.total_cost = 0
        solution.total_latency = 0
        solution.violations = 0
        solution.is_feasible = True

        # Create mapping: point -> cloudlet
        point_to_cloudlet = {}
        for cloudlet_id, point_id in enumerate(solution.cloudlet_placement):
            if point_id != -1:
                if point_id in point_to_cloudlet:
                    # Violation: multiple cloudlets at same point
                    solution.violations += 1
                    solution.is_feasible = False
                point_to_cloudlet[point_id] = cloudlet_id

        # Track resource usage at each point
        resource_usage = {point_id: {'cpu': 0, 'memory': 0, 'storage': 0}
                         for point_id in point_to_cloudlet.keys()}

        # Check device assignments
        assigned_devices = set()
        for device_id, point_id in enumerate(solution.device_assignment):
            if point_id == -1:
                # Violation: device not assigned
                solution.violations += 1
                solution.is_feasible = False
                continue

            if device_id in assigned_devices:
                # Violation: device assigned multiple times
                solution.violations += 1
                solution.is_feasible = False

            assigned_devices.add(device_id)

            # Check if cloudlet is placed at assigned point
            if point_id not in point_to_cloudlet:
                # Violation: device assigned to point without cloudlet
                solution.violations += 1
                solution.is_feasible = False
                continue

            cloudlet_id = point_to_cloudlet[point_id]
            device = self.devices[device_id]

            # Check coverage constraint
            if not self.check_coverage(device_id, cloudlet_id, point_id):
                # Violation: device out of coverage range
                solution.violations += 1
                solution.is_feasible = False

            # Add resource usage
            resource_usage[point_id]['cpu'] += device.cpu_demand
            resource_usage[point_id]['memory'] += device.memory_demand
            resource_usage[point_id]['storage'] += device.storage_demand

            # Add to latency (proportional to distance)
            distance = self.get_distance(device_id, point_id)
            solution.total_latency += distance

        # Check if all devices are assigned
        if len(assigned_devices) < len(self.devices):
            solution.violations += (len(self.devices) - len(assigned_devices))
            solution.is_feasible = False

        # Check capacity constraints and calculate cost
        for point_id, cloudlet_id in point_to_cloudlet.items():
            cloudlet = self.cloudlets[cloudlet_id]
            point = self.candidate_points[point_id]

            # Calculate placement cost
            placement_cost = cloudlet.base_cost * point.placement_cost_multiplier
            solution.total_cost += placement_cost

            # Check capacity constraints
            if resource_usage[point_id]['cpu'] > cloudlet.cpu_capacity:
                solution.violations += 1
                solution.is_feasible = False
            if resource_usage[point_id]['memory'] > cloudlet.memory_capacity:
                solution.violations += 1
                solution.is_feasible = False
            if resource_usage[point_id]['storage'] > cloudlet.storage_capacity:
                solution.violations += 1
                solution.is_feasible = False

        return solution


class GeneticAlgorithmSolver:
    """Genetic Algorithm solver for cloudlet placement problem"""

    def __init__(self,
                 problem: CloudletPlacementProblem,
                 population_size: int = 100,
                 generations: int = 500,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.2,
                 cost_weight: float = 0.5,
                 latency_weight: float = 0.5):

        self.problem = problem
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.cost_weight = cost_weight
        self.latency_weight = latency_weight

        self.best_solution: Optional[Solution] = None
        self.history = []

    def initialize_population(self) -> List[Solution]:
        """Create initial population with random valid solutions"""
        population = []

        for _ in range(self.population_size):
            solution = Solution(
                len(self.problem.candidate_points),
                len(self.problem.cloudlets),
                len(self.problem.devices)
            )

            # Randomly place cloudlets at candidate points
            available_points = list(range(len(self.problem.candidate_points)))
            random.shuffle(available_points)

            for cloudlet_id in range(len(self.problem.cloudlets)):
                if available_points:
                    point_id = available_points.pop()
                    solution.cloudlet_placement[cloudlet_id] = point_id

            # Assign devices to nearest placed cloudlets within coverage
            for device_id in range(len(self.problem.devices)):
                best_point = -1
                best_distance = float('inf')

                for cloudlet_id, point_id in enumerate(solution.cloudlet_placement):
                    if point_id == -1:
                        continue

                    if self.problem.check_coverage(device_id, cloudlet_id, point_id):
                        distance = self.problem.get_distance(device_id, point_id)
                        if distance < best_distance:
                            best_distance = distance
                            best_point = point_id

                solution.device_assignment[device_id] = best_point

            # Evaluate solution
            self.problem.evaluate_solution(solution)
            population.append(solution)

        return population

    def fitness(self, solution: Solution) -> float:
        """Calculate fitness score (lower is better)"""
        # Heavy penalty for infeasible solutions
        if not solution.is_feasible:
            penalty = 1e6 * solution.violations
            return solution.total_cost + solution.total_latency + penalty

        # Normalize and combine objectives
        return (self.cost_weight * solution.total_cost +
                self.latency_weight * solution.total_latency)

    def selection(self, population: List[Solution]) -> Solution:
        """Tournament selection"""
        tournament_size = 5
        tournament = random.sample(population, min(tournament_size, len(population)))
        return min(tournament, key=lambda s: self.fitness(s))

    def crossover(self, parent1: Solution, parent2: Solution) -> Tuple[Solution, Solution]:
        """Two-point crossover for both cloudlet placement and device assignment"""
        if random.random() > self.crossover_rate:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)

        child1 = Solution(
            len(self.problem.candidate_points),
            len(self.problem.cloudlets),
            len(self.problem.devices)
        )
        child2 = Solution(
            len(self.problem.candidate_points),
            len(self.problem.cloudlets),
            len(self.problem.devices)
        )

        # Crossover cloudlet placements
        point = random.randint(1, len(parent1.cloudlet_placement) - 1)
        child1.cloudlet_placement = parent1.cloudlet_placement[:point] + parent2.cloudlet_placement[point:]
        child2.cloudlet_placement = parent2.cloudlet_placement[:point] + parent1.cloudlet_placement[point:]

        # Crossover device assignments
        point = random.randint(1, len(parent1.device_assignment) - 1)
        child1.device_assignment = parent1.device_assignment[:point] + parent2.device_assignment[point:]
        child2.device_assignment = parent2.device_assignment[:point] + parent1.device_assignment[point:]

        return child1, child2

    def mutate(self, solution: Solution):
        """Mutate solution by changing cloudlet placements or device assignments"""
        if random.random() > self.mutation_rate:
            return

        # Mutate cloudlet placement
        if random.random() < 0.5 and len(solution.cloudlet_placement) > 0:
            cloudlet_id = random.randint(0, len(solution.cloudlet_placement) - 1)
            new_point = random.randint(0, len(self.problem.candidate_points) - 1)
            solution.cloudlet_placement[cloudlet_id] = new_point

        # Mutate device assignment
        if len(solution.device_assignment) > 0:
            device_id = random.randint(0, len(solution.device_assignment) - 1)

            # Try to assign to a valid cloudlet
            valid_points = []
            for cloudlet_id, point_id in enumerate(solution.cloudlet_placement):
                if point_id != -1 and self.problem.check_coverage(device_id, cloudlet_id, point_id):
                    valid_points.append(point_id)

            if valid_points:
                solution.device_assignment[device_id] = random.choice(valid_points)

    def repair_solution(self, solution: Solution):
        """Repair infeasible solution"""
        # Remove duplicate cloudlet placements
        used_points = set()
        for cloudlet_id in range(len(solution.cloudlet_placement)):
            point_id = solution.cloudlet_placement[cloudlet_id]
            if point_id in used_points:
                # Find new point
                available = set(range(len(self.problem.candidate_points))) - used_points
                if available:
                    solution.cloudlet_placement[cloudlet_id] = random.choice(list(available))
                else:
                    solution.cloudlet_placement[cloudlet_id] = -1
            if solution.cloudlet_placement[cloudlet_id] != -1:
                used_points.add(solution.cloudlet_placement[cloudlet_id])

        # Reassign devices to valid cloudlets
        for device_id in range(len(solution.device_assignment)):
            current_point = solution.device_assignment[device_id]

            # Check if current assignment is valid
            valid = False
            for cloudlet_id, point_id in enumerate(solution.cloudlet_placement):
                if point_id == current_point:
                    if self.problem.check_coverage(device_id, cloudlet_id, point_id):
                        valid = True
                        break

            if not valid:
                # Find valid assignment
                best_point = -1
                best_distance = float('inf')

                for cloudlet_id, point_id in enumerate(solution.cloudlet_placement):
                    if point_id == -1:
                        continue

                    if self.problem.check_coverage(device_id, cloudlet_id, point_id):
                        distance = self.problem.get_distance(device_id, point_id)
                        if distance < best_distance:
                            best_distance = distance
                            best_point = point_id

                solution.device_assignment[device_id] = best_point

    def solve(self) -> Solution:
        """Run genetic algorithm to find optimal solution"""
        print(f"Initializing population of {self.population_size} solutions...")
        population = self.initialize_population()

        print(f"Running GA for {self.generations} generations...")
        for generation in range(self.generations):
            # Evaluate and track best solution
            for solution in population:
                if self.best_solution is None or self.fitness(solution) < self.fitness(self.best_solution):
                    self.best_solution = copy.deepcopy(solution)

            # Track progress
            avg_fitness = np.mean([self.fitness(s) for s in population])
            feasible_count = sum(1 for s in population if s.is_feasible)

            self.history.append({
                'generation': generation,
                'best_fitness': self.fitness(self.best_solution),
                'avg_fitness': avg_fitness,
                'feasible_count': feasible_count,
                'best_cost': self.best_solution.total_cost,
                'best_latency': self.best_solution.total_latency,
                'is_feasible': self.best_solution.is_feasible
            })

            if generation % 50 == 0:
                print(f"Generation {generation}: Best Fitness={self.fitness(self.best_solution):.2f}, "
                      f"Feasible={feasible_count}/{self.population_size}, "
                      f"Cost={self.best_solution.total_cost:.2f}, "
                      f"Latency={self.best_solution.total_latency:.2f}")

            # Create next generation
            new_population = []

            # Elitism: keep best solutions
            elite_size = int(0.1 * self.population_size)
            elite = sorted(population, key=lambda s: self.fitness(s))[:elite_size]
            new_population.extend([copy.deepcopy(s) for s in elite])

            # Generate offspring
            while len(new_population) < self.population_size:
                parent1 = self.selection(population)
                parent2 = self.selection(population)

                child1, child2 = self.crossover(parent1, parent2)

                self.mutate(child1)
                self.mutate(child2)

                self.repair_solution(child1)
                self.repair_solution(child2)

                self.problem.evaluate_solution(child1)
                self.problem.evaluate_solution(child2)

                new_population.extend([child1, child2])

            population = new_population[:self.population_size]

        print("\nOptimization complete!")
        return self.best_solution
