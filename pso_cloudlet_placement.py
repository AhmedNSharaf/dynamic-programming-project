"""
Particle Swarm Optimization (PSO) Algorithm for Cloudlet Placement
Multi-objective optimization: minimize cost and latency while satisfying all constraints
"""

import numpy as np
import random
import copy
from typing import List, Tuple, Optional
from cloudlet_placement import CloudletPlacementProblem, Solution


class Particle:
    """Represents a particle in PSO with position and velocity"""
    
    def __init__(self, num_points: int, num_cloudlets: int, num_devices: int):
        self.solution = Solution(num_points, num_cloudlets, num_devices)
        
        # Velocity for cloudlet placement (continuous values)
        self.velocity_placement = np.zeros(num_cloudlets)
        
        # Velocity for device assignment (continuous values)
        self.velocity_assignment = np.zeros(num_devices)
        
        # Personal best
        self.best_solution = None
        self.best_fitness = float('inf')


class PSOSolver:
    """
    Particle Swarm Optimization solver for cloudlet placement problem
    
    PSO Components:
    - Particles: Candidate solutions with position and velocity
    - Global Best: Best solution found by the swarm
    - Personal Best: Best solution found by each particle
    - Velocity Update: Based on inertia, cognitive, and social components
    """
    
    def __init__(self,
                 problem: CloudletPlacementProblem,
                 swarm_size: int = 50,
                 max_iterations: int = 500,
                 w: float = 0.7,           # Inertia weight
                 c1: float = 1.5,          # Cognitive coefficient
                 c2: float = 1.5,          # Social coefficient
                 cost_weight: float = 0.5,
                 latency_weight: float = 0.5):
        
        self.problem = problem
        self.swarm_size = swarm_size
        self.max_iterations = max_iterations
        self.w = w  # Inertia weight (exploration vs exploitation)
        self.c1 = c1  # Cognitive component (personal best influence)
        self.c2 = c2  # Social component (global best influence)
        self.cost_weight = cost_weight
        self.latency_weight = latency_weight
        
        # Global best
        self.global_best_solution: Optional[Solution] = None
        self.global_best_fitness: float = float('inf')
        
        self.history = []
    
    def fitness(self, solution: Solution) -> float:
        """Calculate fitness score (lower is better)"""
        if not solution.is_feasible:
            penalty = 1e6 * solution.violations
            return solution.total_cost + solution.total_latency + penalty
        
        return (self.cost_weight * solution.total_cost +
                self.latency_weight * solution.total_latency)
    
    def initialize_particle(self) -> Particle:
        """Initialize a single particle with random position"""
        particle = Particle(
            len(self.problem.candidate_points),
            len(self.problem.cloudlets),
            len(self.problem.devices)
        )
        
        # Random cloudlet placement
        available_points = list(range(len(self.problem.candidate_points)))
        random.shuffle(available_points)
        
        for cloudlet_id in range(len(self.problem.cloudlets)):
            if available_points:
                point_id = available_points.pop()
                particle.solution.cloudlet_placement[cloudlet_id] = point_id
        
        # Assign devices to nearest placed cloudlets within coverage
        self._assign_devices_to_cloudlets(particle.solution)
        
        # Initialize velocities randomly
        particle.velocity_placement = np.random.uniform(-1, 1, len(self.problem.cloudlets))
        particle.velocity_assignment = np.random.uniform(-1, 1, len(self.problem.devices))
        
        # Evaluate solution
        self.problem.evaluate_solution(particle.solution)
        
        # Set personal best
        particle.best_solution = copy.deepcopy(particle.solution)
        particle.best_fitness = self.fitness(particle.solution)
        
        return particle
    
    def initialize_swarm(self) -> List[Particle]:
        """Initialize swarm with random particles"""
        print(f"Initializing swarm of {self.swarm_size} particles...")
        
        swarm = []
        for i in range(self.swarm_size):
            particle = self.initialize_particle()
            swarm.append(particle)
            
            # Update global best
            if self.fitness(particle.solution) < self.global_best_fitness:
                self.global_best_solution = copy.deepcopy(particle.solution)
                self.global_best_fitness = self.fitness(particle.solution)
        
        print(f"  Initial global best fitness: {self.global_best_fitness:.2f}")
        return swarm
    
    def _assign_devices_to_cloudlets(self, solution: Solution):
        """Assign each device to nearest cloudlet within coverage"""
        # Create mapping: point -> cloudlet
        point_to_cloudlet = {}
        for cloudlet_id, point_id in enumerate(solution.cloudlet_placement):
            if point_id != -1:
                point_to_cloudlet[point_id] = cloudlet_id
        
        # Assign devices
        for device_id in range(len(self.problem.devices)):
            best_point = -1
            best_distance = float('inf')
            
            for point_id, cloudlet_id in point_to_cloudlet.items():
                if self.problem.check_coverage(device_id, cloudlet_id, point_id):
                    distance = self.problem.get_distance(device_id, point_id)
                    if distance < best_distance:
                        best_distance = distance
                        best_point = point_id
            
            solution.device_assignment[device_id] = best_point
    
    def update_velocity_and_position(self, particle: Particle):
        """Update particle velocity and position using PSO equations"""
        
        # Update velocity for cloudlet placement
        for cloudlet_id in range(len(self.problem.cloudlets)):
            # Inertia component
            inertia = self.w * particle.velocity_placement[cloudlet_id]
            
            # Cognitive component (personal best)
            r1 = random.random()
            current_pos = particle.solution.cloudlet_placement[cloudlet_id]
            personal_best_pos = particle.best_solution.cloudlet_placement[cloudlet_id]
            cognitive = self.c1 * r1 * (personal_best_pos - current_pos)
            
            # Social component (global best)
            r2 = random.random()
            global_best_pos = self.global_best_solution.cloudlet_placement[cloudlet_id]
            social = self.c2 * r2 * (global_best_pos - current_pos)
            
            # Update velocity
            particle.velocity_placement[cloudlet_id] = inertia + cognitive + social
            
            # Clamp velocity
            max_velocity = len(self.problem.candidate_points) * 0.2
            particle.velocity_placement[cloudlet_id] = np.clip(
                particle.velocity_placement[cloudlet_id],
                -max_velocity, max_velocity
            )
            
            # Update position (discrete: round to nearest valid point)
            new_position = current_pos + particle.velocity_placement[cloudlet_id]
            new_position = int(np.round(new_position))
            new_position = np.clip(new_position, 0, len(self.problem.candidate_points) - 1)
            
            particle.solution.cloudlet_placement[cloudlet_id] = new_position
        
        # Fix duplicate placements
        self._fix_duplicate_placements(particle.solution)
        
        # Update device assignments based on new cloudlet positions
        self._assign_devices_to_cloudlets(particle.solution)
        
        # Optionally: apply local refinement for device assignments
        self._refine_device_assignments(particle)
    
    def _fix_duplicate_placements(self, solution: Solution):
        """Fix duplicate cloudlet placements by reassigning to unused points"""
        used_points = {}
        
        for cloudlet_id, point_id in enumerate(solution.cloudlet_placement):
            if point_id == -1:
                continue
            
            if point_id in used_points:
                # Duplicate found, reassign to unused point
                used = set(p for p in solution.cloudlet_placement if p != -1)
                available = set(range(len(self.problem.candidate_points))) - used
                
                if available:
                    solution.cloudlet_placement[cloudlet_id] = random.choice(list(available))
                else:
                    solution.cloudlet_placement[cloudlet_id] = -1
            else:
                used_points[point_id] = cloudlet_id
    
    def _refine_device_assignments(self, particle: Particle):
        """Refine device assignments using velocity information"""
        for device_id in range(len(self.problem.devices)):
            # Inertia component
            inertia = self.w * particle.velocity_assignment[device_id]
            
            # Cognitive component
            r1 = random.random()
            current_pos = particle.solution.device_assignment[device_id]
            personal_best_pos = particle.best_solution.device_assignment[device_id]
            cognitive = self.c1 * r1 * (personal_best_pos - current_pos)
            
            # Social component
            r2 = random.random()
            global_best_pos = self.global_best_solution.device_assignment[device_id]
            social = self.c2 * r2 * (global_best_pos - current_pos)
            
            # Update velocity
            particle.velocity_assignment[device_id] = inertia + cognitive + social
            
            # Probabilistically decide whether to change assignment
            if abs(particle.velocity_assignment[device_id]) > 0.5:
                # Try to move toward better assignment
                self._try_better_device_assignment(particle.solution, device_id)
    
    def _try_better_device_assignment(self, solution: Solution, device_id: int):
        """Try to find a better assignment for a device"""
        # Create mapping: point -> cloudlet
        point_to_cloudlet = {}
        for cloudlet_id, point_id in enumerate(solution.cloudlet_placement):
            if point_id != -1:
                point_to_cloudlet[point_id] = cloudlet_id
        
        # Find valid assignments
        valid_points = []
        for point_id, cloudlet_id in point_to_cloudlet.items():
            if self.problem.check_coverage(device_id, cloudlet_id, point_id):
                distance = self.problem.get_distance(device_id, point_id)
                valid_points.append((point_id, distance))
        
        if valid_points:
            # Sort by distance and probabilistically select
            valid_points.sort(key=lambda x: x[1])
            
            # Weighted random selection (favor closer points)
            weights = [1.0 / (d + 1) for _, d in valid_points]
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
            
            selected_point = np.random.choice(
                [p for p, _ in valid_points],
                p=weights
            )
            solution.device_assignment[device_id] = selected_point
    
    def solve(self) -> Solution:
        """Run PSO algorithm to find optimal solution"""
        print("="*80)
        print("PARTICLE SWARM OPTIMIZATION (PSO) ALGORITHM")
        print("="*80)
        print(f"\nPSO Parameters:")
        print(f"  Swarm Size: {self.swarm_size}")
        print(f"  Max Iterations: {self.max_iterations}")
        print(f"  Inertia Weight (w): {self.w}")
        print(f"  Cognitive Coefficient (c1): {self.c1}")
        print(f"  Social Coefficient (c2): {self.c2}")
        print(f"  Cost Weight: {self.cost_weight}")
        print(f"  Latency Weight: {self.latency_weight}")
        
        # Initialize swarm
        swarm = self.initialize_swarm()
        
        print(f"\nRunning PSO for {self.max_iterations} iterations...")
        
        # Main PSO loop
        for iteration in range(self.max_iterations):
            # Update each particle
            for particle in swarm:
                # Update velocity and position
                self.update_velocity_and_position(particle)
                
                # Evaluate new position
                self.problem.evaluate_solution(particle.solution)
                current_fitness = self.fitness(particle.solution)
                
                # Update personal best
                if current_fitness < particle.best_fitness:
                    particle.best_solution = copy.deepcopy(particle.solution)
                    particle.best_fitness = current_fitness
                
                # Update global best
                if current_fitness < self.global_best_fitness:
                    self.global_best_solution = copy.deepcopy(particle.solution)
                    self.global_best_fitness = current_fitness
            
            # Track progress
            avg_fitness = np.mean([self.fitness(p.solution) for p in swarm])
            feasible_count = sum(1 for p in swarm if p.solution.is_feasible)
            
            self.history.append({
                'iteration': iteration,
                'best_fitness': self.global_best_fitness,
                'avg_fitness': avg_fitness,
                'feasible_count': feasible_count,
                'best_cost': self.global_best_solution.total_cost,
                'best_latency': self.global_best_solution.total_latency,
                'is_feasible': self.global_best_solution.is_feasible
            })
            
            # Progress reporting
            if iteration % 50 == 0:
                print(f"Iteration {iteration}: Best Fitness={self.global_best_fitness:.2f}, "
                      f"Feasible={feasible_count}/{self.swarm_size}, "
                      f"Cost=${self.global_best_solution.total_cost:.2f}, "
                      f"Latency={self.global_best_solution.total_latency:.2f}")
            
            # Optional: Adaptive inertia weight (linearly decreasing)
            # self.w = 0.9 - (0.5 * iteration / self.max_iterations)
        
        print("\nOptimization complete!")
        print(f"\nFinal Results:")
        print(f"  Best Fitness: {self.global_best_fitness:.2f}")
        print(f"  Total Cost: ${self.global_best_solution.total_cost:.2f}")
        print(f"  Total Latency: {self.global_best_solution.total_latency:.2f}")
        print(f"  Feasible: {self.global_best_solution.is_feasible}")
        print(f"  Cloudlets Used: {sum(1 for p in self.global_best_solution.cloudlet_placement if p != -1)}")
        print(f"  Devices Covered: {sum(1 for p in self.global_best_solution.device_assignment if p != -1)}")
        
        print("="*80)
        return self.global_best_solution