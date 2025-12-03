"""
Main script to run the cloudlet placement optimization
"""

import time
from data_generator import generate_sample_data, save_data_to_json, load_data_from_json
from cloudlet_placement import CloudletPlacementProblem
from greedy_local_search import GreedyLocalSearchSolver
from visualizer import visualize_solution, print_solution_report


def run_optimization(use_existing_data=False):
    """Run the complete optimization pipeline"""

    print("="*80)
    print("CLOUDLET PLACEMENT OPTIMIZATION IN EDGE COMPUTING")
    print("="*80)

    # Step 1: Load or generate data
    if use_existing_data:
        try:
            print("\nLoading data from problem_data.json...")
            candidate_points, cloudlets, devices = load_data_from_json('problem_data.json')
        except FileNotFoundError:
            print("File not found. Generating new data...")
            candidate_points, cloudlets, devices = generate_sample_data()
            save_data_to_json(candidate_points, cloudlets, devices)
    else:
        print("\nGenerating sample problem data...")
        candidate_points, cloudlets, devices = generate_sample_data(
            num_points=15,      # Number of candidate locations
            num_cloudlets=8,    # Number of available cloudlets
            num_devices=50,     # Number of devices to serve
            area_size=100.0,    # Coordinate space size
            seed=42
        )
        save_data_to_json(candidate_points, cloudlets, devices)

    print(f"\nProblem Size:")
    print(f"  Candidate Points: {len(candidate_points)}")
    print(f"  Available Cloudlets: {len(cloudlets)}")
    print(f"  Devices to Serve: {len(devices)}")

    # Display cloudlet types
    print(f"\nCloudlet Types:")
    for cloudlet in cloudlets:
        print(f"  Cloudlet {cloudlet.id}: CPU={cloudlet.cpu_capacity}GHz, "
              f"Memory={cloudlet.memory_capacity}GB, Storage={cloudlet.storage_capacity}GB, "
              f"Radius={cloudlet.coverage_radius}, Cost=${cloudlet.base_cost}")

    # Step 2: Create problem instance
    print("\nInitializing problem instance...")
    problem = CloudletPlacementProblem(candidate_points, cloudlets, devices)

    # Step 3: Configure and run Greedy + Local Search algorithm
    print("\nConfiguring Greedy + Local Search Algorithm:")
    solver_params = {
        'cost_weight': 0.5,              # Weight for cost objective (0-1)
        'latency_weight': 0.5,           # Weight for latency objective (0-1)
        'max_iterations': 1000,          # Maximum local search iterations
        'no_improvement_limit': 100      # Stop if no improvement for N iterations
    }

    for param, value in solver_params.items():
        print(f"  {param}: {value}")

    solver = GreedyLocalSearchSolver(problem, **solver_params)

    # Step 4: Run optimization
    print("\n" + "-"*80)
    start_time = time.time()

    best_solution = solver.solve()

    elapsed_time = time.time() - start_time
    print(f"\nTotal execution time: {elapsed_time:.2f} seconds")
    print("-"*80)

    # Step 5: Display results
    print_solution_report(problem, best_solution)

    # Step 6: Generate visualizations
    print("\nGenerating visualizations...")
    try:
        visualize_solution(problem, best_solution, save_path='solution_visualization.png')
        print("\nVisualization files created successfully!")
    except Exception as e:
        print(f"\nWarning: Could not generate visualizations: {e}")
        print("(Make sure matplotlib is installed: pip install matplotlib)")

    # Step 7: Summary
    print("\n" + "="*80)
    print("OPTIMIZATION SUMMARY")
    print("="*80)
    print(f"Status: {'[SUCCESS]' if best_solution.is_feasible else '[INFEASIBLE]'}")
    print(f"Total Cost: ${best_solution.total_cost:,.2f}")
    print(f"Total Latency: {best_solution.total_latency:.2f}")
    print(f"Cloudlets Used: {sum(1 for p in best_solution.cloudlet_placement if p != -1)}/{len(cloudlets)}")
    print(f"Devices Covered: {sum(1 for p in best_solution.device_assignment if p != -1)}/{len(devices)}")

    if best_solution.is_feasible:
        print("\n[OK] All constraints satisfied!")
        print("[OK] Full device coverage achieved!")
    else:
        print(f"\n[ERROR] Solution has {best_solution.violations} constraint violations")

    print("="*80)

    return problem, best_solution, solver


def run_experiments():
    """Run multiple experiments with different configurations"""

    print("\n" + "="*80)
    print("RUNNING MULTIPLE EXPERIMENTS")
    print("="*80)

    experiments = [
        {'name': 'Small Problem', 'points': 10, 'cloudlets': 5, 'devices': 30},
        {'name': 'Medium Problem', 'points': 15, 'cloudlets': 8, 'devices': 50},
        {'name': 'Large Problem', 'points': 20, 'cloudlets': 12, 'devices': 80},
    ]

    results = []

    for exp in experiments:
        print(f"\n{'-'*80}")
        print(f"Experiment: {exp['name']}")
        print(f"{'-'*80}")

        # Generate data
        points, cloudlets, devices = generate_sample_data(
            num_points=exp['points'],
            num_cloudlets=exp['cloudlets'],
            num_devices=exp['devices']
        )

        # Create problem
        problem = CloudletPlacementProblem(points, cloudlets, devices)

        # Solve with Greedy + Local Search
        solver = GreedyLocalSearchSolver(
            problem,
            cost_weight=0.5,
            latency_weight=0.5,
            max_iterations=500,
            no_improvement_limit=50
        )

        start_time = time.time()
        solution = solver.solve()
        elapsed_time = time.time() - start_time

        # Store results
        results.append({
            'name': exp['name'],
            'points': exp['points'],
            'cloudlets': exp['cloudlets'],
            'devices': exp['devices'],
            'feasible': solution.is_feasible,
            'cost': solution.total_cost,
            'latency': solution.total_latency,
            'time': elapsed_time
        })

        print(f"\nResult: {'FEASIBLE' if solution.is_feasible else 'INFEASIBLE'}")
        print(f"Cost: ${solution.total_cost:,.2f}, Latency: {solution.total_latency:.2f}")
        print(f"Time: {elapsed_time:.2f}s")

    # Summary table
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(f"{'Experiment':<20} {'Points':<8} {'Cloudlets':<10} {'Devices':<8} "
          f"{'Feasible':<10} {'Cost':<12} {'Latency':<10} {'Time(s)':<8}")
    print("-"*80)

    for r in results:
        print(f"{r['name']:<20} {r['points']:<8} {r['cloudlets']:<10} {r['devices']:<8} "
              f"{'Yes' if r['feasible'] else 'No':<10} ${r['cost']:<11.2f} "
              f"{r['latency']:<10.2f} {r['time']:<8.2f}")

    print("="*80)


if __name__ == '__main__':
    # Run single optimization
    problem, solution, solver = run_optimization(use_existing_data=False)

    # Uncomment to run multiple experiments
    # run_experiments()

    print("\n\nTo run again with different parameters, modify the configuration in main.py")
    print("To run experiments with multiple problem sizes, uncomment run_experiments()")
