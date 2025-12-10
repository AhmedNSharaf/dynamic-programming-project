"""
Main script to run the cloudlet placement optimization using PSO
"""

import time
from data_generator import generate_sample_data, save_data_to_json, load_data_from_json
from cloudlet_placement import CloudletPlacementProblem
from pso_cloudlet_placement import PSOSolver
from visualizer import visualize_solution, print_solution_report, plot_convergence


def run_pso_optimization(use_existing_data=False):
    """Run the complete PSO optimization pipeline"""

    print("="*80)
    print("CLOUDLET PLACEMENT OPTIMIZATION USING PSO")
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

    # Step 3: Configure and run PSO algorithm
    print("\nConfiguring PSO Algorithm:")
    solver_params = {
        'swarm_size': 50,            # Number of particles in swarm
        'max_iterations': 500,       # Maximum iterations
        'w': 0.7,                    # Inertia weight (0.4-0.9)
        'c1': 1.5,                   # Cognitive coefficient (1.0-2.0)
        'c2': 1.5,                   # Social coefficient (1.0-2.0)
        'cost_weight': 0.5,          # Weight for cost objective
        'latency_weight': 0.5        # Weight for latency objective
    }

    for param, value in solver_params.items():
        print(f"  {param}: {value}")

    solver = PSOSolver(problem, **solver_params)

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
        visualize_solution(problem, best_solution, save_path='pso_solution_visualization.png')
        
        # Plot convergence
        if solver.history:
            plot_convergence(solver.history, save_path='pso_convergence.png')
        
        print("\nVisualization files created successfully!")
        print("  - pso_solution_visualization.png")
        print("  - pso_convergence.png")
    except Exception as e:
        print(f"\nWarning: Could not generate visualizations: {e}")
        print("(Make sure matplotlib is installed: pip install matplotlib)")

    # Step 7: Summary
    print("\n" + "="*80)
    print("OPTIMIZATION SUMMARY")
    print("="*80)
    print(f"Algorithm: Particle Swarm Optimization (PSO)")
    print(f"Status: {'[SUCCESS]' if best_solution.is_feasible else '[INFEASIBLE]'}")
    print(f"Total Cost: ${best_solution.total_cost:,.2f}")
    print(f"Total Latency: {best_solution.total_latency:.2f}")
    print(f"Cloudlets Used: {sum(1 for p in best_solution.cloudlet_placement if p != -1)}/{len(cloudlets)}")
    print(f"Devices Covered: {sum(1 for p in best_solution.device_assignment if p != -1)}/{len(devices)}")
    print(f"Execution Time: {elapsed_time:.2f} seconds")

    if best_solution.is_feasible:
        print("\n[OK] All constraints satisfied!")
        print("[OK] Full device coverage achieved!")
    else:
        print(f"\n[ERROR] Solution has {best_solution.violations} constraint violations")

    print("="*80)

    return problem, best_solution, solver


def compare_algorithms():
    """Compare PSO with Greedy Local Search"""
    from greedy_local_search import GreedyLocalSearchSolver
    
    print("\n" + "="*80)
    print("ALGORITHM COMPARISON: PSO vs GREEDY LOCAL SEARCH")
    print("="*80)
    
    # Generate data
    print("\nGenerating problem data...")
    candidate_points, cloudlets, devices = generate_sample_data(
        num_points=15,
        num_cloudlets=8,
        num_devices=50,
        seed=42
    )
    
    problem = CloudletPlacementProblem(candidate_points, cloudlets, devices)
    
    results = []
    
    # Run PSO
    print("\n" + "-"*80)
    print("Running PSO Algorithm...")
    print("-"*80)
    
    pso_solver = PSOSolver(
        problem,
        swarm_size=50,
        max_iterations=300,
        w=0.7,
        c1=1.5,
        c2=1.5,
        cost_weight=0.5,
        latency_weight=0.5
    )
    
    start_time = time.time()
    pso_solution = pso_solver.solve()
    pso_time = time.time() - start_time
    
    results.append({
        'algorithm': 'PSO',
        'feasible': pso_solution.is_feasible,
        'cost': pso_solution.total_cost,
        'latency': pso_solution.total_latency,
        'time': pso_time,
        'violations': pso_solution.violations
    })
    
    # Run Greedy Local Search
    print("\n" + "-"*80)
    print("Running Greedy Local Search Algorithm...")
    print("-"*80)
    
    gls_solver = GreedyLocalSearchSolver(
        problem,
        cost_weight=0.5,
        latency_weight=0.5,
        max_iterations=300,
        no_improvement_limit=50
    )
    
    start_time = time.time()
    gls_solution = gls_solver.solve()
    gls_time = time.time() - start_time
    
    results.append({
        'algorithm': 'Greedy + LS',
        'feasible': gls_solution.is_feasible,
        'cost': gls_solution.total_cost,
        'latency': gls_solution.total_latency,
        'time': gls_time,
        'violations': gls_solution.violations
    })
    
    # Comparison table
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    print(f"{'Algorithm':<15} {'Feasible':<10} {'Cost':<15} {'Latency':<12} "
          f"{'Violations':<12} {'Time(s)':<10}")
    print("-"*80)
    
    for r in results:
        print(f"{r['algorithm']:<15} "
              f"{'Yes' if r['feasible'] else 'No':<10} "
              f"${r['cost']:<14.2f} "
              f"{r['latency']:<12.2f} "
              f"{r['violations']:<12} "
              f"{r['time']:<10.2f}")
    
    print("="*80)
    
    # Determine winner
    if results[0]['feasible'] and results[1]['feasible']:
        pso_fitness = 0.5 * results[0]['cost'] + 0.5 * results[0]['latency']
        gls_fitness = 0.5 * results[1]['cost'] + 0.5 * results[1]['latency']
        
        print("\nBoth algorithms found feasible solutions!")
        print(f"PSO Combined Objective: {pso_fitness:.2f}")
        print(f"Greedy+LS Combined Objective: {gls_fitness:.2f}")
        print(f"\nWinner: {'PSO' if pso_fitness < gls_fitness else 'Greedy+LS'} "
              f"(Better by {abs(pso_fitness - gls_fitness):.2f})")
    elif results[0]['feasible']:
        print("\nPSO found a feasible solution, Greedy+LS did not.")
        print("Winner: PSO")
    elif results[1]['feasible']:
        print("\nGreedy+LS found a feasible solution, PSO did not.")
        print("Winner: Greedy+LS")
    else:
        print("\nNeither algorithm found a feasible solution.")
        print(f"PSO violations: {results[0]['violations']}")
        print(f"Greedy+LS violations: {results[1]['violations']}")


def run_pso_experiments():
    """Run multiple PSO experiments with different parameters"""
    
    print("\n" + "="*80)
    print("PSO PARAMETER TUNING EXPERIMENTS")
    print("="*80)
    
    # Generate fixed problem
    points, cloudlets, devices = generate_sample_data(
        num_points=15,
        num_cloudlets=8,
        num_devices=50,
        seed=42
    )
    problem = CloudletPlacementProblem(points, cloudlets, devices)
    
    experiments = [
        {'name': 'Balanced', 'w': 0.7, 'c1': 1.5, 'c2': 1.5},
        {'name': 'Exploration', 'w': 0.9, 'c1': 2.0, 'c2': 1.0},
        {'name': 'Exploitation', 'w': 0.4, 'c1': 1.0, 'c2': 2.0},
        {'name': 'High Inertia', 'w': 0.9, 'c1': 1.5, 'c2': 1.5},
        {'name': 'Low Inertia', 'w': 0.4, 'c1': 1.5, 'c2': 1.5},
    ]
    
    results = []
    
    for exp in experiments:
        print(f"\n{'-'*80}")
        print(f"Experiment: {exp['name']}")
        print(f"  w={exp['w']}, c1={exp['c1']}, c2={exp['c2']}")
        print(f"{'-'*80}")
        
        solver = PSOSolver(
            problem,
            swarm_size=30,
            max_iterations=200,
            w=exp['w'],
            c1=exp['c1'],
            c2=exp['c2'],
            cost_weight=0.5,
            latency_weight=0.5
        )
        
        start_time = time.time()
        solution = solver.solve()
        elapsed_time = time.time() - start_time
        
        results.append({
            'name': exp['name'],
            'w': exp['w'],
            'c1': exp['c1'],
            'c2': exp['c2'],
            'feasible': solution.is_feasible,
            'cost': solution.total_cost,
            'latency': solution.total_latency,
            'time': elapsed_time
        })
    
    # Summary table
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(f"{'Configuration':<15} {'w':<6} {'c1':<6} {'c2':<6} "
          f"{'Feasible':<10} {'Cost':<12} {'Latency':<10} {'Time(s)':<8}")
    print("-"*80)
    
    for r in results:
        print(f"{r['name']:<15} {r['w']:<6} {r['c1']:<6} {r['c2']:<6} "
              f"{'Yes' if r['feasible'] else 'No':<10} "
              f"${r['cost']:<11.2f} {r['latency']:<10.2f} {r['time']:<8.2f}")
    
    print("="*80)


if __name__ == '__main__':
    # Run PSO optimization
    problem, solution, solver = run_pso_optimization(use_existing_data=False)
    
    # Uncomment to compare with Greedy Local Search
    # compare_algorithms()
    
    # Uncomment to run parameter tuning experiments
    # run_pso_experiments()
    
    print("\n\nTo compare algorithms, uncomment compare_algorithms() in main_pso.py")
    print("To tune PSO parameters, uncomment run_pso_experiments() in main_pso.py")