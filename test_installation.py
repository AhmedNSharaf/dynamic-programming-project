"""
Quick test script to verify installation and basic functionality
"""

import sys

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")

    try:
        import numpy as np
        print("  [OK] NumPy installed")
    except ImportError:
        print("  [ERROR] NumPy not found - run: pip install numpy")
        return False

    try:
        import matplotlib.pyplot as plt
        print("  [OK] Matplotlib installed")
    except ImportError:
        print("  [ERROR] Matplotlib not found - run: pip install matplotlib")
        print("    (Optional: Only needed for visualizations)")

    try:
        from cloudlet_placement import CloudletPlacementProblem, Solution
        print("  [OK] Cloudlet placement module loaded")
    except ImportError as e:
        print(f"  [ERROR] Error loading cloudlet_placement: {e}")
        return False

    try:
        from greedy_local_search import GreedyLocalSearchSolver
        print("  [OK] Greedy Local Search module loaded")
    except ImportError as e:
        print(f"  [ERROR] Error loading greedy_local_search: {e}")
        return False

    try:
        from data_generator import generate_sample_data
        print("  [OK] Data generator module loaded")
    except ImportError as e:
        print(f"  [ERROR] Error loading data_generator: {e}")
        return False

    return True


def test_basic_functionality():
    """Test basic problem creation and solving"""
    print("\nTesting basic functionality...")

    try:
        from data_generator import generate_sample_data
        from cloudlet_placement import CloudletPlacementProblem
        from greedy_local_search import GreedyLocalSearchSolver

        # Generate small test problem
        print("  Creating small test problem (5 points, 3 cloudlets, 10 devices)...")
        points, cloudlets, devices = generate_sample_data(
            num_points=5,
            num_cloudlets=3,
            num_devices=10,
            seed=42
        )
        print(f"  [OK] Generated {len(points)} points, {len(cloudlets)} cloudlets, {len(devices)} devices")

        # Create problem instance
        print("  Creating problem instance...")
        problem = CloudletPlacementProblem(points, cloudlets, devices)
        print("  [OK] Problem instance created")

        # Test quick solve with Greedy + Local Search
        print("  Running quick Greedy + Local Search test...")
        solver = GreedyLocalSearchSolver(
            problem,
            cost_weight=0.5,
            latency_weight=0.5,
            max_iterations=50,
            no_improvement_limit=20
        )

        solution = solver.solve()
        print(f"  [OK] Solution found!")
        print(f"    Feasible: {solution.is_feasible}")
        print(f"    Cost: ${solution.total_cost:.2f}")
        print(f"    Latency: {solution.total_latency:.2f}")

        return True

    except Exception as e:
        print(f"  [ERROR] Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*70)
    print("CLOUDLET PLACEMENT - INSTALLATION TEST")
    print("="*70)

    if not test_imports():
        print("\n[ERROR] Import test failed. Please install required packages:")
        print("  pip install -r requirements.txt")
        return

    if not test_basic_functionality():
        print("\n[ERROR] Functionality test failed.")
        return

    print("\n" + "="*70)
    print("[SUCCESS] ALL TESTS PASSED!")
    print("="*70)
    print("\nEverything is working correctly. You can now run:")
    print("  python main.py")
    print("\nTo start the full optimization.")


if __name__ == '__main__':
    main()
