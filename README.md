# Cloudlet Placement in Edge Computing

AI-based solution for optimal cloudlet placement using **Greedy Heuristic + Local Search** with multi-objective optimization.

## Problem Overview

Strategic placement of cloudlets (small data centers) to serve mobile/IoT devices with:
- **Minimal placement cost**
- **Minimal latency** (distance-based)
- **Full device coverage**
- **Resource constraints satisfaction**

## Features

- **Multi-objective optimization**: Balances cost vs. latency
- **Greedy Heuristic + Local Search**: Fast two-phase optimization approach
- **Heterogeneous cloudlets**: Different capacities, coverage radii, and costs
- **Constraint satisfaction**: Coverage, capacity, and assignment constraints
- **Visualization**: Solution maps and resource utilization
- **Scalable**: Works with varying problem sizes
- **Fast execution**: 2-5 seconds for typical problems

## Installation

```bash
pip install -r requirements.txt
```

Requirements:
- Python 3.7+
- NumPy
- Matplotlib

## Quick Start

Run the optimization with default settings:

```bash
python main.py
```

This will:
1. Generate sample problem data (15 points, 8 cloudlets, 50 devices)
2. Run Greedy + Local Search optimization (~2-5 seconds)
3. Display detailed solution report
4. Save visualization as PNG file

## File Structure

- `cloudlet_placement.py` - Core problem representation and evaluation
- `greedy_local_search.py` - Greedy + Local Search algorithm implementation
- `data_generator.py` - Sample data generation utilities
- `visualizer.py` - Visualization and reporting tools
- `main.py` - Main execution script
- `problem_data.json` - Generated problem data (auto-created)
- `ALGORITHM_EXPLANATION.md` - Detailed algorithm documentation

## Usage Examples

### Basic Usage

```python
from data_generator import generate_sample_data
from cloudlet_placement import CloudletPlacementProblem
from greedy_local_search import GreedyLocalSearchSolver
from visualizer import visualize_solution, print_solution_report

# Generate data
points, cloudlets, devices = generate_sample_data(
    num_points=15,
    num_cloudlets=8,
    num_devices=50
)

# Create problem
problem = CloudletPlacementProblem(points, cloudlets, devices)

# Solve with Greedy + Local Search
solver = GreedyLocalSearchSolver(
    problem,
    cost_weight=0.5,
    latency_weight=0.5,
    max_iterations=1000,
    no_improvement_limit=100
)

solution = solver.solve()

# View results
print_solution_report(problem, solution)
visualize_solution(problem, solution)
```

### Custom Configuration

```python
# Adjust GA parameters
solver = GeneticAlgorithmSolver(
    problem,
    population_size=150,     # Larger population
    generations=1000,        # More iterations
    crossover_rate=0.9,      # Higher crossover
    mutation_rate=0.15,      # Lower mutation
    cost_weight=0.7,         # Prioritize cost
    latency_weight=0.3       # Lower latency priority
)
```

### Load Existing Data

```python
from data_generator import load_data_from_json

# Load previously saved data
points, cloudlets, devices = load_data_from_json('problem_data.json')
```

### Run Multiple Experiments

Uncomment `run_experiments()` in `main.py` to test different problem sizes.

## Algorithm Details

### Genetic Algorithm Components

1. **Encoding**:
   - Cloudlet placement: Array mapping cloudlets to candidate points
   - Device assignment: Array mapping devices to serving points

2. **Fitness Function**:
   - Weighted sum: `cost_weight × cost + latency_weight × latency`
   - Heavy penalty for constraint violations

3. **Operators**:
   - **Selection**: Tournament selection
   - **Crossover**: Two-point crossover
   - **Mutation**: Random reassignment
   - **Repair**: Constraint violation fixes
   - **Elitism**: Best 10% solutions preserved

### Constraints Enforced

1. ✓ Full device coverage (100%)
2. ✓ Coverage radius limits
3. ✓ CPU capacity constraints
4. ✓ Memory capacity constraints
5. ✓ Storage capacity constraints
6. ✓ One cloudlet per location
7. ✓ Unique cloudlet placement

## Output

### Console Output
- Problem statistics
- GA progress (every 50 generations)
- Detailed solution report
- Constraint violation details
- Resource utilization per cloudlet

### Generated Files
- `problem_data.json` - Input data
- `solution_visualization.png` - Placement map and resource usage
- `convergence_plot.png` - Algorithm convergence metrics

## Visualization

### Solution Map
- Red squares: Placed cloudlets
- Green dots: Assigned devices
- Gray lines: Device-cloudlet connections
- Dashed circles: Coverage radii

### Resource Utilization
- Bar chart showing CPU/Memory/Storage usage per cloudlet
- Red line indicating 100% capacity limit

### Convergence Plots
- Fitness over generations
- Cost reduction
- Latency improvement
- Feasibility evolution

## Customization

### Problem Size
Edit in `main.py`:
```python
candidate_points, cloudlets, devices = generate_sample_data(
    num_points=20,      # More locations
    num_cloudlets=12,   # More cloudlets
    num_devices=100     # More devices
)
```

### Cloudlet Types
Modify in `data_generator.py`:
```python
cloudlet_types = [
    {'cpu': 32, 'memory': 64, 'storage': 1000, 'radius': 40, 'cost': 2500},
    # Add custom types
]
```

### Objective Weights
Adjust in solver initialization:
```python
solver = GeneticAlgorithmSolver(
    problem,
    cost_weight=0.3,      # Lower cost priority
    latency_weight=0.7    # Higher latency priority
)
```

## Performance

- Small problems (10 points, 30 devices): ~5-10 seconds
- Medium problems (15 points, 50 devices): ~15-30 seconds
- Large problems (20 points, 80 devices): ~45-90 seconds

*Performance varies with GA parameters and hardware*

## Troubleshooting

**No feasible solution found:**
- Increase number of cloudlets
- Use larger coverage radii
- Increase population size or generations
- Adjust cloudlet capacities

**Poor solution quality:**
- Increase number of generations
- Adjust mutation/crossover rates
- Try different weight combinations
- Run multiple times with different seeds

## References

- Multi-objective optimization in edge computing
- Genetic algorithms for combinatorial optimization
- Cloudlet placement strategies
- NP-hard problem approximation

## License

Educational/Research Use
