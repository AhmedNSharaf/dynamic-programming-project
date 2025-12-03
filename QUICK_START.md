# Quick Start Guide

## Installation (1 minute)

```bash
# Install required packages
pip install numpy matplotlib

# Verify installation
python test_installation.py
```

## Run Optimization (2 minutes)

```bash
# Run with default settings
python main.py
```

This will:
1. Generate sample problem (15 points, 8 cloudlets, 50 devices)
2. Run genetic algorithm (500 generations, ~6 seconds)
3. Display detailed results
4. Create visualization PNG files

## Output Files

After running, you'll get:
- `problem_data.json` - The problem instance
- `solution_visualization.png` - Map showing placements
- `convergence_plot.png` - Algorithm progress charts

## Customize Settings

### Change Problem Size

Edit [main.py](main.py#L22):
```python
generate_sample_data(
    num_points=20,      # More locations
    num_cloudlets=10,   # More cloudlets
    num_devices=80      # More devices
)
```

### Adjust Algorithm

Edit [main.py](main.py#L45):
```python
GeneticAlgorithmSolver(
    population_size=150,    # Bigger population
    generations=1000,       # More iterations
    cost_weight=0.7,        # Prioritize cost (0-1)
    latency_weight=0.3      # Prioritize latency (0-1)
)
```

## Understanding Results

### Console Output
```
Status: [SUCCESS]              ← Feasible solution found
Total Cost: $14,295.97         ← Placement cost
Total Latency: 922.58          ← Sum of distances
Cloudlets Used: 8/8            ← All cloudlets placed
Devices Covered: 50/50         ← 100% coverage
```

### Solution Visualization
- **Red squares**: Cloudlet locations
- **Green dots**: Devices (connected)
- **Dashed circles**: Coverage radius
- **Gray lines**: Device-to-cloudlet assignments

### Resource Usage Chart
- **Bar heights**: CPU/Memory/Storage utilization %
- **Red line**: 100% capacity limit
- All bars should be below the red line

## Common Tasks

### Run Multiple Experiments

Edit [main.py](main.py#L195):
```python
# Uncomment this line at the end of main.py
run_experiments()
```

### Load Existing Data

Edit [main.py](main.py#L14):
```python
run_optimization(use_existing_data=True)
```

### Create Custom Problem

```python
from data_generator import CandidatePoint, Cloudlet, Device, save_data_to_json

# Define your own data
points = [
    CandidatePoint(id=0, x=10, y=20, placement_cost_multiplier=1.0),
    # ... more points
]

cloudlets = [
    Cloudlet(id=0, cpu_capacity=32, memory_capacity=64,
             storage_capacity=1000, coverage_radius=40, base_cost=2000),
    # ... more cloudlets
]

devices = [
    Device(id=0, x=15, y=25, cpu_demand=2, memory_demand=4, storage_demand=50),
    # ... more devices
]

# Save and use
save_data_to_json(points, cloudlets, devices)
```

## Troubleshooting

### "No feasible solution found"
- Increase cloudlets: More coverage options
- Increase coverage radius: Wider reach
- Reduce devices: Less demand
- Run more generations: Better optimization

### "Poor solution quality"
- Increase `generations` to 1000+
- Increase `population_size` to 150+
- Try different `cost_weight`/`latency_weight` ratios
- Run multiple times (different random seeds)

### "Takes too long"
- Reduce `generations` to 200-300
- Reduce `population_size` to 50-80
- Use smaller problem sizes for testing

## Tips for Best Results

1. **Start Small**: Test with 10 points, 5 cloudlets, 30 devices
2. **Tune Gradually**: Increase problem size after getting good results
3. **Balance Objectives**: Try weights like 0.5/0.5, 0.7/0.3, 0.3/0.7
4. **Multiple Runs**: GA is stochastic, run 3-5 times for best solution
5. **Check Visualizations**: Visual inspection helps validate results

## Performance Expectations

| Problem Size | Time | Solution Quality |
|-------------|------|------------------|
| Small (30 devices) | ~3s | Excellent |
| Medium (50 devices) | ~6s | Very Good |
| Large (80 devices) | ~12s | Good |
| Very Large (100+ devices) | ~20s+ | Acceptable |

## Next Steps

1. Read [README.md](README.md) for detailed documentation
2. Check [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) for technical details
3. Explore code in [cloudlet_placement.py](cloudlet_placement.py)
4. Experiment with different configurations
5. Try implementing additional features

## Support

If you encounter issues:
1. Run `python test_installation.py` to verify setup
2. Check that numpy and matplotlib are installed
3. Review error messages for constraint violations
4. Adjust problem parameters to be more feasible
