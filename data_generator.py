"""
Sample data generator for cloudlet placement problem
"""

import random
import json
from cloudlet_placement import CandidatePoint, Cloudlet, Device


def generate_sample_data(num_points: int = 15,
                        num_cloudlets: int = 8,
                        num_devices: int = 50,
                        area_size: float = 100.0,
                        seed: int = 42):
    """Generate sample data for testing the algorithm"""

    random.seed(seed)

    # Generate candidate points (e.g., libraries, subway stations)
    candidate_points = []
    for i in range(num_points):
        point = CandidatePoint(
            id=i,
            x=random.uniform(0, area_size),
            y=random.uniform(0, area_size),
            placement_cost_multiplier=random.uniform(0.8, 1.5)  # Some locations more expensive
        )
        candidate_points.append(point)

    # Generate heterogeneous cloudlets
    cloudlet_types = [
        {'cpu': 16, 'memory': 32, 'storage': 500, 'radius': 25, 'cost': 1000},  # Small
        {'cpu': 32, 'memory': 64, 'storage': 1000, 'radius': 35, 'cost': 2000},  # Medium
        {'cpu': 64, 'memory': 128, 'storage': 2000, 'radius': 50, 'cost': 4000},  # Large
    ]

    cloudlets = []
    for i in range(num_cloudlets):
        cloudlet_type = random.choice(cloudlet_types)
        cloudlet = Cloudlet(
            id=i,
            cpu_capacity=cloudlet_type['cpu'],
            memory_capacity=cloudlet_type['memory'],
            storage_capacity=cloudlet_type['storage'],
            coverage_radius=cloudlet_type['radius'],
            base_cost=cloudlet_type['cost']
        )
        cloudlets.append(cloudlet)

    # Generate devices with varying demands
    devices = []
    for i in range(num_devices):
        device = Device(
            id=i,
            x=random.uniform(0, area_size),
            y=random.uniform(0, area_size),
            cpu_demand=random.uniform(0.5, 4.0),
            memory_demand=random.uniform(1.0, 8.0),
            storage_demand=random.uniform(10, 100)
        )
        devices.append(device)

    return candidate_points, cloudlets, devices


def save_data_to_json(candidate_points, cloudlets, devices, filename='problem_data.json'):
    """Save problem data to JSON file"""
    data = {
        'candidate_points': [
            {
                'id': p.id,
                'x': p.x,
                'y': p.y,
                'placement_cost_multiplier': p.placement_cost_multiplier
            } for p in candidate_points
        ],
        'cloudlets': [
            {
                'id': c.id,
                'cpu_capacity': c.cpu_capacity,
                'memory_capacity': c.memory_capacity,
                'storage_capacity': c.storage_capacity,
                'coverage_radius': c.coverage_radius,
                'base_cost': c.base_cost
            } for c in cloudlets
        ],
        'devices': [
            {
                'id': d.id,
                'x': d.x,
                'y': d.y,
                'cpu_demand': d.cpu_demand,
                'memory_demand': d.memory_demand,
                'storage_demand': d.storage_demand
            } for d in devices
        ]
    }

    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Data saved to {filename}")


def load_data_from_json(filename='problem_data.json'):
    """Load problem data from JSON file"""
    with open(filename, 'r') as f:
        data = json.load(f)

    candidate_points = [CandidatePoint(**p) for p in data['candidate_points']]
    cloudlets = [Cloudlet(**c) for c in data['cloudlets']]
    devices = [Device(**d) for d in data['devices']]

    return candidate_points, cloudlets, devices


if __name__ == '__main__':
    # Generate and save sample data
    points, cloudlets, devices = generate_sample_data(
        num_points=15,
        num_cloudlets=8,
        num_devices=50
    )

    save_data_to_json(points, cloudlets, devices)

    print(f"\nGenerated:")
    print(f"  - {len(points)} candidate points")
    print(f"  - {len(cloudlets)} cloudlets")
    print(f"  - {len(devices)} devices")
