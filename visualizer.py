"""
Visualization and reporting tools for cloudlet placement solutions
"""

import matplotlib.pyplot as plt
import numpy as np
from cloudlet_placement import CloudletPlacementProblem, Solution


def visualize_solution(problem: CloudletPlacementProblem, solution: Solution, save_path='solution.png'):
    """Visualize the cloudlet placement and device assignments"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Create mapping: point -> cloudlet
    point_to_cloudlet = {}
    for cloudlet_id, point_id in enumerate(solution.cloudlet_placement):
        if point_id != -1:
            point_to_cloudlet[point_id] = cloudlet_id

    # Plot 1: Placement and assignments
    ax1.set_title('Cloudlet Placement and Device Assignments', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.grid(True, alpha=0.3)

    # Plot candidate points
    for point in problem.candidate_points:
        if point.id in point_to_cloudlet:
            cloudlet_id = point_to_cloudlet[point.id]
            cloudlet = problem.cloudlets[cloudlet_id]

            # Draw coverage radius
            circle = plt.Circle((point.x, point.y), cloudlet.coverage_radius,
                              color='blue', fill=False, linestyle='--', alpha=0.3, linewidth=1)
            ax1.add_patch(circle)

            # Plot cloudlet location
            ax1.scatter(point.x, point.y, c='red', s=300, marker='s',
                       edgecolors='black', linewidths=2, zorder=5,
                       label='Cloudlet' if point.id == list(point_to_cloudlet.keys())[0] else '')
            ax1.text(point.x, point.y, f'C{cloudlet_id}', ha='center', va='center',
                    fontsize=9, fontweight='bold', color='white')
        else:
            # Unused candidate point
            ax1.scatter(point.x, point.y, c='lightgray', s=100, marker='s',
                       alpha=0.5, edgecolors='gray',
                       label='Unused Point' if point.id == 0 else '')

    # Plot devices and their assignments
    for device_id, device in enumerate(problem.devices):
        assigned_point_id = solution.device_assignment[device_id]

        if assigned_point_id != -1 and assigned_point_id in point_to_cloudlet:
            point = problem.candidate_points[assigned_point_id]

            # Draw connection line
            ax1.plot([device.x, point.x], [device.y, point.y],
                    'gray', alpha=0.2, linewidth=0.5, zorder=1)

            # Plot device
            ax1.scatter(device.x, device.y, c='green', s=30, marker='o',
                       alpha=0.7, edgecolors='darkgreen', linewidths=0.5,
                       label='Device (assigned)' if device_id == 0 else '')
        else:
            # Unassigned device
            ax1.scatter(device.x, device.y, c='orange', s=30, marker='x',
                       linewidths=2, label='Device (unassigned)' if device_id == 0 else '')

    # Remove duplicate labels
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=9)

    # Plot 2: Resource utilization
    ax2.set_title('Cloudlet Resource Utilization', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Cloudlet ID')
    ax2.set_ylabel('Utilization (%)')
    ax2.set_ylim(0, 120)
    ax2.grid(True, alpha=0.3, axis='y')

    # Calculate resource usage per cloudlet
    cloudlet_usage = {}
    for cloudlet_id, point_id in enumerate(solution.cloudlet_placement):
        if point_id != -1:
            cloudlet_usage[cloudlet_id] = {'cpu': 0, 'memory': 0, 'storage': 0}

    for device_id, point_id in enumerate(solution.device_assignment):
        if point_id != -1 and point_id in point_to_cloudlet:
            cloudlet_id = point_to_cloudlet[point_id]
            device = problem.devices[device_id]

            cloudlet_usage[cloudlet_id]['cpu'] += device.cpu_demand
            cloudlet_usage[cloudlet_id]['memory'] += device.memory_demand
            cloudlet_usage[cloudlet_id]['storage'] += device.storage_demand

    # Plot bars
    if cloudlet_usage:
        cloudlet_ids = sorted(cloudlet_usage.keys())
        x = np.arange(len(cloudlet_ids))
        width = 0.25

        cpu_utilization = []
        memory_utilization = []
        storage_utilization = []

        for cid in cloudlet_ids:
            cloudlet = problem.cloudlets[cid]
            cpu_pct = (cloudlet_usage[cid]['cpu'] / cloudlet.cpu_capacity) * 100
            memory_pct = (cloudlet_usage[cid]['memory'] / cloudlet.memory_capacity) * 100
            storage_pct = (cloudlet_usage[cid]['storage'] / cloudlet.storage_capacity) * 100

            cpu_utilization.append(cpu_pct)
            memory_utilization.append(memory_pct)
            storage_utilization.append(storage_pct)

        bars1 = ax2.bar(x - width, cpu_utilization, width, label='CPU', color='#FF6B6B')
        bars2 = ax2.bar(x, memory_utilization, width, label='Memory', color='#4ECDC4')
        bars3 = ax2.bar(x + width, storage_utilization, width, label='Storage', color='#45B7D1')

        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax2.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.0f}%', ha='center', va='bottom', fontsize=8)

        # Add horizontal line at 100%
        ax2.axhline(y=100, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Capacity Limit')

        ax2.set_xticks(x)
        ax2.set_xticklabels([f'C{cid}' for cid in cloudlet_ids])
        ax2.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {save_path}")
    plt.close()


def plot_convergence(history, save_path='convergence.png'):
    """Plot algorithm convergence over generations"""

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    generations = [h['generation'] for h in history]

    # Plot 1: Fitness over time
    ax1.set_title('Fitness Convergence', fontsize=12, fontweight='bold')
    ax1.plot(generations, [h['best_fitness'] for h in history], 'b-', linewidth=2, label='Best Fitness')
    ax1.plot(generations, [h['avg_fitness'] for h in history], 'r--', alpha=0.6, label='Avg Fitness')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Fitness Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Cost over time
    ax2.set_title('Total Placement Cost', fontsize=12, fontweight='bold')
    ax2.plot(generations, [h['best_cost'] for h in history], 'g-', linewidth=2)
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Total Cost')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Latency over time
    ax3.set_title('Total Latency', fontsize=12, fontweight='bold')
    ax3.plot(generations, [h['best_latency'] for h in history], 'm-', linewidth=2)
    ax3.set_xlabel('Generation')
    ax3.set_ylabel('Total Latency (distance units)')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Feasibility over time
    ax4.set_title('Feasible Solutions in Population', fontsize=12, fontweight='bold')
    ax4.plot(generations, [h['feasible_count'] for h in history], 'c-', linewidth=2)
    ax4.set_xlabel('Generation')
    ax4.set_ylabel('Number of Feasible Solutions')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Convergence plot saved to {save_path}")
    plt.close()


def print_solution_report(problem: CloudletPlacementProblem, solution: Solution):
    """Print detailed solution report"""

    print("\n" + "="*80)
    print("SOLUTION REPORT")
    print("="*80)

    print(f"\nFeasibility: {'[FEASIBLE]' if solution.is_feasible else '[INFEASIBLE]'}")
    if not solution.is_feasible:
        print(f"  Constraint Violations: {solution.violations}")

    print(f"\nObjectives:")
    print(f"  Total Placement Cost: ${solution.total_cost:,.2f}")
    print(f"  Total Latency: {solution.total_latency:.2f} distance units")

    # Create mapping: point -> cloudlet
    point_to_cloudlet = {}
    for cloudlet_id, point_id in enumerate(solution.cloudlet_placement):
        if point_id != -1:
            point_to_cloudlet[point_id] = cloudlet_id

    print(f"\nCloudlet Placements: ({len(point_to_cloudlet)} cloudlets placed)")

    # Calculate resource usage
    cloudlet_info = {}
    for cloudlet_id, point_id in enumerate(solution.cloudlet_placement):
        if point_id != -1:
            cloudlet_info[cloudlet_id] = {
                'point_id': point_id,
                'devices': [],
                'cpu_used': 0,
                'memory_used': 0,
                'storage_used': 0
            }

    for device_id, point_id in enumerate(solution.device_assignment):
        if point_id != -1 and point_id in point_to_cloudlet:
            cloudlet_id = point_to_cloudlet[point_id]
            device = problem.devices[device_id]

            cloudlet_info[cloudlet_id]['devices'].append(device_id)
            cloudlet_info[cloudlet_id]['cpu_used'] += device.cpu_demand
            cloudlet_info[cloudlet_id]['memory_used'] += device.memory_demand
            cloudlet_info[cloudlet_id]['storage_used'] += device.storage_demand

    for cloudlet_id in sorted(cloudlet_info.keys()):
        info = cloudlet_info[cloudlet_id]
        cloudlet = problem.cloudlets[cloudlet_id]
        point = problem.candidate_points[info['point_id']]

        print(f"\n  Cloudlet {cloudlet_id} at Point {info['point_id']} "
              f"(x={point.x:.1f}, y={point.y:.1f}):")
        print(f"    Coverage Radius: {cloudlet.coverage_radius:.1f}")
        print(f"    Cost: ${cloudlet.base_cost * point.placement_cost_multiplier:,.2f}")
        print(f"    Devices Served: {len(info['devices'])}")
        print(f"    CPU Usage: {info['cpu_used']:.1f}/{cloudlet.cpu_capacity:.1f} GHz "
              f"({info['cpu_used']/cloudlet.cpu_capacity*100:.1f}%)")
        print(f"    Memory Usage: {info['memory_used']:.1f}/{cloudlet.memory_capacity:.1f} GB "
              f"({info['memory_used']/cloudlet.memory_capacity*100:.1f}%)")
        print(f"    Storage Usage: {info['storage_used']:.1f}/{cloudlet.storage_capacity:.1f} GB "
              f"({info['storage_used']/cloudlet.storage_capacity*100:.1f}%)")

    # Device coverage statistics
    assigned_count = sum(1 for p in solution.device_assignment if p != -1)
    print(f"\nDevice Coverage:")
    print(f"  Assigned Devices: {assigned_count}/{len(problem.devices)} "
          f"({assigned_count/len(problem.devices)*100:.1f}%)")

    if assigned_count < len(problem.devices):
        unassigned = [i for i, p in enumerate(solution.device_assignment) if p == -1]
        print(f"  Unassigned Devices: {unassigned[:10]}" +
              (f" ... and {len(unassigned)-10} more" if len(unassigned) > 10 else ""))

    print("\n" + "="*80)
