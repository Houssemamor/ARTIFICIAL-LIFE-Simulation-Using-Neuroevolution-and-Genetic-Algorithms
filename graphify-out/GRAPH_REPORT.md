# Graph Report - Deep  (2026-09-18)

## Corpus Check
- Corpus is ~3,262 words - fits in a single context window. You may not need a graph.

## Summary
- 93 nodes · 78 edges · 29 communities (5 shown, 24 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Organism Core
- World/Environment
- Application & Experiments
- Core Dependencies & Utilities
- Organism Thinking & Sensors

## God Nodes (most connected - your core abstractions)
1. `Organism` - 13 edges
2. `World` - 11 edges
3. `load_config()` - 3 edges
4. `main()` - 3 edges
5. `load_config()` - 3 edges
6. `main()` - 3 edges
7. `Organism module for Artificial Life Neuroevolution Simulation. Contains the…` - 1 edges
8. `Represents an individual agent in the simulation. Attributes: id (int): Unique…` - 1 edges
9. `Initialize an organism. Args: organism_id (int): Unique identifier for this…` - 1 edges
10. `Initialize the organism's genome with random values. Args: genome_size (int):…` - 1 edges

## Surprising Connections (you probably didn't know these)
- None detected - all connections are within the same source files.

## Import Cycles
- None detected.

## Communities (29 total, 24 thin omitted)

### Community 0 - "Organism Core"
Cohesion: 0.10
Nodes (11): Organism, Update the organism's position based on steering and acceleration. Args:…, Represents an individual agent in the simulation. Attributes: id (int): Unique…, Attempt to consume food at current position. Args: world (World): The world…, Update energy level based on metabolism and actions. Args: metabolism_rate…, Increment the organism's age by one simulation step., Get the current state of the organism for serialization. Returns: dict:…, Load organism state from a dictionary. Args: state (dict): Dictionary… (+3 more)

### Community 1 - "World/Environment"
Cohesion: 0.11
Nodes (11): Calculate distance to the nearest food item. Args: x (float): X coordinate y…, Calculate distance to the nearest obstacle. Args: x (float): X coordinate y…, Represents the 2D continuous world where agents live and interact. Attributes:…, Get the current state of the world for serialization. Returns: dict: Dictionary…, Load world state from a dictionary. Args: state (dict): Dictionary containing…, Initialize the world with given dimensions. Args: width (int): Width of the…, Load world configuration from a dictionary. Args: config (dict): Configuration…, Spawn food at random positions in the world. Args: count (int): Number of food… (+3 more)

### Community 2 - "Application & Experiments"
Cohesion: 0.15
Nodes (15): load_config(), main(), Load configuration from JSON file with path traversal protection., Artificial Life Neuroevolution Simulation - GUI Mode Entry Point, Main entry point for GUI mode., # TODO: Implement actual simulation logic, argparse, load_config() (+7 more)

### Community 3 - "Core Dependencies & Utilities"
Cohesion: 0.38
Nodes (5): Organism module for Artificial Life Neuroevolution Simulation. Contains the…, json, numpy, World module for Artificial Life Neuroevolution Simulation. Contains the World…, typing

### Community 4 - "Organism Thinking & Sensors"
Cohesion: 0.40
Nodes (3): Update sensor readings based on current world state. This is a placeholder -…, Process sensor input through the neural network to get actions. This is a…, ndarray

## Knowledge Gaps
- **24 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Organism` connect `Organism Core` to `Core Dependencies & Utilities`, `Organism Thinking & Sensors`?**
  _High betweenness centrality (0.314) - this node is a cross-community bridge._
- **Why does `World` connect `World/Environment` to `Core Dependencies & Utilities`?**
  _High betweenness centrality (0.259) - this node is a cross-community bridge._
- **Should `Organism Core` be split into smaller, more focused modules?**
  _Cohesion score 0.1 - nodes in this community are weakly interconnected._
- **Should `World/Environment` be split into smaller, more focused modules?**
  _Cohesion score 0.11052631578947368 - nodes in this community are weakly interconnected._
- **Should `Application & Experiments` be split into smaller, more focused modules?**
  _Cohesion score 0.14705882352941177 - nodes in this community are weakly interconnected._