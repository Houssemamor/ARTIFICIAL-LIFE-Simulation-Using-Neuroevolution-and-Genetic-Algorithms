# Graph Report - Deep  (2026-09-20)

## Corpus Check
- Corpus is ~17,914 words - fits in a single context window. You may not need a graph.

## Summary
- 420 nodes · 678 edges · 67 communities (22 shown, 45 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 23 edges (avg confidence: 0.91)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Physics Core and Engine
- Spatial Hash Rationales
- AABB Collision Geometry
- Environment and Obstacle Tests
- World Distance Queries
- Config Validation Models
- Organism Lifecycle
- App Render Loop
- Design Core Concepts
- Experiments and Fitness Calibration
- Config Schema Hierarchy
- Vector2D Operations
- Dashboard and Plan Spec
- Renderer Implementation
- CI Pipeline and Dependencies
- Experiments Runner CLI
- NEAT Extension Concepts
- Food and Reset Queries
- GA Crossover Concepts
- Batched Inference and Compute
- Integration Reset Tests
- Reproducibility and Related Work
- Vector Limits and Normalize
- Vector Angle and Dot
- Weight Sum Validation
- Organism Food Consumption
- Organism Initialization
- Organism State Loading
- Phase 8 Packaging
- Apply Force Physics
- Apply Impulse Physics
- Momentum Query
- Vector From Tuple
- Vector To Tuple
- Vector Scalar Multiply
- Vector Magnitude Squared
- Vector Distance To
- Vector Distance Squared
- Vector Distance Tests
- Vector Limits Tests
- Vector Equality Tests
- Vector Operations Tests
- Vector Dot Tests
- Project Package Root
- Phase 0 Plan
- Phase 1 Plan
- Execution Modes

## God Nodes (most connected - your core abstractions)
1. `Vector2D` - 68 edges
2. `SpatialHash` - 37 edges
3. `World` - 29 edges
4. `AABB` - 29 edges
5. `Artificial Life Simulation Design Document v2.0 (Revised, fixes C1-C11)` - 26 edges
6. `PhysicsObject` - 20 edges
7. `Organism` - 15 edges
8. `load_config()` - 12 edges
9. `integrate_euler()` - 10 edges
10. `Artificial Life Simulation Design Document v1.0` - 10 edges

## Surprising Connections (you probably didn't know these)
- `Population-Wide Batched Inference (Phase 1.5 criterion)` --semantically_similar_to--> `Batched Population Inference (weight-stacked torch.bmm, Sec 12.3)`  [INFERRED] [semantically similar]
  README.md → docs/Artificial_Life_Neuroevolution_Design_Document_v2.pdf
- `MVP Dashboard Variant (Phases 1-5)` --semantically_similar_to--> `MVP Dashboard Mockup (no species metric, fixes C6)`  [INFERRED] [semantically similar]
  PLAN.md → docs/Artificial_Life_Neuroevolution_Design_Document_v2.pdf
- `Advanced Dashboard Variant (Phase 6+)` --semantically_similar_to--> `Advanced-Phase Dashboard Mockup (species and co-evolution)`  [INFERRED] [semantically similar]
  PLAN.md → docs/Artificial_Life_Neuroevolution_Design_Document_v2.pdf
- `Core Research Question (emergent adaptive behavior)` --semantically_similar_to--> `Core + Secondary Research Questions (v2.0)`  [INFERRED] [semantically similar]
  README.md → docs/Artificial_Life_Neuroevolution_Design_Document_v2.pdf
- `Fixed-Topology MLP Controller (12-32-16-3)` --semantically_similar_to--> `Fixed Feed-Forward MLP Controller (12-32-16-3)`  [INFERRED] [semantically similar]
  README.md → docs/Artificial_Life_Neuroevolution_Design_Document.pdf

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Development Phases (Build Order 0 to 8)** — plan_phase_0, plan_phase_1, plan_phase_1_5, plan_phase_2, plan_phase_3, plan_phase_4, plan_phase_4_5, plan_phase_5, plan_phase_6, plan_phase_7, plan_phase_8 [EXTRACTED 1.00]
- **Experiment Families (A, B, NC, D, E)** — docs_artificial_life_neuroevolution_design_document_v2_experiment_a, docs_artificial_life_neuroevolution_design_document_v2_experiment_b, docs_artificial_life_neuroevolution_design_document_v2_experiment_nc, docs_artificial_life_neuroevolution_design_document_v2_experiment_d, docs_artificial_life_neuroevolution_design_document_v2_experiment_e [EXTRACTED 1.00]
- **Batched Population Inference (design concept, plan module, README rule, Phase 1.5)** — docs_artificial_life_neuroevolution_design_document_v2_batched_population_inference, readme_batched_inference, plan_neural_batched_inference, plan_phase_1_5 [EXTRACTED 1.00]

## Communities (67 total, 45 thin omitted)

### Community 0 - "Physics Core and Engine"
Cohesion: 0.07
Nodes (42): math, os, Simulation engine for Artificial Life Neuroevolution Simulation. Owns the…, simulation_physics, Axis-Aligned Bounding Box for spatial partitioning and collision detection., integrate_euler(), integrate_verlet(), PhysicsObject (+34 more)

### Community 1 - "Spatial Hash Rationales"
Cohesion: 0.07
Nodes (23): Remove an object from the spatial hash., Grid-based spatial hash for efficient spatial queries. Divides space into…, Update an object's position in the spatial hash., Find all objects that occupy the given point (exact position for points,…, Find all objects that intersect with the given AABB., Find all objects within a circular area., Get IDs of all objects in the spatial hash., Get the current position of an object. (+15 more)

### Community 2 - "AABB Collision Geometry"
Cohesion: 0.06
Nodes (22): AABB, Axis-Aligned Bounding Box for spatial partitioning and collision detection., Create AABB from center point and size., Create AABB that encompasses all given points., Get the width of the bounding box., Get the height of the bounding box., Get the center point of the bounding box., Check if a point is inside the bounding box. (+14 more)

### Community 3 - "Environment and Obstacle Tests"
Cohesion: 0.11
Nodes (17): Obstacle, Represents an obstacle in the world., Initialize an obstacle. Args: x (float): X coordinate y (float): Y coordinate, tempfile, Unit tests for environment module., Test distance to nearest obstacle., Test getting and loading world state., Test world creation with default parameters. (+9 more)

### Community 4 - "World Distance Queries"
Cohesion: 0.12
Nodes (10): Spawn obstacles at random positions in the world. Args: count (int): Number of…, Check if a position is inside the world boundaries (with margin). Args: x…, Calculate distance to the nearest food item. Args: x (float): X coordinate y…, Calculate distance to the nearest obstacle. Args: x (float): X coordinate y…, Get the current state of the world for serialization. Returns: dict: Dictionary…, Represents the 2D continuous world where agents live and interact. Attributes:…, Initialize the world with given dimensions. Args: width (int): Width of the…, Load world configuration from a dictionary. Args: config (dict): Configuration… (+2 more)

### Community 5 - "Config Validation Models"
Cohesion: 0.17
Nodes (15): pydantic, BaselineConfig, load_config(), World configuration module for Artificial Life Neuroevolution Simulation.…, Baseline configuration matching the design document's Section 18.2 example., Load and validate a configuration file. Args: config_path: Path to the JSON…, Unit tests for configuration loading via simulation.world_config., Deliverable 4: baseline.json validates end-to-end via load_config. (+7 more)

### Community 6 - "Organism Lifecycle"
Cohesion: 0.12
Nodes (11): Organism, Update energy level based on metabolism and actions. Args: metabolism_rate…, Represents an individual agent in the simulation. Attributes: id (int): Unique…, Increment the organism's age by one simulation step., Get the current state of the organism for serialization. Returns: dict:…, Initialize the organism's genome with random values. Args: genome_size (int):…, Update the organism's position using placeholder random motion. This is used in…, main() (+3 more)

### Community 7 - "App Render Loop"
Cohesion: 0.26
Nodes (9): Organism module for Artificial Life Neuroevolution Simulation. Contains the…, Artificial Life Neuroevolution Simulation - GUI Mode Entry Point, json, numpy, pygame, simulation_engine, Environment module for Artificial Life Neuroevolution Simulation. Contains the…, typing (+1 more)

### Community 8 - "Design Core Concepts"
Cohesion: 0.17
Nodes (12): Core Research Question (v1.0), Artificial Life Simulation Design Document v1.0, Raw-Coefficient Fitness and Reward Design, Fixed Feed-Forward MLP Controller (12-32-16-3), Fixed-Length Flattened Genome Encoding, Predator/Prey and Reproduction Extension, 7-Ray Sensor Field (-90..+90 degrees), Core + Secondary Research Questions (v2.0) (+4 more)

### Community 9 - "Experiments and Fitness Calibration"
Cohesion: 0.20
Nodes (12): Artificial Life Simulation Design Document v2.0 (Revised, fixes C1-C11), Explicit Energy-Balance Equation (Sec 10.5, fixes C10), Experiment A - Mutation Rate (A1-A4), Experiment B - Population Size (B1-B4), Experiment D - Fitness Weighting (D1-D4), Experiment NC - Network Capacity (NC1-NC3), Fitness Calibration and Normalization (Sec 14.1, fixes C4 scale-domination), Framsticks Artificial-Life Platform (+4 more)

### Community 10 - "Config Schema Hierarchy"
Cohesion: 0.18
Nodes (11): BaseModel, AgentConfig, BrainConfig, EvolutionConfig, ExperimentConfig, PopulationConfig, Neural network architecture configuration., Population configuration. (+3 more)

### Community 12 - "Dashboard and Plan Spec"
Cohesion: 0.20
Nodes (10): Analytics Dashboard and Best-Agent Inspector, Advanced-Phase Dashboard Mockup (species and co-evolution), MVP Dashboard Mockup (no species metric, fixes C6), Advanced Dashboard Variant (Phase 6+), Dashboard Full Specification (MVP and Advanced), Implementation Plan (Companion to Design Document v2.0), Master File Checklist, MVP Dashboard Variant (Phases 1-5) (+2 more)

### Community 13 - "Renderer Implementation"
Cohesion: 0.20
Nodes (6): Handle PyGame events. Returns: False if the user requests to quit, True…, Clean up PyGame resources., PyGame renderer for the simulation world. Attributes: width (int): Width of the…, Initialize the renderer. Args: width (int): Width of the display window…, Render the world, food, obstacles, and agents. Args: world (World): The world…, Renderer

### Community 14 - "CI Pipeline and Dependencies"
Cohesion: 0.25
Nodes (8): Build Job (Python 3.9/3.10/3.11 Matrix), CI Pipeline, flake8 Lint Stage, pytest Test Stage, Core Runtime Dependencies Manifest, NumPy (>=1.24.0), PyGame (>=2.5.0), PyTorch (>=2.0.0)

### Community 15 - "Experiments Runner CLI"
Cohesion: 0.29
Nodes (7): argparse, load_config(), main(), Load configuration from JSON file with path traversal protection., Artificial Life Neuroevolution Experiments - Headless Mode Entry Point, Main entry point for headless experiment mode., # TODO: Implement actual experiment logic

### Community 16 - "NEAT Extension Concepts"
Cohesion: 0.29
Nodes (8): Compatibility-Distance Formula (C.5), Complexification from Minimal Topology (C.6), Innovation-Number Historical Markings (C.2), NEAT-Style Topology Extension (Phase 7, Appendix C), Open Question: Batched Inference under Variable Topology (C.7), Speciation and Fitness Sharing (C.5), evolution/neat package (genome, innovation, speciation), Phase 7 - NEAT-Style Extension (Appendix C)

### Community 17 - "Food and Reset Queries"
Cohesion: 0.25
Nodes (6): Food, Represents a food item in the world., Initialize a food item. Args: x (float): X coordinate y (float): Y coordinate, Load world state from a dictionary. Args: state (dict): Dictionary containing…, Test distance to nearest food., test_distance_to_nearest_food()

### Community 18 - "GA Crossover Concepts"
Cohesion: 0.29
Nodes (7): Custom Genetic Algorithm Design (selection/crossover/mutation), Competing-Conventions Problem (permutation symmetry, fixes C2), Three Crossover Methods (blend default / uniform / mutation-only), Experiment E - Crossover Method (E1-E3), Statistical Comparison Protocol (Mann-Whitney U + bootstrap CI + Holm-Bonferroni), Experiment Families and Statistical Comparison, Real-Coded Genetic Algorithm (tournament + elitism)

### Community 19 - "Batched Inference and Compute"
Cohesion: 0.38
Nodes (7): Batched Population Inference (weight-stacked torch.bmm, Sec 12.3), Compute Budget Estimate from Phase 1.5 Benchmark (Sec 19.6, fixes C7), neural/batched_inference.py (stacked weights + torch.bmm), Phase 1.5 - Batched Population Inference Prototype and Benchmark, Phase 2 - Sensors and Fixed Neural Controller (Batched), simulation/engine.py (main loop orchestration), Population-Wide Batched Inference (Phase 1.5 criterion)

### Community 20 - "Integration Reset Tests"
Cohesion: 0.29
Nodes (6): simulation_environment, Integration test for environment reset., Reset the environment N times and verify entity counts match config., Test reset with different configurations., test_reset_environment(), test_reset_with_different_configs()

### Community 21 - "Reproducibility and Related Work"
Cohesion: 0.33
Nodes (6): Stanley and Miikkulainen NEAT (2002), CPU/GPU Reproducibility Tiers (Sec 15.4, fixes C5), Sims (1994) Evolved Virtual Creatures, Phase 4 - Analytics, Checkpoints, Determinism, Phase 4.5 - Related Work and Statistical Protocol, Reproducibility Tiers (cpu-deterministic vs gpu)

### Community 22 - "Vector Limits and Normalize"
Cohesion: 0.40
Nodes (3): Calculate the magnitude (length) of the vector., Return a normalized version of this vector (length = 1)., Return a vector with magnitude limited to max_mag.

## Knowledge Gaps
- **30 isolated node(s):** `artificial-life-neuroevolution-simulation`, `CI Pipeline`, `flake8 Lint Stage`, `pytest Test Stage`, `NumPy (>=1.24.0)` (+25 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 209 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **45 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Vector2D` connect `Vector2D Operations` to `Physics Core and Engine`, `Spatial Hash Rationales`, `AABB Collision Geometry`, `Organism Lifecycle`, `Vector Limits and Normalize`, `Vector Angle and Dot`, `Organism Food Consumption`, `Organism Initialization`, `Organism State Loading`, `Apply Force Physics`, `Apply Impulse Physics`, `Momentum Query`, `Vector From Tuple`, `Vector To Tuple`, `Vector Scalar Multiply`, `Vector Magnitude Squared`, `Vector Distance To`, `Vector Distance Squared`, `Vector Distance Tests`, `Vector Limits Tests`, `Vector Equality Tests`, `Vector Operations Tests`, `Vector Dot Tests`?**
  _High betweenness centrality (0.241) - this node is a cross-community bridge._
- **Why does `World` connect `World Distance Queries` to `Environment and Obstacle Tests`, `Config Validation Models`, `Organism Lifecycle`, `App Render Loop`, `Renderer Implementation`, `Food and Reset Queries`, `Integration Reset Tests`?**
  _High betweenness centrality (0.093) - this node is a cross-community bridge._
- **Why does `SpatialHash` connect `Spatial Hash Rationales` to `Physics Core and Engine`, `AABB Collision Geometry`, `Vector2D Operations`, `Organism Lifecycle`?**
  _High betweenness centrality (0.087) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `Vector2D` (e.g. with `AABB` and `PhysicsObject`) actually correct?**
  _`Vector2D` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `SpatialHash` (e.g. with `AABB` and `Vector2D`) actually correct?**
  _`SpatialHash` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `AABB` (e.g. with `Vector2D` and `SpatialHash`) actually correct?**
  _`AABB` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `Artificial Life Simulation Design Document v2.0 (Revised, fixes C1-C11)` (e.g. with `PyGame (>=2.5.0)` and `PyTorch (>=2.0.0)`) actually correct?**
  _`Artificial Life Simulation Design Document v2.0 (Revised, fixes C1-C11)` has 2 INFERRED edges - model-reasoned connections that need verification._