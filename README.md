# Automated Hardware-Aware Analysis and Evaluation of Quantum Error-Correction Circuits

Sandbox repository for my Interdisciplinary Project in an Application Subject.

# Idea

Give me a QEC circuit and a hardware platform (with some parameters), and I'll tell you how well suited this circuit is for the platform. "Well suited" here means a combination of compilation cost, runtime cost, and how the hardware-specific noise disrupts the circuit in its quality/effectiveness.

THIS IS NOT A COMPILER: I want to be as compiler-agnostic as possible (to stay applicable to many use cases), while still capturing the inherent properties and restrictions that the respective platforms carry. For this, I want to construct abstract hardware models that find a good balance between these aspects. 

## Scope

Supported Quantum Error-Correction Circuits:
- Encoding
- Logical-State Preparation
- Syndrome Extraction

Supported Hardware Platforms:
- Trapped Ions
- Neutral Atoms
- Superconducting Qubits


