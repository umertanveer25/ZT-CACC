# Contributing to Zero-Trust CAV Platooning

Thank you for your interest in contributing to the **Zero-Trust CAV Platooning Framework**! This open-source repository is built to advance resilient cooperative automated driving research.

## Development Setup

1. Fork and clone the repository:
   ```bash
   git clone https://github.com/your-username/zero-trust-cav.git
   cd zero-trust-cav
   ```

2. Create a clean virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install -e .
   ```

3. Run the unit test suite to verify your setup:
   ```bash
   python -m unittest discover tests
   ```

## Code Guidelines

- **PEP 8**: Follow standard Python conventions.
- **Type Hinting**: Provide explicit type hints for all public functions and methods.
- **Modularity**: Place core vehicle kinematics in `src/zero_trust_cav/core/`, detection algorithms in `src/zero_trust_cav/detection/`, and simulation utilities in `src/zero_trust_cav/simulation/`.
- **Testing**: Add corresponding unit tests in `tests/` whenever introducing a new feature or control law.

## Submitting a Pull Request

1. Create a descriptive feature branch (`git checkout -b feature/new-resilient-controller`).
2. Commit your changes with clear messages (`git commit -m "feat: add sliding mode resilient CACC controller"`).
3. Ensure all tests and benchmarks pass (`python -m unittest discover tests`).
4. Open a Pull Request on GitHub and follow the PR checklist.
