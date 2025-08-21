# OLS - Ordinary Least Squares in Rust


## Project Structure

```
ols/
├── src/
│   ├── lib.rs              # PyO3 bindings and module definition
│   └── ols_module/
│       ├── mod.rs          # Module declaration
│       └── ols.rs          # Core OLS implementation
├── Cargo.toml              # Rust dependencies
├── pyproject.toml          # Python package configuration
└── README.md               # This file
```

## Development Setup

### Prerequisites
- Rust (latest stable)
- Python 3.8+
- Maturin: `pip install maturin`

### Building for Development
```bash
# Build and install in development mode
maturin develop

# Or build without installing
maturin build
```

### Testing in Python
```python
import ols

# Create an OLS model
model = ols.PyOLS(dim=3)
print(model)  # OLS(weights=[0.0, 0.0, 0.0], is_fitted=False)
```

## Current Status
- ✅ Basic OLS struct with PyO3 bindings
- ✅ Project structure for Maturin
- 🔄 OLS implementation (fit, predict methods)
- 🔄 Python API design

