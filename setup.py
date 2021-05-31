from setuptools import setup
from setuptools_rust import RustExtension

if __name__ == "__main__":
    setup(
        rust_extensions=[RustExtension("flowrect.accelerated", "Cargo.toml", debug=False)],
    )
