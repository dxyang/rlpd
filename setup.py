from setuptools import find_packages, setup

# Required dependencies
required = [
]


# Development dependencies
extras = dict()
extras['dev'] = [
]


setup(
    name='rlpd',
    packages=['rlpd'],
    include_package_data=True,
    install_requires=required,
    extras_require=extras,
)
