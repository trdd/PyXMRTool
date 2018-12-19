from distutils.core import setup

setup(
    name='PyXMRTool',
    version='0.9dev',
    packages=['PyXMRTool',],
    license='GNU Lesser General Public License v3.0',
    long_description=open('README.md').read(),
    package_data={'PyXMRTool': ['resources/ChantlerTables/*.cff','resources/ChantlerTables/*.pyt']}
)