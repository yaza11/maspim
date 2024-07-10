from setuptools import setup

setup(
    name='msi_workflow',
    version='1.0',
    packages=['res', 'data', 'util', 'imaging', 'imaging.main', 'imaging.misc', 'imaging.test',
              'imaging.util', 'imaging.XRay', 'imaging.register', 'imaging.align_net', 'Project',
              'exporting', 'exporting.legacy', 'exporting.from_mcf', 'exporting.from_sqlite',
              'exporting.sqlite_mcf_communicator', 'timeSeries'],
    url='https://github.com/weimin-liu/msi_workflow',
    license='',
    author='Yannick Zander, Weimin Liu',
    author_email='yzander@marum.de',
    description=''
)
