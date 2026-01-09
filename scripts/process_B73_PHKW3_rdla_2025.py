from canopy_factory import utils


crop = 'maize'
metadata = {
    'year': '2025',
    'planting_date': '5/27/2025',
    'authors': ['Matthew Runyon'],
}
units = {
    'length': 'cm',
    'mass': 'kg',
    'time': 'days',
    'angle': 'degrees',
}

x = utils.DataProcessor(crop, metadata=metadata, units=units)

# File containing leaf data
datafile = 'B73 PHKW3 rdla Paired Rows.csv'
regex = r'(?P<component>[a-zA-Z]+)\s+(?P<n>\d+)\s+(?P<parameter>[a-zA-Z]+)'
x.process_csv(datafile, regex=regex)

# File containing internode lengths
datafile = 'B73 PHKW3 Paired Rows Internode Length.csv'
regex = r'(?P<component>[a-zA-Z]+)\s+(?P<n>\d+)'
x.process_csv(datafile, regex=regex, parameter='Length')

# File containing leaf angles
ear_leaf_n = 11
datafile = 'B73 rdla Leaf Angle.csv'
regex = r'(?P<parameter>[a-zA-Z]+)\s+Ear\s+(?P<n>[-+]\s+\d+)'
x.process_csv(datafile, regex=regex, component='Leaf',
              noffset=ear_leaf_n)
# Convert angles to relative to vertical (measured wrt horizontal)
for idstr in x.data.keys():
    if 'LeafAngle' not in x.data[idstr]:
        continue
    for n in list(x.data[idstr]['LeafAngle'].keys()):
        x.data[idstr]['LeafAngle'][n] = 90 - x.data[idstr]['LeafAngle'][n]

x.write()
