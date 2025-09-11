from canopy_factory import utils


datafile = 'B73_WT_vs_rdla_Paired_Rows.csv'
crop = 'maize'
genotype = 'B73'
metadata = {
    'year': '2024',
    'authors': ['Matthew Runyon'],
}
units = {
    'length': 'cm',
    'mass': 'kg',
    'time': 'days',
    'angle': 'degrees',
}

x = utils.DataProcessor(crop, metadata=metadata, units=units)
x.process_csv(datafile, genotype=genotype, component='Leaf')
x.write()
