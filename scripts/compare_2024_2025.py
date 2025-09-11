from canopy_factory import utils


datafile_2024 = 'B73_rdla_2024.json'
datafile_2025 = 'B73_PHKW3_rdla_2025.json'
fname = 'compare_2024_vs_2025.png'

x_2024 = utils.DataProcessor.from_file(datafile_2024)
x_2025 = utils.DataProcessor.from_file(datafile_2025)
x_2024.plot(fname=fname, other=x_2025)
