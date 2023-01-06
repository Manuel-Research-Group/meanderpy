YEAR = 365*24*60*60.0

# NUMBER_OF_LAYERS_PER_EVENT: number of materials plus 1. This extra 1 comes from the eroded surface before aggradation.
# We then deposit (aggradate) a certain number of materials. For instance, when using gravel, sand and silt, the number of materials == 3.
# When using gravel, gross sand, medium sand, fine sand, and silt, then number of materials == 5.
# When using gravel, very gross sand, gross sand, medium sand, fine sand, very fine sand, and silt, then number of materials == 7.
NUMBER_OF_LAYERS_PER_EVENT = 9 # there are 7 layers + previous surface + separator
BIG_NUMBER = 10**9