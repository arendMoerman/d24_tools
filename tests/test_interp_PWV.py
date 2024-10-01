import numpy as np
import unittest
from nose2.tools import params

from d24_tools.atmosphere import interp_PWV

class TestModels(unittest.TestCase):
   obsid0 = "20240815123010" # Random, middle of day 

   def test_interp_PWV(self):
       interp_PWV(self.obsid0, obsid=True)


if __name__ == "__main__":
    import nose2
    nose2.main()
