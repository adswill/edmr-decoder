# Copyright (c) 2026 Kristijonas Adomaitis (adswill). All rights reserved.
# edmr-decoder Source Licence 1.1: no copying, redistribution or porting. See LICENSE.
import unittest

from edmr.selftest import run


class SelfTest(unittest.TestCase):
    def test_generated_signals(self):
        self.assertEqual(run(verbose=False), 0)


if __name__ == "__main__":
    unittest.main()
