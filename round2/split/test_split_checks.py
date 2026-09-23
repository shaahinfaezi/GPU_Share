"""Offline regressions for split admission; no Kubernetes or GPU calls."""
import unittest
from split_checks import memory_budget, same_two_devices


class AdmissionTests(unittest.TestCase):
    def test_historical_solo_success_is_not_shared_admission(self):
        cal = {'solo': {'tp': {'vllm_rss_mib': 6575},
                        'pp': {'vllm_rss_mib': 6162}}}
        self.assertFalse(memory_budget(cal, 9000)['ok'])
        self.assertTrue(memory_budget(cal, 14000)['ok'])

    def test_unknown_rss_does_not_silently_use_single_rank_default(self):
        with self.assertRaises(ValueError):
            memory_budget({'solo': {'tp': {}, 'pp': {}}}, 16000)

    def test_one_tenant_on_both_cards_is_not_shared_placement(self):
        self.assertFalse(same_two_devices({'tp': [], 'pp': ['GPU-a', 'GPU-b']}))
        self.assertFalse(same_two_devices({'tp': ['GPU-a', 'GPU-b'],
                                           'pp': ['GPU-b', 'GPU-c']}))
        self.assertTrue(same_two_devices({'tp': ['GPU-a', 'GPU-b'],
                                          'pp': ['GPU-b', 'GPU-a']}))


if __name__ == '__main__':
    unittest.main()
