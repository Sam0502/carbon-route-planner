import unittest
from test import intercept
import math

class TestOneEd(unittest.TestCase):
    def test_1(self):
        roads = [(0, 2, 10, 3), (1, 2, 5, 2), (2, 1, 15, 5), (2, 0, 12, 10)]
        stations = [(0, 5), (1, 5)]
        start = 2
        friendStart = 0
        expected_output = (12, 10, [2, 0])
        result = intercept(roads, stations, start, friendStart)
        self.assertEqual(result, expected_output)

    def test_2(self):
        stations_input = [(i, 2) for i in range(20)]
        start_input = 20
        friendStart_input = 0
        roads_input = [(20, 2, 7, 4), (20, 18, 30, 36)] + \
                      [(i, (i+1) % 20, 1, 2) for i in range(20)]
        expected_output = (7, 4, [20, 2])
        result = intercept(roads_input, stations_input, start_input, friendStart_input)
        self.assertEqual(result, expected_output)

    def test_3(self):
        stations_input = [(i, 5) for i in range(20)]
        start_input = 20
        friendStart_input = 0
        roads_input = [(20, 5, 30, 25), (20, 0, 40, 100)] + \
                      [(i, (i+1) % 20, 1, 1) for i in range(20)]
        expected_output = (30, 25, [20, 5])
        result = intercept(roads_input, stations_input, start_input, friendStart_input)
        self.assertEqual(result, expected_output)

    def test_4(self):
        roads = [(0, 1, 1, 1), (1, 0, 1, 1),
                 (2, 3, 1, 1), (3, 2, 1, 1)]
        stations = [(0, 5), (1, 5)]
        start = 2
        friendStart = 0
        result = intercept(roads, stations, start, friendStart)
        self.assertIsNone(result)

    def test_5(self):
        roads = [(0, 1, 10, 3), (1, 2, 20, 10), (2, 0, 8, 4)]
        stations = [(0, 5), (1, 5), (2, 5)]
        start = 1
        friendStart = 0
        expected_output = (20, 10, [1, 2])
        result = intercept(roads, stations, start, friendStart)
        self.assertEqual(result, expected_output)


class TestTwoEd(unittest.TestCase):
    def test_case_set2(self):
        a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        stations = [(4, 3), (5, 3), (6, 3)]
        start = 1
        friendStart = 6
        roads = [None] * len(a)

        for number in a:
            roads[number - 1] = [
                (1, 2, 1, 3), (2, 3, 1, 3), (3, 1, 1, 3),
                (1, 4, 1, number), (4, 5, 1, 3), (5, 6, 1, 3), (6, 4, 1, 3)
            ]

        self.assertIsNone(intercept(roads[0], stations, start, friendStart))
        self.assertIsNone(intercept(roads[1], stations, start, friendStart))
        self.assertEqual(intercept(roads[2], stations, start, friendStart), (1, 3, [1, 4]))
        self.assertIsNone(intercept(roads[3], stations, start, friendStart))
        self.assertIsNone(intercept(roads[4], stations, start, friendStart))
        self.assertIsNone(intercept(roads[5], stations, start, friendStart))
        self.assertIsNone(intercept(roads[6], stations, start, friendStart))
        self.assertIsNone(intercept(roads[7], stations, start, friendStart))
        self.assertIsNone(intercept(roads[8], stations, start, friendStart))
        self.assertIsNone(intercept(roads[9], stations, start, friendStart))
        self.assertIsNone(intercept(roads[10], stations, start, friendStart))
        self.assertEqual(intercept(roads[11], stations, start, friendStart), (1, 12, [1, 4]))


class TestRubric(unittest.TestCase):
    def test_example_1(self):
        roads = [
            (6, 0, 3, 1), (6, 7, 4, 3), (6, 5, 6, 2), (5, 7, 10, 5), (4, 8, 8, 5),
            (5, 4, 8, 2), (8, 9, 1, 2), (7, 8, 1, 3), (8, 3, 2, 3), (1, 10, 5, 4),
            (0, 1, 10, 3), (10, 2, 7, 2), (3, 2, 15, 2), (9, 3, 2, 2), (2, 4, 10, 5)
        ]
        stations = [(0, 1), (5, 1), (4, 1), (3, 1), (2, 1), (1, 1)]
        start = 6
        friendStart = 0
        expected_output = (7, 9, [6, 7, 8, 3])
        result = intercept(roads, stations, start, friendStart)
        self.assertEqual(result, expected_output)

    def test_example_2(self):
        roads = [
            (0, 1, 35, 3), (1, 2, 5, 2), (2, 0, 35, 4), (0, 4, 10, 1), (4, 1, 22, 2),
            (1, 5, 65, 1), (5, 2, 70, 1), (2, 3, 10, 1), (3, 0, 20, 3)
        ]
        stations = [(4, 3), (5, 2), (3, 4)]
        start = 0
        friendStart = 4
        result = intercept(roads, stations, start, friendStart)
        self.assertIsNone(result)

    def test_example_3(self):
        roads = [
            (0, 1, 35, 7), (1, 2, 5, 4), (2, 0, 35, 6), (0, 4, 10, 5), (4, 1, 22, 3),
            (1, 5, 60, 4), (5, 3, 70, 2), (3, 0, 10, 7)
        ]
        stations = [(4, 2), (5, 1), (3, 4)]
        start = 0
        friendStart = 3
        expected_output = (160, 39, [0, 1, 2, 0, 1, 2, 0, 4])
        result = intercept(roads, stations, start, friendStart)
        self.assertEqual(result, expected_output)

    def test_example_4(self):
        roads = [
            (0, 1, 10, 7), (0, 2, 10, 3), (2, 0, 1, 4), (1, 0, 1, 7)
        ]
        stations = [(2, 4), (1, 3)]
        start = 0
        friendStart = 1
        expected_output = (10, 3, [0, 2])
        result = intercept(roads, stations, start, friendStart)
        self.assertEqual(result, expected_output)
        
    def test_miniloop(self):
        roads = [(0, 1, 20, 3), (1, 2, 10, 2), (2, 0, 15, 4), (0, 3, 50, 2), (3, 4, 10, 1), (4, 5, 10, 1), (5, 6, 10, 1), (6, 1, 30, 2)]
        stations = [(3, 2), (6, 1), (1, 4)]   
        start = 0
        friendStart = 6
        self.assertEqual(intercept(roads, stations, start, friendStart), (125, 14, [0, 1, 2, 0, 3, 4, 5, 6]))
        
        
    def test_linear_path(self):
        """
        * A straight path, driver must travel along a linear road 0->1->2->3->4
        
        ! There is also a bad shortcut from 1->4 which will give too high of a
        ! cost, even though it has a time of 0 to travel.
        !  The driver should skip this "bad shortcut".
        """
        roads = [(0,1,5,2), (1,2,5,2), (2,3,5,2), (3,4,5,2), (1,4,30,2), (4,0,5,2)]
        stations = [(4,2), (2,2)]
        start = 0
        friend_start = 4
        
        self.assertEqual(intercept(roads, stations, start, friend_start), 
                         (20, 8, [0,1,2,3,4]))

    def test_branch_choice(self):
        """
        * Need to choose between two paths, one clearly better than the other
        
        ! The driver must choose to take the shortcut from 1->4.
        ! This shortcut has both better cost & time compared to 1->2->3->4.
        """
        # Need to choose between two paths, one clearly better than the other
        roads = [(0,1,5,2), (1,2,5,2), (2,3,5,2), (3,4,5,2), (1,4,14,2)]
        stations = [(4,2), (2,2)]
        start = 0
        friend_start = 4
        
        self.assertEqual(intercept(roads, stations, start, friend_start), 
                         (19,4,[0,1,4]))

    def test_repeatedV2(self):
        roads = [(0, 1, 20, 3), (1, 2, 10, 2), (2, 0, 15, 4), (0, 3, 50, 2), (3, 4, 10, 1), (4, 5, 10, 1), (5, 6, 10, 1), (6, 1, 30, 2)]
        stations = [(3, 2), (6, 1), (1, 4)]   
        start = 0
        friendStart = 6

        self.assertEqual(intercept(roads, stations, start, friendStart), (125, 14, [0, 1, 2, 0, 3, 4, 5, 6]))
        
    def test_long_chase(self):
            import sys
            rec_lim = sys.getrecursionlimit()
            sys.setrecursionlimit(2000)
            roads = [(i, i+1, 3, 5) for i in range(19)] + [(19, 0, 3, 4)]
            stations = [(i, 5) for i in range(20)]
            start = 0
            friend_start = 19

            #1900 edges traversed/95 cycles
            self.assertEqual(intercept(roads, stations, start, friend_start), (5700, 9405, [i for i in range(20)]*95+[0]))
            # sys.setrecursionlimit(rec_lim)
      

class TestAdditionalInterceptCases(unittest.TestCase):

    def test1(self):
        roads = []
        stations = [(0, 5), (1, 5)]
        start = 0
        friendStart = 0
        output = (0, 0, [0])
        result = intercept(roads, stations, start, friendStart)
        self.assertEqual(result, output)

    def test2(self):
        roads = [
            (0, 1, 1, 3), (0, 2, 1, 5), (1, 3, 2, 3), (2, 3, 2, 2)
        ]
        stations = [(3, 2), (0, 2)]
        start = 0
        friendStart = 3
        result = intercept(roads, stations, start, friendStart)
        self.assertIsNone(result)

    def test3(self):
        roads = [(0, 1, 5, 5)]
        stations = [(1, 3), (0, 3)]
        start = 0
        friendStart = 1
        result = intercept(roads, stations, start, friendStart)
        self.assertIsNone(result)

    def test4(self):
        roads = [(0, 1, 2, 2), (1, 2, 2, 2), (2, 0, 2, 2)]
        stations = [(2, 3), (0, 3)]
        start = 0
        friendStart = 0
        output = (0, 0, [0])
        result = intercept(roads, stations, start, friendStart)
        self.assertEqual(result, output)

    def test5(self):
        roads = [(0, 1, 2, 2)]
        stations = [(3, 1), (4, 1)]
        start = 0
        friendStart = 3
        result = intercept(roads, stations, start, friendStart)
        self.assertIsNone(result)

    def test6(self):
        roads = [(0, 1, 2, 2), (1, 2, 2, 2)]
        stations = [(5, 2), (6, 2)]
        start = 0
        friendStart = 5
        result = intercept(roads, stations, start, friendStart)
        self.assertIsNone(result)

    def test7(self):
        roads = [(0, 1, 2, 2)]
        stations = [(1, 5), (0, 5)]
        start = 0
        friendStart = 1
        result = intercept(roads, stations, start, friendStart)
        self.assertIsNone(result)

    def test8(self):
        roads = [(0, 1, 3, 6)]
        stations = [(1, 2), (2, 2), (0, 2)]
        start = 0
        friendStart = 1
        output = (3, 6, [0, 1])
        result = intercept(roads, stations, start, friendStart)
        self.assertEqual(result, output)

    def test9(self):
        roads = [
            (0, 1, 2, 2), (1, 2, 3, 2), (2, 3, 1, 2), (3, 4, 2, 4), (4, 5, 1, 5),
            (0, 6, 10, 4), (6, 7, 1, 1), (7, 8, 1, 1), (8, 5, 1, 1),
            (1, 9, 5, 2), (9, 10, 2, 1), (10, 5, 2, 1),
            (2, 11, 3, 1), (11, 12, 2, 1), (12, 13, 3, 1), (13, 5, 4, 1),
            (3, 14, 2, 1), (14, 5, 2, 1), (0, 10, 20, 6), (6, 12, 15, 4)
        ]
        stations = [(5, 3), (13, 3), (10, 3), (0, 3), (1, 3)]
        start = 0
        friendStart = 5

        output = (9, 15, [0, 1, 2, 3, 4, 5])
        result = intercept(roads, stations, start, friendStart)
        self.assertEqual(result, output)

class TestInterceptEdgeCases(unittest.TestCase):
    def test_cas_direct_interception_next_station(self):
        roads = [(0, 1, 1, 1), (1, 0, 1, 1), (2, 1, 5, 1)]
        stations = [(0, 1), (1, 1)]
        start = 2
        friendStart = 0
        expected_output = (5, 1, [2, 1])
        result = intercept(roads, stations, start, friendStart)
        self.assertEqual(result, expected_output)

    def test_case_loop_alignment(self):
        roads = [(0, 1, 35, 3), (1, 0, 1, 1), (2, 0, 10, 5)]
        stations = [(0, 3), (1, 2)]
        start = 2
        friendStart = 0
        expected_output = (10, 5, [2, 0])
        result = intercept(roads, stations, start, friendStart)
        self.assertEqual(result, expected_output)

    def test_case_multi_loop_interception(self):
        roads = [(0, 1, 1, 2), (1, 2, 1, 3), (2, 0, 1, 5), (3, 0, 10, 10)]
        stations = [(0, 2), (1, 3), (2, 5)]
        start = 3
        friendStart = 0
        expected_output = (10, 10, [3, 0])
        result = intercept(roads, stations, start, friendStart)
        self.assertEqual(result, expected_output)

    def test_case_minimal_valid_config(self):
        roads = [(0, 1, 1, 1), (1, 0, 5, 2)]
        stations = [(0, 1), (1, 1)]
        start = 1
        friendStart = 0
        expected_output = (5, 2, [1, 0])
        result = intercept(roads, stations, start, friendStart)
        self.assertEqual(result, expected_output)

    def test_case_multi_station_interception(self):
        roads = [(0, 1, 1, 2), (1, 2, 1, 3), (2, 0, 1, 5), (3, 2, 10, 5), (3, 0, 5, 1)]
        stations = [(0, 2), (1, 3), (2, 5)]
        start = 3
        friendStart = 0
        expected_output = (10, 5, [3, 2])
        result = intercept(roads, stations, start, friendStart)
        self.assertEqual(result, expected_output)

    def test_case_three_station_loop(self):
        roads = [(0, 1, 1, 1), (1, 2, 1, 2), (2, 0, 1, 3), (3, 2, 10, 3), (3, 0, 5, 1)]
        stations = [(0, 1), (1, 2), (2, 3)]
        start = 3
        friendStart = 0
        expected_output = (10, 3, [3, 2])
        result = intercept(roads, stations, start, friendStart)
        self.assertEqual(result, expected_output)

    def test_case_multi_hop_interception(self):
        roads = [(3, 2, 5, 2), (0, 1, 1, 1), (1, 2, 1, 1), (2, 0, 1, 1), (3, 0, 10, 1)]
        stations = [(0, 1), (1, 1), (2, 1)]
        start = 3
        friendStart = 0
        expected_output = (5, 2, [3, 2])
        result = intercept(roads, stations, start, friendStart)
        self.assertEqual(result, expected_output)

class TestAdditionalInterceptions(unittest.TestCase):
    def test_cycle_detour(self):
        roads = [(0, 1, 3, 1), (1, 2, 3, 1), (2, 0, 3, 1), (0, 3, 10, 5), (2, 4, 2, 1), (3, 4, 1, 1), (4, 0, 5, 2)]
        stations = [(3, 2), (4, 2)]
        start = 0
        friendStart = 3
        expected_output = (11, 6, [0, 3, 4])
        result = intercept(roads, stations, start, friendStart)
        self.assertEqual(result, expected_output)

    def test_indirect_route(self):
        roads = [(0, 1, 1, 1), (1, 2, 1, 1), (2, 0, 5, 2), (1, 0, 2, 1), (2, 1, 2, 1)]
        stations = [(1, 2), (2, 3)]
        start = 0
        friendStart = 1
        expected_output = (2, 2, [0, 1, 2])
        result = intercept(roads, stations, start, friendStart)
        self.assertEqual(result, expected_output)

    def test_multiple_stations_choice(self):
        roads = [(0, 1, 3, 2), (1, 2, 3, 2), (2, 3, 3, 2), (0, 4, 6, 3), (4, 5, 1, 1), (5, 3, 1, 1), (3, 0, 10, 2)]
        stations = [(3, 2), (5, 1)]
        start = 0
        friendStart = 5
        expected_output = (26, 12, [0, 1, 2, 3, 0, 4, 5])
        result = intercept(roads, stations, start, friendStart)
        self.assertEqual(result, expected_output)

    def test_detour_lower_cost(self):
        roads = [(0, 1, 10, 2), (0, 2, 100, 1), (1, 2, 5, 1), (2, 3, 10, 2), (1, 3, 50, 5), (3, 4, 2, 1), (2, 4, 30, 2), (4, 0, 5, 2)]
        stations = [(3, 2), (4, 2)]
        start = 0
        friendStart = 3
        expected_output = (27, 6, [0, 1, 2, 3, 4])
        result = intercept(roads, stations, start, friendStart)
        self.assertEqual(result, expected_output)

    def test_revisiting_vertices(self):
        roads = [(0, 1, 2, 1), (1, 2, 2, 1), (2, 3, 2, 1), (3, 1, 1, 1), (3, 4, 2, 1), (4, 5, 2, 1), (5, 0, 10, 3), (2, 5, 5, 2)]
        stations = [(3, 2), (5, 1)]
        start = 0
        friendStart = 3
        expected_output = (6, 3, [0, 1, 2, 3])
        result = intercept(roads, stations, start, friendStart)
        self.assertEqual(result, expected_output)

    def test_multiple_loops_required(self):
        roads = [(0, 1, 5, 2), (1, 2, 5, 2), (2, 3, 5, 2), (3, 0, 5, 2), (3, 4, 1, 1), (4, 0, 10, 3), (1, 3, 10, 4)]
        stations = [(3, 2), (4, 3)]
        start = 0
        friendStart = 3
        expected_output = (16, 7, [0, 1, 3, 4])
        result = intercept(roads, stations, start, friendStart)
        self.assertEqual(result, expected_output)

class EricTestCase(unittest.TestCase):
    def test_case2_multiple_stations_trick1(self):
        roads = [
            (0, 4, 1, 1),
            (4, 1, 1, 3),
            (1, 7, 2, 5),  
            (0, 1, 3, 4),
            (1, 2, 2, 2),
            (2, 7, 1, 4),
            (0, 3, 2, 3),
            (3, 5, 2, 6)
        ]
        stations = [(2, 1), (5, 3), (7, 2)]
        start = 0
        friendStart = 5 
        res = intercept(roads, stations, start, friendStart)
        expected_output = (4, 9, [0, 4, 1, 7])
        self.assertIsNotNone(expected_output, res)
    
    def test_case3_direct_interception(self):
        roads = [
            (0, 9, 4, 5),
            (0, 3, 2, 2),
            (3, 9, 10, 3),
            (0, 1, 1, 1),
            (1, 2, 2, 2),
            (2, 8, 1, 3),
            (8, 9, 2, 1),
            (0, 4, 3, 2),
            (4, 9, 1, 3)
        ]
        stations = [(2, 3), (4, 2), (6, 4), (7, 5), (9, 1)]
        start = 0
        friendStart = 7  
        res = intercept(roads, stations, start, friendStart)
        expected_output = (4, 5, [0, 9])
        self.assertIsNotNone(expected_output, res)
    
    def test_case4_late_cycle_interception(self):
        roads = [
            (0, 1, 2, 4),
            (1, 3, 2, 3),
            (3, 5, 4, 4),
            (5, 9, 3, 9), 
            (0, 2, 3, 3),
            (2, 9, 2, 4)
        ]
        stations = [(2, 5), (8, 5), (9, 5)]
        start = 0
        friendStart = 8
        
        expected_output = (11, 20, [0, 1, 3, 5, 9])
        res = intercept(roads, stations, start, friendStart)
        self.assertEqual(res, expected_output)
    
    def test_case5_max_station_cycle(self):
        stations = [
            (10, 1), (11, 2), (12, 3), (13, 4), (14, 5),
            (15, 1), (16, 2), (17, 3), (18, 4), (19, 5),
            (20, 1), (21, 2), (22, 3), (23, 4), (24, 5),
            (25, 1), (26, 2), (27, 3), (28, 4), (29, 5)
        ]
        roads = [
            (0, 29, 7, 15), 
            (0, 2, 3, 7),
            (2, 29, 2, 8), 
            (0, 10, 6, 16),
            (10, 11, 1, 2),
            (11, 12, 1, 2)
        ]
        start = 0
        friendStart = 24
        res = intercept(roads, stations, start, friendStart)
        expected_output = (5, 15, [0, 2, 29])
        self.assertEqual(expected_output, res)
    
    def test_case6_multiple_candidate_paths(self):
        roads = [
            (0, 2, 2, 4),
            (2, 7, 1, 5),
            (0, 1, 4, 6),
            (1, 3, 0, 4),
            (0, 4, 10, 3),
            (4, 7, 0, 6)
        ]
        stations = [(3,2), (5,3), (7,1)]
        start = 0
        friendStart = 5
        res = intercept(roads, stations, start, friendStart)
        expected_output = (3, 9, [0, 2, 7])
        self.assertEqual(expected_output, res)

class ExtraTestCases(unittest.TestCase):
    def test_loop_finish(self):
        roads = [(1,5,2,1),(1,6,11,6),(2,5,6,3),(3,7,1,10),(4,2,4,1),(5,3,8,8),(5,4,20,2),(5,1,4,1),(6,2,9,6),(7,6,15,2)]
        stations = [(1,3),(2,3),(3,5)]
        start = 1
        friend_start = 2

        self.assertEqual(intercept(roads, stations, start, friend_start), (24, 8, [1,5,1,5,1,5,1,5,1]))
    
    def test_station_chase(self):
        roads = [(1,2,2,1),(1,3,50,2),(2,3,3,2),(3,4,5,2),(3,5,15,3),(4,5,1,2),(5,2,4,1)]
        stations = [(1,1),(2,2),(3,2),(4,2),(5,1)]
        start = 1
        friend_start = 2

        self.assertEqual(intercept(roads, stations, start, friend_start), (15, 8, [1,2,3,4,5,2]))
    
    def test_equal_costs(self):
        roads = [(1,2,3,3),(2,5,1,1),(3,4,5,2),(3,1,11,4),(3,2,10,7),(3,5,5,7),(4,1,5,2),(5,2,5,7)]
        stations = [(1,3),(2,4)]
        start = 3
        friend_start = 2

        self.assertEqual(intercept(roads, stations, start, friend_start), (10, 4, [3,4,1]))
    
    def test_valid_path(self):
        roads = [(1,5,1,1),(2,5,4,2),(3,2,3,3),(3,4,2,1),(4,2,1,1),(5,1,5,1)]
        stations = [(1,2),(2,3)]
        start = 3
        friend_start = 1

        self.assertEqual(intercept(roads, stations, start, friend_start), (3,2,[3,4,2]))

class SkooyTestCases(unittest.TestCase):
    def test_alot_of_updates(self):
        roads = [(3, 1, 50, 4), (3, 4, 40, 2), (4, 1, 9, 2), (4, 5, 2, 2), (5, 6, 5, 2), (5, 1, 2, 4), (6, 1, 20, 2), (2, 3, 10, 2)]
        stations = [(1, 2), (2, 2)]
        friendStart = 1
        start = 3
        res = intercept(roads, stations, start, friendStart)
        expected_output = (44, 8, [3, 4, 5, 1])
        self.assertEqual(res, expected_output)
    
    def test_hi(self):
        mult = 1000000000000
        roads = [(0,1,9223372036854775807,7), (0,2,10,3 + 7 * mult), (2,0,1,4), (1,0,1,7)]
        stations = [(2,4), (1,3)]
        start = 0
        friendStart = 1

        self.assertEqual(intercept(roads, stations, start, friendStart), (10, 3 + 7 * mult, [0,2]))

    def test_hi2(self):
        mult = 1000000000000
        roads = [(0,1,10,7 + 7 * 5000), (0,2,10,3 + 7 * mult), (2,0,1,4), (1,0,1,7)]
        stations = [(2,4), (1,3)]
        start = 0
        friendStart = 1

        self.assertEqual(intercept(roads, stations, start, friendStart), (10, 7 + 7 * 5000, [0,1]))
    
    def test_double_road(self):
        roads = [(0, 1, 5, 5), (0, 1, 5, 5), (1, 2, 5, 5), (2, 0, 5, 5)]
        stations = [(2, 5), (1, 5)]
        start = 0
        friend_start = 2
        
        self.assertEqual(intercept(roads, stations, start, friend_start), (5, 5, [0,1]))

    def test_double_road_all(self):
        roads = [(0, 1, 5, 5), (0, 1, 5, 5), (0, 1, 6, 5), (0, 1, 5, 6), (1, 2, 5, 5), (2, 0, 5, 5)]
        stations = [(2, 5), (1, 5)]
        start = 0
        friend_start = 2
        
        self.assertEqual(intercept(roads, stations, start, friend_start), (5, 5, [0,1]))

    def test_short_none(self):
        roads = [(0, 1, 2, 1), (1, 0, 2, 1)]
        stations = [(0, 5), (1, 5)]
        start = 0
        friend_start = 1

        self.assertIsNone(intercept(roads, stations, start, friend_start))

    def test_short_long(self):
        roads = [(0, 1, 2, 1), (1, 0, 2, 1)]
        stations = [(0, 4), (1, 5)]
        start = 0
        friend_start = 1

        self.assertEqual(intercept(roads, stations, start, friend_start), (18, 9, [0,1]*5))


class TestIntercept:
    def __init__(self, roads, stations, start, friend_start, expected):
        self.roads = roads
        self.stations = stations
        self.start = start
        self.friend_start = friend_start
        self.expected = expected
    def test(self):
        result = intercept(self.roads, self.stations, self.start, self.friend_start) == self.expected
        print("-------final output for test case:------",
              (intercept(self.roads, self.stations, self.start, self.friend_start)))
        return result
    
def test_intercept():
    tests = [
        # example 1, Simple
        TestIntercept(
            roads=[(6, 0, 3, 1), (6, 7, 4, 3), (6, 5, 6, 2), (5, 7, 10, 5), (4, 8, 8, 5), (5, 4, 8, 2),
                   (8, 9, 1, 2), (7, 8, 1, 3), (8, 3, 2, 3), (1, 10, 5, 4), (0, 1, 10, 3), (10, 2, 7, 2),
                   (3, 2, 15, 2), (9, 3, 2, 2), (2, 4, 10, 5)],
            stations=[(0, 1), (5, 1), (4, 1), (3, 1), (2, 1), (1, 1)],
            start=6,
            friend_start=0,
            expected=(7, 9, [6, 7, 8, 3])
        ),
        # example 2, Unsolvable
        TestIntercept(
            roads=[(0, 1, 35, 3), (1, 2, 5, 2), (2, 0, 35, 4), (0, 4, 10, 1), (4, 1, 22, 2),
                   (1, 5, 65, 1), (5, 2, 70, 1), (2, 3, 10, 1), (3, 0, 20, 3)],
            stations=[(4, 3), (5, 2), (3, 4)],
            start=0,
            friend_start=4,
            expected=None
        ),
        # example 3, Repeated Locations
        TestIntercept(
            roads=[(0, 1, 35, 7), (1, 2, 5, 4), (2, 0, 35, 6), (0, 4, 10, 5), (4, 1, 22, 3),
                   (1, 5, 60, 4), (5, 3, 70, 2), (3, 0, 10, 7)],
            stations=[(4, 2), (5, 1), (3, 4)],
            start=0,
            friend_start=3,
            expected=(160, 39, [0, 1, 2, 0, 1, 2, 0, 4])
        ),
        # example 4, Multiple routes with same cost but different total time
        TestIntercept(
            roads=[(0, 1, 10, 7), (0, 2, 10, 3), (2, 0, 1, 4), (1, 0, 1, 7)],
            stations=[(2, 4), (1, 3)],
            start=0,
            friend_start=1,
            expected=(10, 3, [0, 2])
        ),
        # example 5, mini boss
        TestIntercept(
            roads=[(1, 2, 1, 10), (2, 1, 1, 10), (2, 3, 1, 10), (2, 4, 1, 10)],
            stations=[(3, 55), (4, 55)],
            start=2,
            friend_start=3,
            expected=(11, 110, [2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 3])
        ),
        # example 6, final boss 
        TestIntercept(
            roads=[(0, 1, 1, 1), (1, 0, 1, 1), (1, 2, 1, 1), (1, 3, 1, 1)],
            stations=[(2, 5), (3, 5)],
            start=1,
            friend_start=2,
            expected=(5, 5, [1, 0, 1, 0, 1, 3])
        ),
        # example 7, Lunatic
        TestIntercept(
            roads=[(0, 1, 1, 1), (1, 0, 1, 1), (1, 2, 1, 1), (2, 1, 1, 1), (0, 2, 1, 1), (2, 0, 1, 1), (0, 3, 1, 1),
                   (3, 0, 1, 1), (1, 3, 1, 1), (3, 1, 1, 1), (2, 3, 1, 1), (3, 2, 1, 1), (0, 4, 1, 1), (0, 5, 1, 1),
                   (1, 4, 1, 1), (1, 5, 1, 1), (2, 4, 1, 1), (2, 5, 1, 1), (3, 4, 1, 1), (3, 5, 1, 1), ],
            stations=[(4, 5), (5, 5)],
            start=0,
            friend_start=4,
            expected=(5, 5, [0, 2, 3, 2, 3, 5])
        )
    ]
    failures = []
    for i, test in enumerate(tests):
        if not test.test():
            failures.append(f"Test case {i + 1} failedesddd")
    if failures:
        for failure in failures:
            print(failure)
        raise AssertionError("Some tests failed")
          
if __name__ == '__main__':
    test_intercept()
    unittest.main()