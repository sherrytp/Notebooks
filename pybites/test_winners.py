from winners import GAME_STATS, print_game_stats

def test_print_game_stats(capfd): 
    prints = ["sara has won 0 games", 
            "bob has won 1 game", 
            "tim has won 5 games", 
            "julian has won 3 games", 
            "jim has won 1 game"]
    print_game_stats(GAME_STATS) 
    output = capfd.readouterr()[0].splitlines()

    for line in prints: 
        assert line in output
