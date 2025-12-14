from Zigong.Environment import Environment

# Ensure current words and neutral words are disjoint
def test_neutral_words_disjoint():
    config = {
        "teams": 1,
        "word_list_file": "test_word_list.txt",
        "max_words": 25
    }
    env = Environment(config)
    team_words = set(env.word_sets[1])
    neutral_words = set(env.neutral_words)
    assert team_words.isdisjoint(neutral_words), "Neutral words overlap with team words"